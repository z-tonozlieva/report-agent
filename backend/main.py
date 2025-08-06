from fastapi import FastAPI, Request, Form, Depends, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
from datetime import datetime
import uvicorn
import os
import logging
import json
import gc
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add health check endpoint for Render
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- SQLAlchemy setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./updates.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UpdateDB(Base):
    __tablename__ = "updates"
    id = Column(Integer, primary_key=True, index=True)
    employee = Column(String, index=True)
    role = Column(String)
    date = Column(String)
    update = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Lazy initialization for LLM and reporting tool ---
_llm = None
_tool = None

def get_llm():
    global _llm
    if _llm is None:
        logger.info("Initializing LLM...")
        from .free_llm_providers import create_llm
        _llm = create_llm("groq")
        logger.info("LLM initialized successfully")
    return _llm

def get_reporting_tool():
    global _tool
    if _tool is None:
        logger.info("Initializing reporting tool...")
        from .reporting_tool import PMReportingTool
        _tool = PMReportingTool(llm=get_llm())
        logger.info("Reporting tool initialized successfully")
    return _tool

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI app starting up...")
    logger.info("Database tables created")
    logger.info("App startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI app shutting down...")

@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/updates", response_class=HTMLResponse)
def updates_page(request: Request, db: Session = Depends(get_db)):
    updates = db.query(UpdateDB).order_by(UpdateDB.id.desc()).all()
    return templates.TemplateResponse("updates.html", {"request": request, "updates": updates})

@app.get("/manage_updates", response_class=HTMLResponse)
def manage_updates_page(request: Request):
    return templates.TemplateResponse("manage_updates.html", {"request": request})

@app.get("/generate_report", response_class=HTMLResponse)
def generate_report_form(request: Request):
    return templates.TemplateResponse("report_form.html", {"request": request})

@app.post("/generate_report", response_class=HTMLResponse)
async def generate_report(
    request: Request, 
    start_date: str = Form(...), 
    end_date: str = Form(...), 
    report_type: str = Form("weekly"),
    custom_prompt: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        # Set a timeout for the entire operation
        async def generate_report_with_timeout():
            # Lazy import to avoid startup delays
            from .config import Config
            from .models import Update
            
            logger.info(f"Starting report generation for {start_date} to {end_date}")
            tool = get_reporting_tool()
            
            # Filter updates by date range
            updates = db.query(UpdateDB).filter(UpdateDB.date >= start_date, UpdateDB.date <= end_date).all()
            logger.info(f"Found {len(updates)} updates for report")
            
            tool.clear_updates()
            for u in updates:
                tool.add_update(Update(employee=str(u.employee), role=str(u.role), date=str(u.date), update=str(u.update)))
            
            logger.info("Generating report with LLM...")
            report = tool.generate_report(report_type=report_type, custom_prompt=custom_prompt.strip() if custom_prompt else None)
            logger.info("Report generation completed")
            
            # Force garbage collection to free memory
            gc.collect()
            
            return report
        
        # Wait for report generation with 50-second timeout (Render has ~60s limit)
        report = await asyncio.wait_for(generate_report_with_timeout(), timeout=50.0)
        
    except asyncio.TimeoutError:
        logger.error("Report generation timed out")
        return templates.TemplateResponse("report.html", {
            "request": request, 
            "report": "⚠️ Report generation timed out. This may be due to a cold start on the free tier. Please try again in a moment.", 
            "start_date": start_date, 
            "end_date": end_date,
            "error": True
        })
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return templates.TemplateResponse("report.html", {
            "request": request, 
            "report": f"❌ Error generating report: {str(e)}. If this persists, the service may be starting up.", 
            "start_date": start_date, 
            "end_date": end_date,
            "error": True
        })
    return templates.TemplateResponse("report.html", {
        "request": request, 
        "report": report, 
        "start_date": start_date, 
        "end_date": end_date
    })

@app.post("/submit_update", response_class=HTMLResponse)
def submit_update(
    request: Request, 
    employee: str = Form(...), 
    role: str = Form(...), 
    update: str = Form(...), 
    date: Optional[str] = Form(None), 
    db: Session = Depends(get_db)
):
    if not date:
        from .config import Config
        date = datetime.now().strftime(Config.DATE_FORMAT)
    
    db_update = UpdateDB(employee=employee, role=role, date=date, update=update)
    db.add(db_update)
    db.commit()
    return RedirectResponse("/manage_updates", status_code=303)

@app.post("/bulk_upload", response_class=HTMLResponse)
async def bulk_upload(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Parse JSON
        try:
            updates_data = json.loads(contents)
        except json.JSONDecodeError:
            return templates.TemplateResponse("manage_updates.html", {
                "request": request, 
                "error": "Invalid JSON file. Please check the file format."
            })
        
        # Validate that it's a list
        if not isinstance(updates_data, list):
            return templates.TemplateResponse("manage_updates.html", {
                "request": request, 
                "error": "JSON file must contain an array of update objects."
            })
        
        # Validate each update has required fields
        required_fields = ["employee", "role", "date", "update"]
        added_count = 0
        
        for i, update_data in enumerate(updates_data):
            if not isinstance(update_data, dict):
                return templates.TemplateResponse("manage_updates.html", {
                    "request": request, 
                    "error": f"Invalid update format at index {i}. Each update must be an object."
                })
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in update_data]
            if missing_fields:
                return templates.TemplateResponse("manage_updates.html", {
                    "request": request, 
                    "error": f"Missing required fields at index {i}: {', '.join(missing_fields)}"
                })
            
            # Add to database
            db_update = UpdateDB(
                employee=str(update_data["employee"]),
                role=str(update_data["role"]),
                date=str(update_data["date"]),
                update=str(update_data["update"])
            )
            db.add(db_update)
            added_count += 1
        
        db.commit()
        
        return templates.TemplateResponse("manage_updates.html", {
            "request": request, 
            "success": f"Successfully added {added_count} updates from {file.filename}"
        })
        
    except Exception as e:
        logger.error(f"Error processing bulk upload: {str(e)}")
        return templates.TemplateResponse("manage_updates.html", {
            "request": request, 
            "error": f"Error processing file: {str(e)}"
        })

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...), db: Session = Depends(get_db)):
    from .models import Update
    
    tool = get_reporting_tool()
    updates = db.query(UpdateDB).all()
    tool.clear_updates()
    for u in updates:
        tool.add_update(Update(employee=str(u.employee), role=str(u.role), date=str(u.date), update=str(u.update)))
    
    answer = tool.answer_stakeholder_question(question)
    return templates.TemplateResponse("answer.html", {
        "request": request, 
        "question": question, 
        "answer": answer
    })

@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request, db: Session = Depends(get_db)):
    from .models import Update
    
    tool = get_reporting_tool()
    updates = db.query(UpdateDB).all()
    tool.clear_updates()
    for u in updates:
        tool.add_update(Update(employee=str(u.employee), role=str(u.role), date=str(u.date), update=str(u.update)))
    
    stats = tool.get_summary_stats()
    return templates.TemplateResponse("stats.html", {"request": request, "stats": stats})

@app.get("/load_mock_data", response_class=HTMLResponse)
def load_mock_data(request: Request, db: Session = Depends(get_db)):
    tool = get_reporting_tool()
    db.query(UpdateDB).delete()
    db.commit()
    tool.clear_updates()
    tool.load_mock_data()
    for u in tool.updates:
        db_update = UpdateDB(employee=u.employee, role=u.role, date=u.date, update=u.update)
        db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.get("/import_mock_data", response_class=HTMLResponse)
def import_mock_data(request: Request, db: Session = Depends(get_db)):
    from .data_loader import DataLoader
    
    # Import all mock updates from DataLoader
    updates = DataLoader.get_mock_updates()
    for u in updates:
        db_update = UpdateDB(employee=u.employee, role=u.role, date=u.date, update=u.update)
        db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.get("/import_additional_mock_data", response_class=HTMLResponse)
def import_additional_mock_data(request: Request, db: Session = Depends(get_db)):
    from .data_loader import DataLoader
    
    updates = DataLoader.get_additional_mock_updates()
    for u in updates:
        db_update = UpdateDB(employee=u.employee, role=u.role, date=u.date, update=u.update)
        db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.get("/import_mock_updates_with_blockers", response_class=HTMLResponse)
def import_mock_updates_with_blockers(request: Request, db: Session = Depends(get_db)):
    from .data_loader import DataLoader
    
    updates = DataLoader.get_mock_updates_with_blockers()
    for u in updates:
        db_update = UpdateDB(employee=u.employee, role=u.role, date=u.date, update=u.update)
        db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.get("/delete_all_updates", response_class=HTMLResponse)
def delete_all_updates(request: Request, db: Session = Depends(get_db)):
    db.query(UpdateDB).delete()
    db.commit()
    return RedirectResponse("/updates", status_code=303)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)