from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
from datetime import datetime
import uvicorn

# Import business logic and models
from reporting_tool import PMReportingTool
from models import Update
from free_llm_providers import create_llm
from config import Config
from data_loader import DataLoader
from mock_llm import LLMInterface

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

# --- LLM and reporting tool ---
llm: LLMInterface = create_llm("ollama")
tool = PMReportingTool(llm=llm)

@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/updates", response_class=HTMLResponse)
def updates_page(request: Request, db: Session = Depends(get_db)):
    updates = db.query(UpdateDB).order_by(UpdateDB.id.desc()).all()
    return templates.TemplateResponse("updates.html", {"request": request, "updates": updates})

@app.get("/generate_report", response_class=HTMLResponse)
def generate_report_form(request: Request):
    return templates.TemplateResponse("report_form.html", {"request": request})

@app.post("/generate_report", response_class=HTMLResponse)
def generate_report(request: Request, start_date: str = Form(...), end_date: str = Form(...), report_type: str = Config.DEFAULT_REPORT_TYPE.value, db: Session = Depends(get_db)):
    # Filter updates by date range
    updates = db.query(UpdateDB).filter(UpdateDB.date >= start_date, UpdateDB.date <= end_date).all()
    tool.clear_updates()
    for u in updates:
        tool.add_update(Update(employee=str(u.employee), role=str(u.role), date=str(u.date), update=str(u.update)))
    report = tool.generate_report(report_type=report_type)
    return templates.TemplateResponse("report.html", {"request": request, "report": report, "start_date": start_date, "end_date": end_date})

@app.post("/submit_update", response_class=HTMLResponse)
def submit_update(request: Request, employee: str = Form(...), role: str = Form(...), update: str = Form(...), date: Optional[str] = Form(None), db: Session = Depends(get_db)):
    if not date:
        date = datetime.now().strftime(Config.DATE_FORMAT)
    db_update = UpdateDB(employee=employee, role=role, date=date, update=update)
    db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...), db: Session = Depends(get_db)):
    updates = db.query(UpdateDB).all()
    tool.clear_updates()
    for u in updates:
        tool.add_update(Update(employee=str(u.employee), role=str(u.role), date=str(u.date), update=str(u.update)))
    answer = tool.answer_stakeholder_question(question)
    return templates.TemplateResponse("answer.html", {"request": request, "question": question, "answer": answer})

@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request, db: Session = Depends(get_db)):
    updates = db.query(UpdateDB).all()
    tool.clear_updates()
    for u in updates:
        tool.add_update(Update(employee=str(u.employee), role=str(u.role), date=str(u.date), update=str(u.update)))
    stats = tool.get_summary_stats()
    return templates.TemplateResponse("stats.html", {"request": request, "stats": stats})

@app.get("/load_mock_data", response_class=HTMLResponse)
def load_mock_data(request: Request, db: Session = Depends(get_db)):
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
    # Import all mock updates from DataLoader
    updates = DataLoader.get_mock_updates()
    for u in updates:
        db_update = UpdateDB(employee=u.employee, role=u.role, date=u.date, update=u.update)
        db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.get("/import_additional_mock_data", response_class=HTMLResponse)
def import_additional_mock_data(request: Request, db: Session = Depends(get_db)):
    updates = DataLoader.get_additional_mock_updates()
    for u in updates:
        db_update = UpdateDB(employee=u.employee, role=u.role, date=u.date, update=u.update)
        db.add(db_update)
    db.commit()
    return RedirectResponse("/updates", status_code=303)

@app.get("/import_mock_updates_with_blockers", response_class=HTMLResponse)
def import_mock_updates_with_blockers(request: Request, db: Session = Depends(get_db)):
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