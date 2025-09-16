import asyncio
import gc
import json
import logging
import os
from datetime import datetime
from typing import Optional

import markdown
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# SQLAlchemy now handled by scalable repository in data package

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Simple API key auth for production (optional)
security = HTTPBearer(auto_error=False)

def get_api_key() -> Optional[str]:
    """Get API key from environment if set"""
    return os.environ.get("API_KEY")

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if set in environment"""
    api_key = get_api_key()
    
    # If no API key is configured, allow access
    if not api_key:
        return True
        
    # If API key is configured but no credentials provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Verify the API key
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True

app = FastAPI(
    # Add file upload size limit (5MB)
    docs_url="/docs" if os.environ.get("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.environ.get("ENVIRONMENT") != "production" else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware to check file upload sizes and manage memory
@app.middleware("http")
async def limit_upload_size_and_manage_memory(request: Request, call_next):
    # Check if this is a file upload request
    if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            # 5MB limit for file uploads
            if content_length > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")
    
    response = await call_next(request)
    
    # Force garbage collection after memory-intensive operations in low memory mode
    from backend.core import settings
    if settings.FORCE_GC_AFTER_REQUESTS and settings.LOW_MEMORY_MODE:
        if request.url.path in ["/generate_report", "/intelligent_ask", "/sync_vector_db"]:
            gc.collect()
    
    return response


# Add health check endpoint for Render
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Add database health check endpoint
@app.get("/health/database")  
async def database_health_check():
    try:
        from backend.data import DatabaseInitializer
        
        db_info = DatabaseInitializer.get_database_info()
        return {
            "status": "healthy" if db_info["connection_status"] == "Connected" else "unhealthy",
            "database_info": db_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# Add Markdown filter to Jinja2
def markdown_filter(text):
    """Convert Markdown to HTML"""
    return markdown.markdown(text, extensions=["nl2br"])


templates.env.filters["markdown"] = markdown_filter

# Note: Database setup now handled by scalable repository in data package


# --- Lazy initialization for LLM, reporting tool, vector service, and query router ---
_llm = None
_tool = None
_vector_service = None
_query_router = None


def get_llm():
    global _llm
    if _llm is None:
        logger.info("=== DEBUG: About to initialize LLM ===")
        import os
        logger.info(f"GROQ_API_KEY in environment: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
        logger.info("Initializing LLM...")
        from backend.providers.llm_providers import create_llm

        _llm = create_llm()
        logger.info("LLM initialized successfully")
    return _llm


def get_reporting_tool():
    global _tool
    if _tool is None:
        logger.info("Initializing reporting tool...")
        from backend.services.scalable_reporting_tool import ScalableReportingTool
        from backend.data import SQLAlchemyUpdateRepository

        # Create SQLAlchemy repository and scalable reporting tool
        repository = SQLAlchemyUpdateRepository()
        _tool = ScalableReportingTool(llm=get_llm(), repository=repository)
        logger.info("Reporting tool initialized successfully")
    return _tool


def get_vector_service():
    global _vector_service
    if _vector_service is None:
        # Check if vector DB is enabled (disabled by default in low memory mode)
        from backend.core import settings
        if not settings.ENABLE_VECTOR_DB:
            logger.info("Vector service disabled in low memory mode")
            return None
            
        logger.info("Initializing vector service...")
        from backend.services.vector_service import VectorService

        _vector_service = VectorService()
        logger.info("Vector service initialized successfully")
    return _vector_service


def get_query_router():
    global _query_router
    if _query_router is None:
        logger.info("Initializing query router...")
        from backend.query_handlers.router import SmartQueryRouter

        _query_router = SmartQueryRouter(llm=get_llm())
        logger.info("Query router initialized successfully")
    return _query_router


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI app starting up...")
    
    # Initialize database with SQLAlchemy
    try:
        from backend.data import init_database
        
        init_database()
        logger.info("Database initialized with SQLAlchemy and indexes")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # Initialize entity configuration
    try:
        from backend.query_handlers.entity_config import initialize_entity_config

        entity_config = initialize_entity_config()
        logger.info(
            f"Entity config loaded: {len(entity_config.teams)} teams, {len(entity_config.technologies)} technologies"
        )
    except Exception as e:
        logger.warning(f"Could not load entity config: {str(e)}, using defaults")

    logger.info("App startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI app shutting down...")


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/updates", response_class=HTMLResponse)
def updates_page(request: Request, employee: str = None, department: str = None):
    tool = get_reporting_tool()
    
    # Apply filters if provided - use memory-optimized limits
    from backend.core import settings
    max_limit = settings.MAX_UPDATES_IN_MEMORY if settings.LOW_MEMORY_MODE else 100
    
    if employee or department:
        if employee:
            # Use database-level filtering for employee with limit
            updates = tool.repository.get_by_employee(employee, limit=max_limit)
        elif department:
            # For department filtering, use a smaller limit and database query if possible
            try:
                # Try database-level filtering if repository supports it
                updates = tool.repository.get_by_department(department, limit=max_limit)
            except (AttributeError, NotImplementedError):
                # Fallback to in-memory filtering with smaller dataset
                all_updates = tool.repository.get_recent(limit=max_limit)
                updates = [u for u in all_updates if 
                          u.department and u.department.lower() == department.lower()][:25]
    else:
        # No filters - show recent updates
        updates = tool.repository.get_recent(limit=max_limit)
    
    # Get filter options - limit in low memory mode
    if settings.LOW_MEMORY_MODE:
        # In low memory mode, get limited filter options
        available_employees = tool.repository.get_all_employee_names()[:settings.MAX_FILTER_EMPLOYEES]
        available_departments = tool.repository.get_unique_departments()[:settings.MAX_FILTER_DEPARTMENTS]
        
        # Force garbage collection after loading data
        import gc
        gc.collect()
    else:
        available_employees = tool.repository.get_all_employee_names()
        available_departments = tool.repository.get_unique_departments()
    
    return templates.TemplateResponse(
        "updates.html", {
            "request": request, 
            "updates": updates,
            "available_employees": available_employees,
            "available_departments": available_departments,
            "selected_employee": employee,
            "selected_department": department
        }
    )


@app.get("/manage_updates", response_class=HTMLResponse)
def manage_updates_page(request: Request):
    # Get available roles from database
    tool = get_reporting_tool()
    available_roles = tool.repository.get_unique_roles()
    
    return templates.TemplateResponse(
        "manage_updates.html", 
        {"request": request, "available_roles": available_roles}
    )


@app.get("/generate_report", response_class=HTMLResponse)
def generate_report_form(request: Request):
    return templates.TemplateResponse("report_form.html", {"request": request})


@app.get("/get_default_prompt")
async def get_default_prompt():
    """Get the default prompt for report generation"""
    try:
        tool = get_reporting_tool()
        prompt = tool.get_default_prompt_preview()
        return {"prompt": prompt, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting default prompt: {str(e)}")
        return {"error": str(e), "status": "error"}


@app.post("/generate_report", response_class=HTMLResponse)
@limiter.limit("5/minute")  # Limit report generation to prevent abuse
async def generate_report(
    request: Request,
    start_date: str = Form(...),
    end_date: str = Form(...),
    custom_prompt: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key)  # Optional API key auth
):
    try:
        # Set a timeout for the entire operation
        async def generate_report_with_timeout():
            logger.info(f"Starting report generation for {start_date} to {end_date}")
            tool = get_reporting_tool()

            # Parse dates for date range
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            date_range = (start_date_obj, end_date_obj)

            logger.info("Generating report with scalable method...")

            # Use scalable method with memory-optimized limits
            from backend.core import settings
            max_updates = settings.MAX_REPORT_UPDATES if settings.LOW_MEMORY_MODE else 100
            
            report = tool.generate_smart_report(
                date_range=date_range,
                custom_prompt=custom_prompt.strip()
                if custom_prompt and custom_prompt.strip()
                else None,
                max_updates=max_updates
            )
            logger.info("Report generation completed")

            # Force garbage collection to free memory
            gc.collect()

            return report

        # Wait for report generation with 50-second timeout (Render has ~60s limit)
        report = await asyncio.wait_for(generate_report_with_timeout(), timeout=50.0)

    except asyncio.TimeoutError:
        logger.error("Report generation timed out")
        return templates.TemplateResponse(
            "report.html",
            {
                "request": request,
                "report": "⚠️ Report generation timed out. This may be due to a cold start on the free tier. Please try again in a moment.",
                "start_date": start_date,
                "end_date": end_date,
                "error": True,
            },
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return templates.TemplateResponse(
            "report.html",
            {
                "request": request,
                "report": f"❌ Error generating report: {str(e)}. If this persists, the service may be starting up.",
                "start_date": start_date,
                "end_date": end_date,
                "error": True,
            },
        )
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "report": report,
            "start_date": start_date,
            "end_date": end_date,
        },
    )


@app.post("/submit_update", response_class=HTMLResponse)
def submit_update(
    request: Request,
    employee: str = Form(...),
    role: str = Form(...),
    update: str = Form(...),
    date: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    manager: Optional[str] = Form(None),
):
    if not date:
        from backend.core.config import Config
        date = datetime.now().strftime(Config.DATE_FORMAT)

    # Use scalable repository for consistent architecture
    from backend.core.models import Update
    
    update_obj = Update(
        employee=employee, 
        role=role, 
        date=date, 
        update=update,
        department=department if department else None,
        manager=manager if manager else None
    )
    
    # Add to main repository
    tool = get_reporting_tool()
    tool.add_update(update_obj)

    # Also add to vector database (if enabled)
    try:
        vector_service = get_vector_service()
        if vector_service is not None:
            vector_service.add_update(update_obj)
    except Exception as e:
        logger.warning(f"Failed to add update to vector DB: {str(e)}")

    return RedirectResponse("/manage_updates", status_code=303)


@app.post("/bulk_upload", response_class=HTMLResponse)
@limiter.limit("10/hour")  # Limit bulk uploads
async def bulk_upload(request: Request, file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()

        # Parse JSON
        try:
            updates_data = json.loads(contents)
        except json.JSONDecodeError:
            return templates.TemplateResponse(
                "manage_updates.html",
                {
                    "request": request,
                    "error": "Invalid JSON file. Please check the file format.",
                },
            )

        # Validate that it's a list
        if not isinstance(updates_data, list):
            return templates.TemplateResponse(
                "manage_updates.html",
                {
                    "request": request,
                    "error": "JSON file must contain an array of update objects.",
                },
            )

        # Validate each update has required fields
        required_fields = ["employee", "role", "date", "update"]
        
        # Convert to Update objects for validation
        from backend.core.models import Update
        updates_to_add = []
        
        for i, update_data in enumerate(updates_data):
            if not isinstance(update_data, dict):
                return templates.TemplateResponse(
                    "manage_updates.html",
                    {
                        "request": request,
                        "error": f"Invalid update format at index {i}. Each update must be an object.",
                    },
                )

            # Check required fields
            missing_fields = [
                field for field in required_fields if field not in update_data
            ]
            if missing_fields:
                return templates.TemplateResponse(
                    "manage_updates.html",
                    {
                        "request": request,
                        "error": f"Missing required fields at index {i}: {', '.join(missing_fields)}",
                    },
                )

            # Create Update object
            update = Update(
                employee=str(update_data["employee"]),
                role=str(update_data["role"]),
                date=str(update_data["date"]),
                update=str(update_data["update"]),
                department=str(update_data["department"]) if update_data.get("department") else None,
                manager=str(update_data["manager"]) if update_data.get("manager") else None
            )
            updates_to_add.append(update)

        # Use scalable repository for bulk insert
        tool = get_reporting_tool()
        tool.add_updates(updates_to_add)
        added_count = len(updates_to_add)

        return templates.TemplateResponse(
            "manage_updates.html",
            {
                "request": request,
                "success": f"Successfully added {added_count} updates from {file.filename}",
            },
        )

    except Exception as e:
        logger.error(f"Error processing bulk upload: {str(e)}")
        return templates.TemplateResponse(
            "manage_updates.html",
            {"request": request, "error": f"Error processing file: {str(e)}"},
        )


@app.post("/ask", response_class=HTMLResponse)
@limiter.limit("20/minute")  # Limit Q&A requests
def ask(request: Request, question: str = Form(...)):
    tool = get_reporting_tool()
    
    # Use scalable method that works directly with repository - no manual loading needed
    answer = tool.answer_contextual_question(question, max_updates=100)
    return templates.TemplateResponse(
        "answer.html", {"request": request, "question": question, "answer": answer}
    )


@app.get("/stats", response_class=HTMLResponse)
def stats(request: Request):
    tool = get_reporting_tool()
    
    # Use repository stats directly instead of manually loading all updates
    stats = tool.repository.get_stats()

    # Add vector database stats (if enabled)
    try:
        vector_service = get_vector_service()
        if vector_service is not None:
            vector_stats = vector_service.get_collection_stats()
            stats["vector_db"] = vector_stats
        else:
            stats["vector_db"] = {"status": "disabled", "reason": "Low memory mode"}
    except Exception as e:
        logger.warning(f"Could not get vector DB stats: {str(e)}")
        stats["vector_db"] = {"error": str(e)}

    return templates.TemplateResponse(
        "stats.html", {"request": request, "stats": stats}
    )


@app.get("/api/employees")
async def get_employees_api():
    """API endpoint to get all employee names for autocomplete"""
    try:
        tool = get_reporting_tool()
        employees = tool.repository.get_all_employee_names()
        return {"employees": employees}
    except Exception as e:
        return {"error": str(e)}


@app.get("/load_mock_data", response_class=HTMLResponse)
def load_mock_data(request: Request):
    # Use scalable repository and DataLoader
    from backend.services.data_loader import DataLoader
    
    tool = get_reporting_tool()
    tool.clear_updates()  # Clear existing data
    
    # Load mock data using DataLoader
    mock_updates = DataLoader.get_mock_updates()
    tool.add_updates(mock_updates)
    
    return RedirectResponse("/updates", status_code=303)


@app.get("/import_mock_data", response_class=HTMLResponse)
def import_mock_data(request: Request):
    from backend.services.data_loader import DataLoader

    # Import all mock updates using scalable repository
    tool = get_reporting_tool()
    updates = DataLoader.get_mock_updates()
    tool.add_updates(updates)
    
    return RedirectResponse("/updates", status_code=303)


@app.get("/import_additional_mock_data", response_class=HTMLResponse)
def import_additional_mock_data(request: Request):
    from backend.services.data_loader import DataLoader

    # Import additional mock updates using scalable repository
    tool = get_reporting_tool()
    updates = DataLoader.get_additional_mock_updates()
    tool.add_updates(updates)
    
    return RedirectResponse("/updates", status_code=303)


@app.get("/import_mock_updates_with_blockers", response_class=HTMLResponse)
def import_mock_updates_with_blockers(request: Request):
    from backend.services.data_loader import DataLoader

    # Import mock updates with blockers using scalable repository
    tool = get_reporting_tool()
    updates = DataLoader.get_mock_updates_with_blockers()
    tool.add_updates(updates)
    
    return RedirectResponse("/updates", status_code=303)


@app.get("/delete_all_updates", response_class=HTMLResponse)
def delete_all_updates(request: Request):
    # Clear all updates using scalable repository
    tool = get_reporting_tool()
    tool.clear_updates()

    # Also clear vector database (if enabled)
    vector_service = get_vector_service()
    if vector_service is not None:
        vector_service.clear_collection()

    return RedirectResponse("/updates", status_code=303)


@app.get("/intelligent_ask_page", response_class=HTMLResponse)
def intelligent_ask_page(request: Request):
    return templates.TemplateResponse("intelligent_ask.html", {"request": request})


@app.post("/intelligent_ask_page", response_class=HTMLResponse)
@limiter.limit("15/minute")  # Limit intelligent queries
async def intelligent_ask_page_results(
    request: Request,
    question: str = Form(...),
    method_override: Optional[str] = Form(None),
):
    try:
        # Get services (no need to manually load updates)
        tool = get_reporting_tool()
        vector_service = get_vector_service()
        router = get_query_router()
        
        # Skip intelligent routing if vector service is disabled
        if vector_service is None:
            return templates.TemplateResponse(
                "intelligent_ask.html",
                {
                    "request": request,
                    "question": question,
                    "answer": "Intelligent query routing is disabled in low memory mode. Please use the basic Q&A instead.",
                    "method_used": "disabled",
                    "confidence": 0,
                },
            )

        # Route the query
        response = router.route_query(
            query=question,
            reporting_tool=tool,
            vector_service=vector_service,
            db_session=None,  # Not needed with scalable repository
            override_method=method_override if method_override else None,
        )

        return templates.TemplateResponse(
            "intelligent_ask.html",
            {
                "request": request,
                "question": question,
                "answer": response.answer,
                "method_used": response.method_used,
                "confidence": response.confidence,
                "classification": response.additional_info.get("classification")
                if response.additional_info
                else None,
                "routing_info": response.additional_info.get("routing_info")
                if response.additional_info
                else None,
            },
        )

    except Exception as e:
        logger.error(f"Intelligent ask page error: {str(e)}")
        return templates.TemplateResponse(
            "intelligent_ask.html",
            {"request": request, "question": question, "error": str(e)},
        )


@app.post("/sync_vector_db")
async def sync_vector_database():
    """Sync all updates from scalable repository to vector database"""
    try:
        vector_service = get_vector_service()
        tool = get_reporting_tool()

        # Get all updates from scalable repository
        updates = tool.repository.get_all(limit=1000)  # Reasonable limit

        # Clear and repopulate vector database
        vector_service.clear_collection()
        synced_count = vector_service.add_updates_batch(updates)

        return {
            "message": f"Synced {synced_count} updates to vector database",
            "total_repository_updates": len(updates),
            "synced_updates": synced_count,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Vector sync error: {str(e)}")
        return {"error": str(e), "status": "error"}


@app.post("/intelligent_ask")
@limiter.limit("15/minute")  # Limit intelligent queries  
async def intelligent_ask(
    request: Request,
    question: str = Form(...),
    method_override: Optional[str] = Form(None),
):
    """Intelligent query routing that automatically selects the best approach"""
    try:
        # Get services (no need to manually load updates)
        tool = get_reporting_tool()
        vector_service = get_vector_service()
        router = get_query_router()
        
        # Use basic Q&A if vector service is disabled
        if vector_service is None:
            answer = tool.answer_contextual_question(question, max_updates=25)
            return {
                "question": question,
                "answer": answer,
                "method_used": "basic_qa",
                "confidence": 0.7,
                "status": "success",
            }

        # Route the query
        response = router.route_query(
            query=question,
            reporting_tool=tool,
            vector_service=vector_service,
            db_session=None,  # Not needed with scalable repository
            override_method=method_override,
        )

        return {
            "question": question,
            "answer": response.answer,
            "method_used": response.method_used,
            "confidence": response.confidence,
            "classification": response.additional_info.get("classification").__dict__
            if response.additional_info and "classification" in response.additional_info
            else None,
            "routing_info": response.additional_info.get("routing_info")
            if response.additional_info
            else None,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Intelligent ask error: {str(e)}")
        return {"error": str(e), "question": question, "status": "error"}


@app.post("/classify_query")
async def classify_query(question: str = Form(...)):
    """Classify a query to understand how it would be routed"""
    try:
        router = get_query_router()
        classification = router.classify_query(question)

        return {
            "question": question,
            "classification": classification.__dict__,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Query classification error: {str(e)}")
        return {"error": str(e), "question": question, "status": "error"}


@app.get("/download_entity_config")
async def download_entity_config():
    """Download the current entity configuration file"""
    try:
        from backend.query_handlers.entity_config import get_entity_config

        config = get_entity_config()
        config_path = config.config_path

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return RedirectResponse(
                "/manage_updates?error=Config file not found", status_code=303
            )

        return FileResponse(
            path=config_path, filename="entities.yaml", media_type="application/x-yaml"
        )
    except Exception as e:
        logger.error(f"Error downloading config: {str(e)}")
        return RedirectResponse(
            f"/manage_updates?error=Error downloading config: {str(e)}", status_code=303
        )


@app.post("/upload_entity_config")
async def upload_entity_config(config_file: UploadFile = File(...)):
    """Upload a new entity configuration file"""
    try:
        # Validate file type
        if not config_file.filename.endswith((".yaml", ".yml")):
            return RedirectResponse(
                "/manage_updates?error=Please upload a YAML file (.yaml or .yml)",
                status_code=303,
            )

        # Read and validate file content
        contents = await config_file.read()

        try:
            import yaml

            yaml_data = yaml.safe_load(contents.decode("utf-8"))
        except yaml.YAMLError as e:
            return RedirectResponse(
                f"/manage_updates?error=Invalid YAML format: {str(e)}", status_code=303
            )
        except UnicodeDecodeError:
            return RedirectResponse(
                "/manage_updates?error=File must be UTF-8 encoded", status_code=303
            )

        # Validate required sections
        required_sections = ["teams", "technologies", "projects", "focus_areas"]
        missing_sections = [
            section for section in required_sections if section not in yaml_data
        ]

        if missing_sections:
            return RedirectResponse(
                f"/manage_updates?error=Missing required sections: {', '.join(missing_sections)}",
                status_code=303,
            )

        # Validate teams structure (basic validation)
        try:
            teams = yaml_data.get("teams", [])
            for i, team in enumerate(teams):
                if not isinstance(team, dict):
                    raise ValueError(f"Team {i} must be an object")
                if "name" not in team:
                    raise ValueError(f"Team {i} missing required 'name' field")
                if "aliases" not in team or not isinstance(team["aliases"], list):
                    raise ValueError(f"Team {i} missing required 'aliases' list")
        except ValueError as e:
            return RedirectResponse(
                f"/manage_updates?error=Invalid teams structure: {str(e)}",
                status_code=303,
            )

        # Save the new configuration
        from backend.query_handlers.entity_config import get_entity_config

        config = get_entity_config()

        # Create backup of current config
        backup_path = config.config_path + ".backup"
        if os.path.exists(config.config_path):
            import shutil

            shutil.copy2(config.config_path, backup_path)
            logger.info(f"Created backup at {backup_path}")

        # Write new config
        with open(config.config_path, "w", encoding="utf-8") as f:
            f.write(contents.decode("utf-8"))

        # Reload configuration
        config.reload_config()

        # Update query router with new config
        global _query_router
        if _query_router:
            _query_router.entity_config = config
            logger.info("Updated query router with new entity configuration")

        logger.info(
            f"Successfully updated entity configuration from {config_file.filename}"
        )

        return RedirectResponse(
            f"/manage_updates?success=Successfully uploaded and applied new entity configuration with {len(yaml_data.get('teams', []))} teams and {len(yaml_data.get('technologies', []))} technologies",
            status_code=303,
        )

    except Exception as e:
        logger.error(f"Error uploading config: {str(e)}")
        return RedirectResponse(
            f"/manage_updates?error=Error uploading config: {str(e)}", status_code=303
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
