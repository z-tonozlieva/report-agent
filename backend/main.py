# main.py
"""Main entry point for the PM Reporting Tool."""

import uvicorn
from web.app import app

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
