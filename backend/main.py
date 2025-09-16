# main.py  
"""Main entry point for the PM Reporting Tool - debug env vars."""

import os
from pathlib import Path

import uvicorn
from backend.web.app import app

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    # Look for .env file in parent directory (where it actually is)
    env_path = Path(__file__).parent.parent / ".env"
    
    if env_path.exists():
        print(f"Loading .env from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Environment variables loaded")
    else:
        print(f"⚠️  .env file not found at: {env_path}")
        print("   You can create one or set GROQ_API_KEY manually")
        
# Load environment variables before starting the app
load_env()

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
