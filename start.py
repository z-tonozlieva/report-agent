#!/usr/bin/env python3
"""
Wrapper script to start the ReportAgent app from the correct directory.
This solves the module import issue when deploying to Render.
"""
import os
import sys
import subprocess

def main():
    # Change to the backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    # Get the port from environment variable (Render sets this)
    port = os.environ.get('PORT', '8000')
    
    # Start uvicorn from the backend directory
    cmd = [
        sys.executable, '-m', 'uvicorn', 
        'main:app', 
        '--host', '0.0.0.0', 
        '--port', port
    ]
    
    print(f"Starting ReportAgent from {backend_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()