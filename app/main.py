"""Main module for the FastAPI application."""
import uvicorn
from app.application import app

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001, log_level='info')
