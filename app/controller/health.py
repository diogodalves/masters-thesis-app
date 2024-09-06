from fastapi import APIRouter

router = APIRouter()

async def health():
    return {"status": "healthy", "message": "Service is up and running!"}