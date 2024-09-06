from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from router.router import router as app_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(app_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)