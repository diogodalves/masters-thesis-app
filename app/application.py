from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.router.router import router as app_router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.domain.sentiment_analyzer import init_inference_pipeline
from app.domain.face_detector import init_haar_cascade
from app.config.config import AppSettings

app_settings = AppSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):

    pipeline = init_inference_pipeline(
        model_path=app_settings.best_model_path,
        label_encoder_path=app_settings.label_encoder_path
        )

    haar_cascade = init_haar_cascade(
        app_settings.haar_cascade_path
    )

    yield

def start_app() -> FastAPI:
    """Start the FastAPI application.

    Returns:
        FastAPI: The FastAPI application.
    """
    app = FastAPI(lifespan=lifespan)

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[""],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(app_router)

    return app

app = start_app()
