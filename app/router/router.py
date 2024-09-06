from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.controller.camera import websocket_endpoint
from app.controller.health import health

router = APIRouter()

router.add_api_route("/health", health, methods=["GET"])
router.websocket("/camera/ws")(websocket_endpoint)
router.add_api_route("/camera", lambda: FileResponse('app/static/index.html'), methods=["GET"])