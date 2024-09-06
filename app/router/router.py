from fastapi import APIRouter
from controller.camera import websocket_endpoint
from controller.health import health
from fastapi.responses import FileResponse

router = APIRouter()

router.add_api_route("/health", health, methods=["GET"])
router.websocket("/camera/ws")(websocket_endpoint)
router.add_api_route("/camera", lambda: FileResponse('static/index.html'), methods=["GET"])