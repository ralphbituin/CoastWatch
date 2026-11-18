"""WebSocket endpoint for real-time dashboard updates.

Phase 1 provides the connection plumbing (connect, heartbeat ping/pong, and
broadcast helper). Later phases push shoreline observations, tide updates and
anomaly events through :data:`manager`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Tracks connected dashboard clients and broadcasts JSON messages."""

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> int:
        """Send ``message`` to every client; returns the number delivered."""
        delivered = 0
        for websocket in list(self.active):
            try:
                await websocket.send_json(message)
                delivered += 1
            except Exception:  # pragma: no cover - dropped clients
                self.disconnect(websocket)
        return delivered


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Dashboard WebSocket.

    Protocol (Phase 1): the server greets the client, then answers ``ping``
    with ``pong`` and echoes any other text message.
    """
    await manager.connect(websocket)
    try:
        await websocket.send_json(
            {
                "type": "welcome",
                "service": "coastwatch",
                "timestamp": datetime.now(UTC).isoformat(),
                "message": "Connected to CoastWatch realtime channel.",
            }
        )
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": datetime.now(UTC).isoformat()}
                )
            else:
                await websocket.send_json({"type": "echo", "data": message})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:  # pragma: no cover - defensive
        logger.exception("websocket_error")
        manager.disconnect(websocket)
