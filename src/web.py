"""FastAPI application and helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from state import SharedState

# Setup paths
APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app(state: SharedState) -> FastAPI:
    app = FastAPI(title="Duckiebot Control", version="0.2.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _serve_frame(fetcher) -> Response:
        frame = fetcher()
        if frame is None:
            return Response(status_code=204)
        headers = {"Cache-Control": "no-store"}
        return Response(content=frame, media_type="image/jpeg", headers=headers)

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        if state.has_active_client():
            return HTMLResponse("Another device is accessing this page...")
        context = {
            "request": request,
            "vehicle_name": state.vehicle_name,
            "color_refs": state.get_color_references(),
            "color_tolerance_scale": state.get_color_tolerance(),
        }
        return templates.TemplateResponse("index.html", context)

    @app.get("/video/raw/frame")
    async def video_raw_frame() -> Response:
        return _serve_frame(state.get_raw_jpeg)

    @app.get("/video/out/frame")
    async def video_out_frame() -> Response:
        return _serve_frame(state.get_out_jpeg)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if not state.try_acquire_client():
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Another device is accessing this page. Please exit and try again.",
                }
            )
            await websocket.close()
            return

        await websocket.accept()
        stop_event = asyncio.Event()
        mode_handlers = {
            "manual": state.set_manual_mode,
            "lane_follow": state.set_lane_follow_enabled,
            "obstacle_avoid": state.set_obstacle_avoid_enabled,
            "traffic_light": state.set_traffic_light_enabled,
            "all_auto": state.set_all_auto_components,
        }

        async def _state_stream() -> None:
            try:
                while not stop_event.is_set():
                    await websocket.send_json(
                        {"type": "state", "data": state.snapshot()}
                    )
                    await asyncio.sleep(0.3)
            except (WebSocketDisconnect, RuntimeError):
                pass
            except asyncio.CancelledError:
                raise

        stream_task = asyncio.create_task(_state_stream())

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "set_mode":
                    mode = data.get("mode")
                    handler = mode_handlers.get(mode)
                    if handler is None:
                        continue
                    handler(bool(data.get("enabled")))
                    await websocket.send_json(
                        {"type": "state", "data": state.snapshot()}
                    )

                elif msg_type == "manual":
                    if not state.is_manual_mode():
                        continue
                    try:
                        left = float(data.get("left", 0.0))
                        right = float(data.get("right", 0.0))
                    except (TypeError, ValueError):
                        continue
                    state.set_manual_command(left, right)
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "set_speed_limit":
                    try:
                        value = float(data.get("value"))
                    except (TypeError, ValueError):
                        continue
                    state.set_speed_limit(value)
                    await websocket.send_json(
                        {"type": "state", "data": state.snapshot()}
                    )
                elif msg_type == "set_color_reference":
                    color = data.get("color")
                    value = data.get("value")
                    if not isinstance(color, str):
                        continue
                    try:
                        state.set_color_reference(
                            color, value if isinstance(value, str) else ""
                        )
                    except ValueError as exc:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc)}
                        )
                        continue
                    await websocket.send_json(
                        {"type": "state", "data": state.snapshot()}
                    )
                elif msg_type == "set_color_tolerance":
                    try:
                        value = float(data.get("value"))
                    except (TypeError, ValueError):
                        continue
                    state.set_color_tolerance(value)
                    await websocket.send_json(
                        {"type": "state", "data": state.snapshot()}
                    )
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            state.release_client()

    return app


def run_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8888) -> None:
    config = uvicorn.Config(
        app, host=host, port=port, log_level="error", access_log=False
    )
    server = uvicorn.Server(config)
    server.run()
