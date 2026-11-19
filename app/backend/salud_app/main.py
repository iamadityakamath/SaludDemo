"""FastAPI app. Run (from app/backend):  .venv/bin/uvicorn salud_app.main:app --reload --host 127.0.0.1 --port 8000

Local only for now: there is no login and the app shows Salud's real calls, so it binds to 127.0.0.1 and only the local
frontend (Vite on port 5173) may call it. Every database connection is read-only.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import DatabaseNotConfigured
from .routers import calls, health, phase2

app = FastAPI(title="Salud app API", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().allowed_origins, allow_methods=["GET"],
                   allow_headers=["*"])
for r in (health.router, calls.router, phase2.router):
    app.include_router(r, prefix="/api")


@app.exception_handler(DatabaseNotConfigured)
def no_database(_: Request, exc: DatabaseNotConfigured) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})
