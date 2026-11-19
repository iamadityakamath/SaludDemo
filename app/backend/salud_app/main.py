"""FastAPI app. Run (from app/backend):  .venv/bin/uvicorn salud_app.main:app --reload --host 127.0.0.1 --port 8000

Local only for now: there is no login and the app shows Salud's real calls, so it binds to 127.0.0.1 and only the
local frontend (Vite on port 5173) may call it. Agent Call Reviews writes to the shared team database exactly like
the team dashboard: uploaded calls, their sentiment and their AI review (phase1 tables).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import DatabaseNotConfigured
from .routers import health, phase1, phase2


@asynccontextmanager
async def lifespan(_: FastAPI):
    stop = None
    if get_settings().salud_auto_analyze and get_settings().database_url:
        from .phase1.analysis import start_sweeper
        stop = start_sweeper()   # analyze calls added any other way, every 15 s
    yield
    if stop:
        stop.set()


app = FastAPI(title="Salud app API", version="0.2.0", docs_url="/api/docs", openapi_url="/api/openapi.json",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().allowed_origins, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])
for r in (health.router, phase1.router, phase2.router):
    app.include_router(r, prefix="/api")


@app.exception_handler(DatabaseNotConfigured)
def no_database(_: Request, exc: DatabaseNotConfigured) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
def runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:   # e.g. database unreachable
    return JSONResponse(status_code=503, content={"detail": str(exc)[:300]})
