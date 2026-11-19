from fastapi import APIRouter

from .. import db
from ..config import get_settings
from ..schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
def health() -> Health:
    database = db.ping()
    master = get_settings().master_file.exists()
    return Health(status="ok" if database else "degraded", database=database, phase2_master_file=master)
