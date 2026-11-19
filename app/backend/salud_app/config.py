"""Settings for the app backend.

The backend is self-contained: it does not import the research code in `src/`. It reads
- the team database (Neon) through `DATABASE_URL`, taken from the environment or the repository root `.env`;
- local Phase 2 outputs under `Phase 2/reports/` (gitignored; they include Salud's real calls).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str | None = None
    # Local only for now: the app shows Salud's real calls and has no login, so it binds to localhost and only the
    # local frontend may call it.
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    phase2_reports: Path = REPO_ROOT / "Phase 2" / "reports"
    cache_seconds: int = 60

    @property
    def master_file(self) -> Path:
        return self.phase2_reports / "pre_training_data.xlsx"

    @property
    def real_sentiment_file(self) -> Path:
        return self.phase2_reports / "sentiment_cache" / "real.csv"

    @property
    def real_qa_file(self) -> Path:
        return self.phase2_reports / "qa_real.csv"

    @property
    def feature_sets_file(self) -> Path:
        return self.phase2_reports / "experiments" / "feature_sets_master.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
