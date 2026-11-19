"""Settings for the app backend.

The backend is self-contained: it does not import the research code in `src/`. It reads
- the team database (Neon) through `DATABASE_URL`, taken from the environment or the repository root `.env`;
- local Phase 2 outputs under `Phase 2/reports/` (gitignored; they include Salud's real calls);
- the LLM settings (GROQ_API_KEY, ...) for the AI review, from the same .env.
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
    # LLM for the AI review (Groq by default; see phase1/llm.py)
    groq_api_key: str | None = None
    groq_model: str | None = None
    salud_llm_base_url: str | None = None
    salud_llm_model: str | None = None
    salud_llm_api_key: str | None = None
    # Who is using the app (no login): shown in the header and saved as "added by" on uploads.
    salud_app_user: str = "aditya"
    # Analyze calls missing results in the background (every 15 s), like the team dashboard on a laptop.
    salud_auto_analyze: bool = True
    # Local only for now: the app shows Salud's real calls and has no login, so it binds to localhost and only the
    # local frontend may call it.
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    phase2_reports: Path = REPO_ROOT / "Phase 2" / "reports"
    # The AI review cache (one file per reviewed transcript), shared with the team dashboard's local cache.
    llm_cache: Path = REPO_ROOT / "Phase 1" / "reports" / "llm_cache"

    @property
    def master_file(self) -> Path:
        return self.phase2_reports / "pre_training_data.xlsx"

    @property
    def feature_sets_file(self) -> Path:
        return self.phase2_reports / "experiments" / "feature_sets_master.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
