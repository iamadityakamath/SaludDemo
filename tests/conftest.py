import os
import uuid

import pytest

from salud_pipeline.llm_client import load_dotenv


@pytest.fixture(scope="session")
def test_db_url():
    """The Neon 'tests' branch (a copy of the team database). Never the team's live database."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL_TEST")
    if not url:
        pytest.skip("DATABASE_URL_TEST not set (Neon 'tests' branch)")
    return url


@pytest.fixture()
def store(test_db_url):
    from salud_pipeline.phase1.store import Store

    s = Store(test_db_url)
    tag = uuid.uuid4().hex[:8]
    s.test_rater = f"pytest-{tag}"
    s.test_call = f"synthetic/pytest-{tag}.txt"
    yield s
    with s.pool.connection() as c:  # remove everything this test created
        c.execute("DELETE FROM phase1.reviewed_tags WHERE rater = %s OR call_id = %s", (s.test_rater, s.test_call))
        for table in ("ai_coaching", "ai_checklist", "ai_analyses"):
            c.execute(f"DELETE FROM phase1.{table} WHERE call_id = %s", (s.test_call,))
        c.execute("DELETE FROM phase1.calls WHERE call_id = %s", (s.test_call,))
    s.close()
