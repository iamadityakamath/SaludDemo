from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from salud_pipeline import final_package as fp


def test_convert_keeps_values_exact():
    assert fp.convert("Contact_ABC", "text") == "Contact_ABC"
    assert fp.convert(612.61, "numeric") == Decimal("612.61")
    assert fp.convert(pd.Timestamp("2026-07-01 13:02:26"), "timestamp without time zone") == datetime(2026, 7, 1, 13, 2, 26)
    assert fp.convert(float("nan"), "text") is None
    assert fp.convert(pd.NaT, "timestamp without time zone") is None
    assert fp.convert(3.0, "integer") == 3
    with pytest.raises(fp.PackageError):
        fp.convert(3.5, "integer")


@pytest.mark.skipif(not (fp.FINAL_PACKAGE / fp.WORKBOOK).exists() or not fp.HANDOFF_JSON.exists(),
                    reason="Salud's final package is not on this machine (gitignored)")
def test_package_reads_every_tab_and_transcript():
    pkg = fp.read_package()
    assert set(pkg) == set(fp.LOAD_ORDER)
    assert len(pkg["transcripts"]) == len(pkg["contact_history"])
    assert set(pkg["transcripts"]["salud_TCNRecordID"]) == set(pkg["contact_history"]["salud_TCNRecordID"])
    assert not pkg["transcripts"]["transcript"].str.contains("\r").any()
