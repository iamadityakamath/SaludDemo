"""Independent check of the master file: rebuild the model fields and targets from the RAW tables with separate code.

    salud-propensity verify      # real: Neon `final` (read only); synthetic: data/phase2/synthetic_world_master/crm

Deliberately does not use sources.py or snapshots.py: raw column names, option codes decoded by number. Prints mismatch
counts only (no row contents). Any mismatch means the pipeline maps Salud's data differently from this definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...db import connect
from ...paths import PHASE2_WORLDS
from .master import MASTER_XLSX


def main(sample: int = 600, world: str = "master") -> int:
    m = pd.read_excel(MASTER_XLSX)
    m["as_of"] = pd.to_datetime(m["as_of"])

    # ---------- raw tables -> one common shape (independent decoding) ----------
    with connect() as c:
        q = lambda s: pd.DataFrame(c.execute(s).fetchall(), columns=[d.name for d in c.execute(s).description])
        ra = q('select "salud_gphpvaccountsId" a, "salud_GuarantorId" g, "salud_HBPB"::text hb, "salud_PlacementDate" pd, "salud_PlacementPatientBalance"::float pb from final.accounts')
        rt = q('select "salud_ReceiptsId" a, "salud_Guarantor" g, "salud_PostDate" d, "salud_Amount"::float amt, "salud_TXNType"::text typ, "salud_PaymentType"::text pt from final.transactions')
        rc = q('select "salud_gphpvcontacthistoryId" cid, "salud_GuarantorId" g, "salud_DateandTimeofContact" t from final.contact_history')
    ra["hosp"] = ra.hb == "Hospital Account"
    rt["patpay"] = (rt.typ == "Payment") & (rt.pt == "Patient Payment")
    rt["inspay"] = (rt.typ == "Payment") & (rt.pt == "Insurance Payment")
    rt["adj"] = rt.typ == "Adjustment"

    W = str(PHASE2_WORLDS[world] / "crm") + "/"
    sa = pd.read_csv(W + "salud_gphpvaccounts.csv")
    st = pd.read_csv(W + "salud_gphpvreceipts.csv")
    sc = pd.read_csv(W + "salud_gphpvcontacthistory.csv")
    sa = sa.rename(columns={"salud_gphpvaccountsId": "a", "salud_GuarantorId": "g", "salud_PlacementDate": "pd",
                            "salud_PlacementPatientBalance": "pb"})
    sa["hosp"] = sa["salud_HBPB"] == 785570000
    st = st.rename(columns={"salud_ReceiptsId": "a", "salud_Guarantor": "g", "salud_PostDate": "d", "salud_Amount": "amt"})
    inc = st["salud_ExcludeTransaction"] == 785570001
    st["patpay"] = (st.salud_TXNType == 785570000) & (st.salud_PaymentType == 785570000) & inc
    st["inspay"] = (st.salud_TXNType == 785570000) & (st.salud_PaymentType == 785570001)
    st["adj"] = st.salud_TXNType == 785570001
    sc = sc.rename(columns={"salud_gphpvcontacthistoryId": "cid", "salud_GuarantorId": "g", "salud_DateandTimeofContact": "t"})

    def prep(a, t):
        a = a.copy(); t = t.copy()
        a["pd"] = pd.to_datetime(a["pd"]); t["d"] = pd.to_datetime(t["d"]); t["day"] = t["d"].dt.normalize()
        return a, t, t["d"].max().normalize()

    def recompute(row, A, T, end):
        t = row.as_of; day = t.normalize(); g = row.guarantor_id
        acc = A[(A.g == g) & (A["pd"].dt.normalize() <= day)]
        tx = T[T.g == g]
        pay = tx[tx.patpay & (tx.day < day)]
        first = acc["pd"].min() if len(acc) else None
        out = {
            "n_accounts": len(acc), "n_hospital_accounts": int(acc.hosp.sum()), "n_physician_accounts": int((~acc.hosp).sum()),
            "balance_at_t": max(acc.pb.sum() - (pay[pay.day >= first.normalize()].amt.sum() if first is not None else 0), 0.0),
            "days_since_first_placement": (t - first).days if first is not None else np.nan,
            "days_since_last_placement": (t - acc["pd"].max()).days if len(acc) else np.nan,
            "n_payments_90d": int((pay.day >= day - pd.Timedelta(days=90)).sum()),
            "n_payments_365d": int((pay.day >= day - pd.Timedelta(days=365)).sum()),
            "days_since_last_payment": (day - pay.day.max()).days if len(pay) else np.nan,
            "paid_share_so_far": (min(pay[pay.day >= first.normalize()].amt.sum() / acc.pb.sum(), 1.0)
                                  if first is not None and acc.pb.sum() > 0 else np.nan),
            "last_payment_amount": pay.loc[pay.day == pay.day.max(), "amt"].sum() if len(pay) else 0.0,
            "n_insurance_payments_365d": int((tx.inspay & (tx.day < day) & (tx.day >= day - pd.Timedelta(days=365))).sum()),
            "n_adjustments_365d": int((tx.adj & (tx.day < day) & (tx.day >= day - pd.Timedelta(days=365))).sum()),
            "paid_30d": bool((tx.patpay & (tx.day >= day) & (tx.day <= day + pd.Timedelta(days=30))).any()),
            "paid_after_call_30d": bool((tx.patpay & (tx.day > day) & (tx.day <= day + pd.Timedelta(days=30))).any()),
            "window_complete": bool(day + pd.Timedelta(days=30) <= end),
            # leakage probe: any feature input dated on/after the call? (should never be counted)
            "future_tx_exists": bool((tx.day >= day).any()),
        }
        return out

    FIELDS = ["n_accounts", "n_hospital_accounts", "n_physician_accounts", "balance_at_t", "days_since_first_placement",
              "days_since_last_placement", "n_payments_90d", "n_payments_365d", "days_since_last_payment",
              "paid_share_so_far", "last_payment_amount",
              "n_insurance_payments_365d", "n_adjustments_365d", "paid_30d", "paid_after_call_30d", "window_complete"]

    def compare(rows, A, T, end, name):
        bad = {f: 0 for f in FIELDS}; n = 0; fut = 0
        for row in rows.itertuples():
            if not isinstance(row.guarantor_id, str):
                continue
            n += 1; r = recompute(row, A, T, end); fut += r["future_tx_exists"]
            for f in FIELDS:
                a, b = getattr(row, f), r[f]
                if isinstance(b, bool) or f.startswith("paid") or f == "window_complete":
                    same = bool(a) == bool(b)
                elif pd.isna(a) or pd.isna(b):
                    same = pd.isna(a) and pd.isna(b)
                else:
                    tol = {"balance_at_t": 0.011, "last_payment_amount": 0.011, "paid_share_so_far": 0.0001}.get(f, 0)
                    same = abs(float(a) - float(b)) <= tol
                bad[f] += not same
        print(f"{name}: {n} rows checked ({fut} have later transactions that must be ignored)")
        print("  mismatches:", {k: v for k, v in bad.items() if v} or "none")
        return sum(bad.values())

    ra_, rt_, rend = prep(ra, rt)
    bad = compare(m[m.data_type == "real"], ra_, rt_, rend, "real")
    sa_, st_, send = prep(sa, st)
    rows = m[m.data_type == "synthetic"]
    sample_rows = rows.sample(min(sample, len(rows)), random_state=1)
    bad += compare(sample_rows, sa_, st_, send, f"synthetic ({len(sample_rows)} random)")

    # call ids exist in raw contact history; as_of equals the raw contact time
    for name, rows, C in [("real", m[m.data_type == "real"], rc), ("synthetic", m[m.data_type == "synthetic"], sc)]:
        C = C.assign(t=pd.to_datetime(C.t)).set_index("cid")
        ok = rows.call_id.isin(C.index)
        tdiff = (rows.loc[ok, "as_of"] - C.loc[rows.loc[ok, "call_id"], "t"].dt.floor("min").values).abs().max()
        print(f"{name}: call_ids found in raw contact history {ok.mean():.0%}; max as_of difference {tdiff}")
        bad += int((~ok).sum()) + int(tdiff > pd.Timedelta(0))

    return bad
