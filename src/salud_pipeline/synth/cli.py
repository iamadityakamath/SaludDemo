"""Generate a synthetic run.

    salud-synth --config configs/default.json --out data/synthetic/run_default

Output layout:
    crm/          six CRM-shaped tables (CSV), the only inputs modeling code may read as features
    truth/        planted latents and outcomes, for evaluation only, never as features
    transcripts/  <contact id>.txt per connected call
    llm_prompts.jsonl   one generation prompt per connected call (for the llm mode or batch jobs)
    synthetic.sqlite    all tables, for SQL access
    manifest.json       config, seed, counts, validation results
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..llm_client import LLMEndpoint
from .config import GeneratorConfig
from .llm import generate_with_endpoint, prompt_record
from .transcripts import render_template
from .validate import validate
from .world import generate


def _write_transcripts(calls: pd.DataFrame, out: Path, cfg: GeneratorConfig, mode: str,
                       endpoint: LLMEndpoint, limit: int | None) -> int:
    tdir = out / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 1)  # separate stream: transcripts never change the tables
    records = calls.to_dict("records")
    with open(out / "llm_prompts.jsonl", "w") as fh:
        for call in records:
            fh.write(json.dumps(prompt_record(call)) + "\n")
    if mode == "none":
        return 0
    if mode == "template":
        for call in records:
            (tdir / f"{call['contact_id']}.txt").write_text(render_template(call, rng, cfg.transcript_noise))
        return len(records)

    # LLM mode: a fixed random order (so a run can stop and resume), written to llm_order.txt.
    # Downstream evaluation takes the first N of this order as its test set.
    order = np.random.default_rng(cfg.seed + 2).permutation(len(records))
    chosen = [records[i] for i in order[: limit or len(records)]]
    (out / "llm_order.txt").write_text("\n".join(c["contact_id"] for c in chosen) + "\n")
    header = (f"Editorial note: Synthetic call written by {endpoint.model} from planted labels. "
              "Fictional dialogue; no real person, account or recording.")
    done = failed = 0
    t0 = time.time()
    for k, call in enumerate(chosen, 1):
        path = tdir / f"{call['contact_id']}.txt"
        if path.exists():
            done += 1
            continue
        try:
            text = generate_with_endpoint(prompt_record(call)["prompt"], endpoint)
        except RuntimeError as e:
            failed += 1
            print(f"  [{k}/{len(chosen)}] skipped {call['contact_id'][:8]}: {e}", flush=True)
            continue
        path.write_text(header + "\n\n" + text + "\n")
        done += 1
        if k % 10 == 0 or k == len(chosen):
            print(f"  [{k}/{len(chosen)}] written={done} failed={failed} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return done


def _summary(t: dict[str, pd.DataFrame]) -> dict:
    acc = t["truth_accounts"]
    openm = acc[acc["open_at_cutoff"]]
    calls = t["truth_calls"]
    q = pd.qcut(openm["feat_call_quality"], 4, duplicates="drop")
    return {
        "accounts_open_at_cutoff": int(len(openm)),
        "pay_rate_in_window": round(float(openm["planted_paid"].mean()), 3),
        "pay_rate_by_call_quality_quartile": {str(k): round(float(v), 3)
                                              for k, v in openm.groupby(q, observed=True)["planted_paid"].mean().items()},
        "connected_calls": int(len(calls)),
        "qa_pass_rate_by_item": {k: round(float(calls[f"qa_{k}"].mean()), 3) for k in S.QA_KEYS},
        "sentiment_mix": calls["sentiment_label"].value_counts(normalize=True).round(3).to_dict(),
        "true_result_mix": calls["true_result"].value_counts(normalize=True).round(3).to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="JSON config (see configs/)")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/run"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-guarantors", type=int)
    ap.add_argument("--transcripts", choices=["template", "llm", "none"], default="template")
    ap.add_argument("--llm-base-url", help="default: $SALUD_LLM_BASE_URL or local Ollama")
    ap.add_argument("--llm-model", help="default: $SALUD_LLM_MODEL; use a different family than the scorer")
    ap.add_argument("--llm-limit", type=int, help="max calls to send to the LLM")
    args = ap.parse_args(argv)

    cfg = GeneratorConfig.from_json(args.config) if args.config else GeneratorConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_guarantors is not None:
        cfg.n_guarantors = args.n_guarantors

    out: Path = args.out
    tables = generate(cfg)
    cutoff = pd.Timestamp(cfg.prediction_cutoff)
    checks = validate(tables, cutoff, cfg.outcome_horizon_days)

    (out / "crm").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        folder = "truth" if name.startswith("truth_") else "crm"
        df.to_csv(out / folder / f"{name}.csv", index=False)
    with sqlite3.connect(out / "synthetic.sqlite") as con:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False)

    endpoint = LLMEndpoint()
    endpoint.base_url = args.llm_base_url or endpoint.base_url
    endpoint.model = args.llm_model or endpoint.model
    endpoint.max_retries = 8  # long batch runs: wait out rate-limit windows instead of failing
    n_tx = _write_transcripts(tables["truth_calls"], out, cfg, args.transcripts, endpoint, args.llm_limit)

    manifest = {
        "generator": "salud_pipeline.synth v0.1",
        "config": cfg.to_dict(),
        "conventions": {"timezone": "America/Chicago (naive local)", "currency": "USD",
                        "actions_taken": "';'-joined option codes",
                        "receipt_amount_sign": "positive = cash/credit received",
                        "worklist_as_of": str(cutoff + pd.Timedelta(days=cfg.outcome_horizon_days + 1))},
        "counts": {k: int(len(v)) for k, v in tables.items()} | {"transcripts": n_tx},
        "summary": _summary(tables),
        "validation": [{"check": c.name, "ok": bool(c.ok), "detail": c.detail} for c in checks],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    failed = [c for c in checks if not c.ok]
    print(f"Wrote {out}  ({manifest['counts']})")
    print(f"Validation: {len(checks) - len(failed)}/{len(checks)} checks passed")
    for c in failed:
        print(f"  FAIL {c.name}: {c.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
