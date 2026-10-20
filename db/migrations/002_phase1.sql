-- Phase 1 results for the synthetic calls: AI analyses and reviewed tags.
-- SYNTHETIC CALLS ONLY. Every table checks that call_id starts with 'synthetic/', so the 10 real Salud
-- calls (and the AI analyses that quote them) cannot be stored here. This Neon project is not HIPAA-enabled.
--
-- Each synthetic call is a real row in big_synthetic.salud_gphpvcontacthistory (same contact ID), so call
-- quality joins straight to the account and payment tables: the Stage 1 -> Stage 2 link.

CREATE SCHEMA IF NOT EXISTS phase1;

-- The calls in the Phase 1 test set (configs/phase1_synthetic_calls.txt).
CREATE TABLE IF NOT EXISTS phase1.calls (
    call_id        text PRIMARY KEY CHECK (call_id LIKE 'synthetic/%'),
    contact_id     uuid NOT NULL UNIQUE
                   REFERENCES big_synthetic."salud_gphpvcontacthistory" ("salud_gphpvcontacthistoryId"),
    -- Planted labels the transcript was written from (synthetic ground truth)
    planted_scenario  text,
    planted_sentiment text,
    planted_outcome   text
);

-- One row per call: the script's AI analysis (reports/phase1/ai_analyses.json).
CREATE TABLE IF NOT EXISTS phase1.ai_analyses (
    call_id        text PRIMARY KEY REFERENCES phase1.calls (call_id) CHECK (call_id LIKE 'synthetic/%'),
    model          text NOT NULL,
    prompt_version text NOT NULL,
    analyzed_at    timestamp,
    sentiment      text,
    outcome        text,
    quality_score  numeric(4,3),          -- Yes / (Yes + No) over the 7 transcript items; N/A excluded
    summary        text,
    quotes_dropped int,
    raw            jsonb NOT NULL          -- the full analysis as saved
);

-- One row per call and checklist item.
CREATE TABLE IF NOT EXISTS phase1.ai_checklist (
    call_id      text NOT NULL REFERENCES phase1.ai_analyses (call_id) CHECK (call_id LIKE 'synthetic/%'),
    item         text NOT NULL,
    answer       text CHECK (answer IS NULL OR answer IN ('Yes', 'No', 'N/A')),
    reason       text,
    evidence     jsonb NOT NULL DEFAULT '[]',   -- [{"turn": 3, "quote": "..."}], verified against the transcript
    needs_review boolean NOT NULL,              -- no answer, or no supporting quote
    PRIMARY KEY (call_id, item)
);

-- One row per AI coaching tip.
CREATE TABLE IF NOT EXISTS phase1.ai_coaching (
    call_id   text NOT NULL REFERENCES phase1.ai_analyses (call_id) CHECK (call_id LIKE 'synthetic/%'),
    tip_no    int NOT NULL,
    item      text,
    tip       text NOT NULL,
    turn      int,
    PRIMARY KEY (call_id, tip_no)
);

-- One row per call per rater, as saved from the dashboard (data/phase1/reviewed_tags/synthetic_calls.csv).
CREATE TABLE IF NOT EXISTS phase1.reviewed_tags (
    call_id                  text NOT NULL REFERENCES phase1.calls (call_id) CHECK (call_id LIKE 'synthetic/%'),
    rater                    text NOT NULL,
    greeting                 text,
    culture                  text,
    capturing_data           text,
    understanding_needs      text,
    correct_info             text,
    first_contact_resolution text,
    compliance               text,
    sentiment                text,
    outcome                  text,
    notes                    text,
    llm_model                text,     -- model whose answers pre-filled the form; blank = tagged without AI
    overrides                text[],   -- items the rater changed from the AI's answer
    PRIMARY KEY (call_id, rater)
);
