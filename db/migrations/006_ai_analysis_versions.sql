-- Trial prompt versions' AI analyses of a call, kept side by side so prompt changes can be scored against the
-- same tags. Written by `salud-phase1 trial`, plus the one-time copy below of the analyses that existed when this
-- ran; later dashboard analyses and Re-analyze results are NOT copied here (they live only in phase1.ai_analyses,
-- the single version the dashboard pre-fills from). Like ai_analyses, no ON DELETE CASCADE: a call with analyses
-- can't be deleted by accident. (Numbered 006 but applied after 007; neither depends on the other.)
CREATE TABLE IF NOT EXISTS phase1.ai_analysis_versions (
    call_id        text NOT NULL REFERENCES phase1.calls (call_id),
    model          text NOT NULL,
    prompt_version text NOT NULL,
    analyzed_at    timestamp,
    sentiment      text,
    outcome        text,
    quality_score  numeric(4,3),
    summary        text,
    quotes_dropped int,
    raw            jsonb NOT NULL,
    PRIMARY KEY (call_id, model, prompt_version)
);

-- Keep the analyses already in use as their own version.
INSERT INTO phase1.ai_analysis_versions
SELECT call_id, model, prompt_version, analyzed_at, sentiment, outcome, quality_score, summary, quotes_dropped, raw
FROM phase1.ai_analyses
ON CONFLICT DO NOTHING;
