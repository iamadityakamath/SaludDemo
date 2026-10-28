-- Turn-level sentiment (salud-sentiment run, or the website's Analyze) and calls pasted on the /analyze page.

-- 1. Calls pasted on /analyze: source 'new', with who added them and when. They stay out of the tagging
--    page, assignments and every AI-vs-tags comparison (store.py filters on source).
ALTER TABLE phase1.calls DROP CONSTRAINT IF EXISTS calls_source_check;
ALTER TABLE phase1.calls ADD CONSTRAINT calls_source_check CHECK (source IN ('real', 'synthetic', 'new'));
ALTER TABLE phase1.calls ADD COLUMN IF NOT EXISTS added_by text;
ALTER TABLE phase1.calls ADD COLUMN IF NOT EXISTS added_at timestamptz;

-- 2. One row per scored turn (caller and agent turns; notes and automated systems are skipped).
--    Turn numbers are the dashboard's (split_turns on the stored transcript).
CREATE TABLE IF NOT EXISTS phase1.turn_sentiment (
    call_id    text NOT NULL REFERENCES phase1.calls (call_id) ON DELETE CASCADE,
    turn       int  NOT NULL,
    speaker    text NOT NULL,
    role       text NOT NULL CHECK (role IN ('caller', 'agent')),
    words      int  NOT NULL,
    weight     real NOT NULL,
    score      real NOT NULL CHECK (score BETWEEN -1 AND 1),        -- P(positive) - P(negative)
    magnitude  real NOT NULL CHECK (magnitude BETWEEN 0 AND 1),     -- 1 - P(neutral)
    label      text NOT NULL CHECK (label IN ('negative', 'neutral', 'positive')),
    p_negative real NOT NULL,
    p_neutral  real NOT NULL,
    p_positive real NOT NULL,
    PRIMARY KEY (call_id, turn)
);

-- 3. One row per scored call: the model, and the caller_* / agent_* summaries (start, end, shift, lowest turn...).
CREATE TABLE IF NOT EXISTS phase1.call_sentiment (
    call_id   text PRIMARY KEY REFERENCES phase1.calls (call_id) ON DELETE CASCADE,
    model     text NOT NULL,
    scored_at timestamptz NOT NULL DEFAULT now(),
    summary   jsonb NOT NULL
);
