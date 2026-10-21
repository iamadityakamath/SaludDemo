-- Which rater tags which call. Filled by `salud-phase1 assign`; the dashboard shows each rater their calls.
-- A call can have several raters (real calls get two, for rater agreement). Rater names are lower case.
CREATE TABLE IF NOT EXISTS phase1.assignments (
    call_id     text NOT NULL REFERENCES phase1.calls (call_id) ON DELETE CASCADE,
    rater       text NOT NULL CHECK (rater <> '' AND rater = lower(btrim(rater))),
    assigned_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (call_id, rater)
);
