-- Salud's final-package calls in Agent Call Reviews: source 'salud', call_id 'salud/<file name>'.
-- They get sentiment and an AI review like every other call (stored in the same phase1 tables). They are not
-- tagged by raters, so the Phase 1 AI-vs-tags agreement (computed on tagged calls only) is unchanged.
-- Loaded by the app: `python -m salud_app.phase1.import_package` (app/backend).
ALTER TABLE phase1.calls DROP CONSTRAINT IF EXISTS calls_source_check;
ALTER TABLE phase1.calls ADD CONSTRAINT calls_source_check CHECK (source IN ('real', 'synthetic', 'new', 'salud'));
