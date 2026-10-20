-- Planted labels for synthetic calls (every checklist item, sentiment and outcome), so the comparison's
-- "Synthetic check" runs from the database alone. NULL for real calls.
ALTER TABLE phase1.calls ADD COLUMN IF NOT EXISTS planted jsonb;
ALTER TABLE phase1.calls ADD CONSTRAINT calls_planted_only_synthetic CHECK (source = 'synthetic' OR planted IS NULL);
