-- Neon becomes the team's shared, live database for Phase 1: real and synthetic calls, their transcripts,
-- AI analyses and every rater's tags. (Replaces the synthetic-only rule from 002.)
--
-- Real calls are Salud's redacted transcripts. This Neon project is not HIPAA-enabled; the team lead decided
-- to keep them here (Oct 2026). Confirm with Salud before sharing access beyond the team.

-- 1. Allow real calls in every phase1 table.
ALTER TABLE phase1.calls         DROP CONSTRAINT IF EXISTS calls_call_id_check;
ALTER TABLE phase1.ai_analyses   DROP CONSTRAINT IF EXISTS ai_analyses_call_id_check;
ALTER TABLE phase1.ai_checklist  DROP CONSTRAINT IF EXISTS ai_checklist_call_id_check;
ALTER TABLE phase1.ai_coaching   DROP CONSTRAINT IF EXISTS ai_coaching_call_id_check;
ALTER TABLE phase1.reviewed_tags DROP CONSTRAINT IF EXISTS reviewed_tags_call_id_check;

-- 2. Calls carry their source, file name and transcript, so teammates need no local call files.
--    Real calls have no row in the synthetic CRM, so contact_id becomes optional.
ALTER TABLE phase1.calls ALTER COLUMN contact_id DROP NOT NULL;
ALTER TABLE phase1.calls ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'synthetic';
ALTER TABLE phase1.calls ADD COLUMN IF NOT EXISTS file_id text;
ALTER TABLE phase1.calls ADD COLUMN IF NOT EXISTS transcript text;
UPDATE phase1.calls SET file_id = substring(call_id from position('/' in call_id) + 1) WHERE file_id IS NULL;
ALTER TABLE phase1.calls ALTER COLUMN file_id SET NOT NULL;
ALTER TABLE phase1.calls ALTER COLUMN source DROP DEFAULT;
ALTER TABLE phase1.calls ADD CONSTRAINT calls_source_check CHECK (source IN ('real', 'synthetic'));
ALTER TABLE phase1.calls ADD CONSTRAINT calls_id_matches_source CHECK (call_id = source || '/' || file_id);
ALTER TABLE phase1.calls ADD CONSTRAINT calls_real_have_no_contact CHECK (source = 'synthetic' OR contact_id IS NULL);

-- 3. Know when each tag was last saved.
ALTER TABLE phase1.reviewed_tags ADD COLUMN IF NOT EXISTS saved_at timestamptz NOT NULL DEFAULT now();

-- 4. Tags made before the dashboard pre-filled from the AI file, kept apart and never used in comparisons.
CREATE TABLE IF NOT EXISTS phase1.legacy_manual_tags (LIKE phase1.reviewed_tags INCLUDING DEFAULTS);
ALTER TABLE phase1.legacy_manual_tags ADD PRIMARY KEY (call_id, rater);
ALTER TABLE phase1.legacy_manual_tags
    ADD CONSTRAINT legacy_manual_tags_call_fk FOREIGN KEY (call_id) REFERENCES phase1.calls (call_id);
