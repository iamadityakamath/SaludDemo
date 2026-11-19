// Mirrors the backend responses (app/backend/salud_app/routers). Keep the two in step.

export type CallStatus = "done" | "processing" | "incomplete";

export interface CallRow {
  call_id: string;
  source: "new" | "real" | "salud" | "synthetic";   // salud = Salud's final-package calls
  file_id: string;
  added_by: string | null;
  added_at: string | null;
  analyzed: boolean;
  scored: boolean;
  end_label: string | null;
  status: CallStatus;
}

export interface CallList {
  me: string;
  website: boolean;
  llm: boolean;
  items: string[];
  definitions: Record<string, string>;
  calls: CallRow[];
}

export interface TranscriptTurn { n: number; speaker: string; text: string }
export interface Transcript { text: string; turns: TranscriptTurn[] }

export interface Evidence { turn: number; quote: string }

export interface Analysis {
  none?: boolean;
  checklist?: Record<string, { answer: string | null; reason?: string; evidence?: Evidence[] }>;
  sentiment?: { label: string | null; evidence?: Evidence[] };
  outcome?: { label: string | null; evidence?: Evidence[] };
  summary?: string;
  coaching?: { item: string; tip: string; turn: number | null }[];
  model?: string;
}

// Call-level summary: caller_* and agent_* numbers (see sentiment/scoring.py).
export interface SentimentCall {
  call_turns: number;
  caller_turns: number;
  caller_overall: number;
  caller_label: string;
  caller_start: number;
  caller_end: number;
  caller_shift: number;
  caller_end_label: string;
  caller_min_score: number;
  caller_min_turn: number;
  caller_max_score: number;
  caller_max_turn: number;
  [k: string]: number | string;
}

export interface Sentiment {
  none?: boolean;
  model?: string;
  call?: SentimentCall;
  turns?: { turn: number; role: string; score: number; magnitude: number; label: string }[];
}

export interface CallResult { sentiment: Sentiment; analysis: Analysis; notes: string[] }

export interface Health { status: "ok" | "degraded"; database: boolean; phase2_master_file: boolean }

export interface Phase2Part {
  data_type: string; split: string; calls: number; patients: number; usable: number;
  pay_rate_pre_call: number | null; pay_rate_post_call: number | null;
}

export interface Phase2Summary {
  available: boolean;
  rows: number;
  by_part: Phase2Part[];
  date_range: { first_call?: string | null; last_call?: string | null };
  feature_sets: Record<string, Record<string, string[]>>;
}
