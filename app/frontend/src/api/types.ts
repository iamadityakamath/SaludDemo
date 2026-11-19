// Mirrors app/backend/salud_app/schemas.py. Keep the two in step.

export type Source = "real" | "synthetic" | "pasted";
export type Role = "agent" | "caller" | "system" | "unknown";

export interface SentimentSummary {
  caller_overall: number | null;
  caller_start: number | null;
  caller_end: number | null;
  caller_shift: number | null;
  caller_label: string | null;
  caller_end_label: string | null;
  agent_overall: number | null;
  agent_label: string | null;
}

export interface CallSummary {
  key: string;
  source: Source;
  dataset: "final" | "phase1";
  contacted_at: string | null;
  outcome: string | null;
  caller_end_sentiment: number | null;
  caller_end_label: string | null;
  qa_score: number | null;
  has_ai_review: boolean;
  turns: number;
}

export interface CallList {
  total: number;
  items: CallSummary[];
}

export interface Turn {
  turn: number;
  speaker: string;
  role: Role;
  text: string;
  sentiment: number | null;
  sentiment_label: string | null;
}

export interface AiReview {
  model: string | null;
  prompt_version: string | null;
  summary: string | null;
  outcome: string | null;
  sentiment: string | null;
  quality_score: number | null;
  checklist: { item: string; answer: string | null; reason: string | null }[];
  coaching: string[];
}

export interface CallDetail extends CallSummary {
  transcript: Turn[];
  sentiment: SentimentSummary | null;
  ai_review: AiReview | null;
  call_facts: Record<string, string | number | null>;
}

export interface Health {
  status: "ok" | "degraded";
  database: boolean;
  phase2_master_file: boolean;
}

export interface Phase2Part {
  data_type: string;
  split: string;
  calls: number;
  patients: number;
  usable: number;
  pay_rate_pre_call: number | null;
  pay_rate_post_call: number | null;
}

export interface Phase2Summary {
  available: boolean;
  rows: number;
  by_part: Phase2Part[];
  date_range: { first_call?: string | null; last_call?: string | null };
  feature_sets: Record<string, Record<string, string[]>>;
}
