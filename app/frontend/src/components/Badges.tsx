import type { Source } from "../api/types";

const SOURCE_TEXT: Record<Source, string> = { real: "Real", synthetic: "Synthetic", pasted: "Pasted" };

export function SourceBadge({ source }: { source: Source }) {
  return <span className={`badge badge-${source}`}>{SOURCE_TEXT[source]}</span>;
}

/** Sentiment score from -1 to 1, shown with its label so meaning never depends on colour alone. */
export function SentimentChip({ score, label }: { score: number | null; label?: string | null }) {
  if (score === null || score === undefined) return <span className="muted">—</span>;
  // Colour follows the stored label when there is one (the model's own thresholds), otherwise the sign of the score.
  const l = (label ?? "").toLowerCase();
  const tone = l ? (l.startsWith("pos") ? "pos" : l.startsWith("neg") ? "neg" : "neu")
    : score > 0.1 ? "pos" : score < -0.1 ? "neg" : "neu";
  return <span className={`chip chip-${tone}`}>{score.toFixed(2)}{label ? ` · ${label}` : ""}</span>;
}

export function Status({ loading, error }: { loading: boolean; error: string | null }) {
  if (loading) return <p className="muted">Loading…</p>;
  if (error) return <p className="error">Couldn't load: {error}</p>;
  return null;
}
