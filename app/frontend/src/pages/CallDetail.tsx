import { Fragment } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { useApi } from "../api/useApi";
import { SentimentChip, SourceBadge, Status } from "../components/Badges";

export function CallDetail() {
  const key = decodeURIComponent(useParams().key ?? "");
  const call = useApi(() => api.call(key), [key]);
  const c = call.data;

  return (
    <section>
      <p><Link to="/calls">← All calls</Link></p>
      <Status loading={call.loading} error={call.error} />
      {c && (
        <>
          <h1>Call review <SourceBadge source={c.source} /></h1>
          <p className="muted">
            {c.contacted_at ? new Date(c.contacted_at).toLocaleString() : "No call date (Phase 1 call)"} · {c.turns} turns
            {c.outcome ? ` · Outcome: ${c.outcome}` : ""}
          </p>
          <div className="grid">
            <div className="panel">
              <h2>Sentiment</h2>
              {c.sentiment ? (
                <dl className="facts">
                  <dt>Caller, overall</dt><dd><SentimentChip score={c.sentiment.caller_overall} label={c.sentiment.caller_label} /></dd>
                  <dt>Caller, start</dt><dd><SentimentChip score={c.sentiment.caller_start} /></dd>
                  <dt>Caller, end</dt><dd><SentimentChip score={c.sentiment.caller_end} label={c.sentiment.caller_end_label} /></dd>
                  <dt>Caller, shift</dt><dd><SentimentChip score={c.sentiment.caller_shift} /></dd>
                  <dt>Agent, overall</dt><dd><SentimentChip score={c.sentiment.agent_overall} label={c.sentiment.agent_label} /></dd>
                </dl>
              ) : <p className="muted">No sentiment stored for this call.</p>}
            </div>
            <div className="panel">
              <h2>AI review</h2>
              {c.ai_review ? (
                <>
                  {c.ai_review.summary && <p>{c.ai_review.summary}</p>}
                  <p className="muted">QA score {c.ai_review.quality_score ?? "—"} · {c.ai_review.model} · {c.ai_review.prompt_version}</p>
                  {c.ai_review.checklist.length > 0 && (
                    <ul className="checklist">
                      {c.ai_review.checklist.map((i) => <li key={i.item}><b>{i.item}</b>: {i.answer ?? "—"}</li>)}
                    </ul>
                  )}
                  {c.ai_review.coaching.length > 0 && (<><h3>Coaching</h3><ul>{c.ai_review.coaching.map((t, n) => <li key={n}>{t}</li>)}</ul></>)}
                </>
              ) : <p className="muted">{c.qa_score !== null ? `QA score ${c.qa_score.toFixed(2)}. ` : ""}No AI review stored for this call.</p>}
            </div>
            {Object.keys(c.call_facts).length > 0 && (
              <div className="panel">
                <h2>Salud's call record</h2>
                <dl className="facts">
                  {Object.entries(c.call_facts).map(([k, v]) => (<Fragment key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{v ?? "—"}</dd></Fragment>))}
                </dl>
              </div>
            )}
          </div>
          <h2>Transcript</h2>
          <ol className="transcript">
            {c.transcript.map((t) => (
              <li key={t.turn} className={`turn turn-${t.role}`}>
                <div className="turn-head"><b>{t.speaker}</b>{t.sentiment !== null && <SentimentChip score={t.sentiment} label={t.sentiment_label} />}</div>
                <p>{t.text}</p>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
