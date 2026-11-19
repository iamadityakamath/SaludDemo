import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Source } from "../api/types";
import { useApi } from "../api/useApi";
import { SentimentChip, SourceBadge, Status } from "../components/Badges";

const PAGE = 50;

export function CallReviews() {
  const [source, setSource] = useState<Source | "">("");
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const calls = useApi(() => api.calls({ source: source || undefined, search: query, limit: PAGE, offset }),
    [source, query, offset]);

  return (
    <section>
      <h1>Agent Call Reviews</h1>
      <form className="filters" onSubmit={(e) => { e.preventDefault(); setOffset(0); setQuery(search); }}>
        <select value={source} onChange={(e) => { setOffset(0); setSource(e.target.value as Source | ""); }}>
          <option value="">All calls</option>
          <option value="real">Real (Salud)</option>
          <option value="synthetic">Synthetic</option>
          <option value="pasted">Pasted</option>
        </select>
        <input placeholder="Search transcripts or outcomes" value={search} onChange={(e) => setSearch(e.target.value)} />
        <button type="submit">Search</button>
      </form>
      <Status loading={calls.loading} error={calls.error} />
      {calls.data && (
        <>
          <p className="muted">{calls.data.total} calls</p>
          <table className="table">
            <thead>
              <tr><th>Call</th><th>Source</th><th>Date</th><th>Outcome</th><th>Caller sentiment at the end</th>
                <th>QA score</th><th>AI review</th><th>Turns</th></tr>
            </thead>
            <tbody>
              {calls.data.items.map((c, i) => (
                <tr key={c.key}>
                  <td><Link to={`/calls/${encodeURIComponent(c.key)}`}>Call {offset + i + 1}</Link></td>
                  <td><SourceBadge source={c.source} /></td>
                  <td>{c.contacted_at ? new Date(c.contacted_at).toLocaleDateString() : "—"}</td>
                  <td>{c.outcome ?? "—"}</td>
                  <td><SentimentChip score={c.caller_end_sentiment} label={c.caller_end_label} /></td>
                  <td>{c.qa_score !== null ? c.qa_score.toFixed(2) : "—"}</td>
                  <td>{c.has_ai_review ? "Yes" : "—"}</td>
                  <td>{c.turns}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pager">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
            <span className="muted">{offset + 1}–{Math.min(offset + PAGE, calls.data.total)} of {calls.data.total}</span>
            <button disabled={offset + PAGE >= calls.data.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
          </div>
        </>
      )}
    </section>
  );
}
