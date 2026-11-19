// Agent Call Reviews: the team dashboard (analyze.html), ported to React with the same layout, class names and
// behaviour. Pick a call on the left; its stored sentiment, AI summary, transcript, QA checklist and coaching show.
// Calls missing a part are filled in when opened; uploads are saved and analyzed right away; Re-analyze replaces both.

import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Analysis, CallList, CallResult, CallRow, Sentiment, Transcript } from "../api/types";
import { AppHeader } from "../components/AppHeader";
import { SentimentCard } from "../components/SentimentCard";
import { cap, fmt, sColor, sVar } from "../lib/format";

const LABEL: Record<string, string> = {
  greeting: "Appropriate greeting", culture: "Culture & values", capturing_data: "Captures key data",
  understanding_needs: "Understands needs", correct_info: "Correct information",
  first_contact_resolution: "First-contact resolution", compliance: "Privacy compliance", general: "General",
};
const GROUPS: [CallRow["source"], string][] = [["new", "Uploaded calls"], ["real", "Real calls"], ["salud", "Salud package calls"],
  ["synthetic", "Synthetic calls"]];
const reEsc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

interface State {
  calls: CallRow[]; byId: Record<string, CallRow>; cur: string | null; text: Record<string, Transcript>;
  res: Record<string, CallResult>; q: string; cfg: Partial<CallList>; busy: Set<string>; tried: Set<string>;
  failed: Set<string>; poll: number; status: React.ReactNode; redoing: boolean; listError: string | null;
  turnsMsg: string | null; mark: { turn: number; quote?: string } | null; flash: number | null;
}

export function AgentDashboard() {
  // One mutable state object, as in the original page, so async steps always see the latest values; tick re-renders.
  const S = useRef<State>({
    calls: [], byId: {}, cur: null, text: {}, res: {}, q: "", cfg: {}, busy: new Set(), tried: new Set(),
    failed: new Set(), poll: 0, status: "", redoing: false, listError: null, turnsMsg: null, mark: null, flash: null,
  }).current;
  const [, setTick] = useState(0);
  const render = () => setTick((t) => t + 1);
  const scrollRef = useRef<HTMLDivElement>(null);
  const redoDlg = useRef<HTMLDialogElement>(null);
  const newDlg = useRef<HTMLDialogElement>(null);
  const files = useRef<HTMLInputElement>(null);
  const [newText, setNewText] = useState("");
  const [newErr, setNewErr] = useState("");
  const [uploading, setUploading] = useState(false);

  // A call's state: "processing" (being analyzed, here or by the server), "incomplete" (missing a part) or "done".
  const state = (c: CallRow) => (S.busy.has(c.call_id) ? "processing" : S.failed.has(c.call_id) ? "incomplete" : c.status);
  const missing = (r?: CallResult) => !r || r.sentiment.none || (r.analysis.none && S.cfg.llm);

  async function loadList(select?: string) {
    const d = await api.list();
    S.cfg = d; S.calls = d.calls; S.byId = Object.fromEntries(d.calls.map((c) => [c.call_id, c])); S.listError = null;
    render();
    // While any call is processing, check again in a few seconds; the open call refreshes once it's done.
    window.clearTimeout(S.poll);
    if (S.calls.some((c) => state(c) === "processing")) S.poll = window.setTimeout(() => loadList().catch(() => {}), 4000);
    const cur = S.cur ? S.byId[S.cur] : undefined;
    if (!select && cur && state(cur) === "done" && missing(S.res[cur.call_id])) refresh(cur.call_id);
    if (select) show(select);
  }

  function setStatus(id: string, node?: React.ReactNode) {
    if (S.cur !== id) return;
    const c = S.byId[id];
    if (node !== undefined) S.status = node;
    else if (c && state(c) === "processing") S.status = <><span className="spin" />Processing: results appear here automatically</>;
    render();
  }

  async function show(id: string) {
    S.cur = id; S.status = ""; S.mark = null; S.turnsMsg = S.text[id] ? null : "Loading transcript…"; render();
    setStatus(id);
    try {
      // Stored results come straight from the database; nothing is computed here.
      const [t, r] = await Promise.all([S.text[id] || api.call(id), missing(S.res[id]) ? api.result(id) : S.res[id]]);
      S.text[id] = t; if (!S.busy.has(id)) S.res[id] = r;
    } catch (e) {
      if (S.cur === id) { S.turnsMsg = `Couldn't load this call: ${(e as Error).message}`; render(); }
      return;
    }
    if (S.cur !== id) return;
    S.turnsMsg = null; scrollRef.current?.scrollTo({ top: 0 }); setStatus(id);
    // A part still missing on a call nobody is processing: finish it now, once per page visit.
    const cur = S.byId[id];
    if (cur && state(cur) === "incomplete" && missing(S.res[id]) && !S.tried.has(id)) run(id, false);
  }

  async function refresh(id: string) {   // a call finished processing elsewhere: load its stored results
    try { S.res[id] = await api.result(id); } catch { return; }
    if (S.cur === id) { S.status = ""; render(); }
  }

  // An in-page dialog rather than window.confirm(), which some embedded browsers block.
  function confirmRedo(): Promise<boolean> {
    const dlg = redoDlg.current!;
    return new Promise((resolve) => {
      dlg.returnValue = "";
      dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
      dlg.showModal();
    });
  }

  // redo false: fill in only what's missing (uploads, incomplete calls). redo true: Re-analyze, replacing both parts.
  async function run(id: string, redo: boolean) {
    S.busy.add(id); S.tried.add(id); S.failed.delete(id);
    if (S.cur === id) {
      S.redoing = redo;
      S.status = <><span className="spin" />{redo ? "Re-analyzing" : "Processing"}: this can take up to 30 seconds</>;
    }
    render();
    let err: Error | null = null;
    try { S.res[id] = await api.run(id, redo); } catch (e) { err = e as Error; }
    S.busy.delete(id);
    if (err || missing(S.res[id])) S.failed.add(id);
    await loadList().catch(() => {});
    if (S.cur !== id) return;
    S.redoing = false;
    S.status = err ? `${redo ? "Re-analyze" : "Analysis"} failed: ${err.message}` : redo ? "Re-analyzed just now" : "";
    render();
  }

  async function reanalyze() {
    const id = S.cur, c = id ? S.byId[id] : undefined;
    if (!id) return;
    if (c && c.source !== "new" && !(await confirmRedo())) return;
    run(id, true);
  }

  function jumpTo(n: number, quote?: string) {
    const el = document.getElementById("turn-" + n);
    if (!el) return;
    S.mark = { turn: n, quote }; S.flash = n; render();
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    window.setTimeout(() => { if (S.flash === n) { S.flash = null; render(); } }, 1200);
  }

  // Upload: save the transcripts, then analyze them one at a time (the provider's rate limit).
  function openUpload() { setNewErr(""); if (files.current) files.current.value = ""; newDlg.current?.showModal(); }
  async function submitUpload(e: React.FormEvent) {
    e.preventDefault();
    const fromFiles = await Promise.all([...(files.current?.files ?? [])].map((f) => f.text()));
    const texts = [...fromFiles, newText].map((t) => t.trim()).filter(Boolean);
    if (!texts.length) { setNewErr("Paste a transcript or choose .txt files."); return; }
    setUploading(true); setNewErr("");
    try {
      const { ids } = await api.upload(texts);
      newDlg.current?.close(); setNewText(""); if (files.current) files.current.value = "";
      ids.forEach((id) => S.busy.add(id));          // processing from the moment they're saved
      await loadList(ids[0]);
      for (const id of ids) await run(id, false);
    } catch (err) { setNewErr("Couldn't upload: " + (err as Error).message); }
    setUploading(false);
  }

  useEffect(() => {
    loadList().catch((e) => { S.listError = `Couldn't load calls: ${(e as Error).message}`; render(); });
    return () => window.clearTimeout(S.poll);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- rendering ----------
  const cur = S.cur ? S.byId[S.cur] : undefined;
  const r = S.cur ? S.res[S.cur] : undefined;
  const t = S.cur ? S.text[S.cur] : undefined;
  const a: Analysis | null = r && !r.analysis.none ? r.analysis : null;
  const sd: Sentiment | null = r && !r.sentiment.none ? r.sentiment : null;
  const busyCur = !!cur && state(cur) === "processing";

  return (
    <div className="app">
      <AppHeader title="Agent Dashboard" sub="Call sentiment & AI quality review">
        {S.cfg.llm === false && (
          <span className="chip off" title="No LLM API key is configured on this server, so new calls get sentiment but no AI checklist.">
            AI review off: sentiment only for new calls</span>
        )}
        {S.cfg.me && <span className="chip">{cap(S.cfg.me)}</span>}
      </AppHeader>
      <div className="body">
        <nav>
          <div className="navtop">
            <button type="button" className="btn primary" onClick={openUpload}>+ Upload calls</button>
            <input type="search" placeholder="Search calls…" aria-label="Search calls" value={S.q}
              onChange={(e) => { S.q = e.target.value; render(); }} />
          </div>
          {CallListView()}
        </nav>
        <section className="center">
          <div className="toolbar">
            <button type="button" className="btn" disabled={!S.cur || busyCur || !t} onClick={reanalyze}
              title="Score sentiment and run the AI review again, replacing the stored results">
              {S.redoing ? <><span className="spin" />Re-analyzing…</> : "↻ Re-analyze"}</button>
            <span className="fid">{cur ? (cur.source === "new" ? `Uploaded call · ${cur.file_id}` : cur.file_id) : S.cur ?? "Pick a call on the left"}</span>
            <span className="status">{S.status}</span>
          </div>
          <div id="top">
            {r && Kpis()}
            {r && sd?.call && sd.call.caller_turns ? <SentimentCard data={sd} onJump={(n) => jumpTo(n)} /> : null}
          </div>
          <div id="scroll" ref={scrollRef}>
            {r && r.notes?.map((n, i) => <div className="notes" key={i}>{n}</div>)}
            {a?.summary && <div className="summary"><span className="tag">AI summary</span><span>{a.summary}</span></div>}
            {Turns()}
          </div>
        </section>
        <aside id="results">{Results()}</aside>
      </div>

      <dialog ref={redoDlg}>
        <form method="dialog">
          <h3>Re-analyze this Phase 1 call?</h3>
          <p>This replaces its stored sentiment and AI review. The comparison with raters' tags will then use the new review.</p>
          <div className="row"><span />
            <span><button type="submit" className="btn" value="cancel">Cancel</button>{" "}
              <button type="submit" className="btn primary" value="ok" autoFocus>Re-analyze</button></span></div>
        </form>
      </dialog>

      <dialog ref={newDlg}>
        <form method="dialog" onSubmit={submitUpload}>
          <h3>Upload calls</h3>
          <p>Choose one or more <code>.txt</code> transcripts, or paste one below. Each is saved to the team database and
            analyzed straight away. One turn per paragraph, each starting with the speaker, e.g. <code>Agent: …</code> then
            a blank line, then <code>Patient: …</code>.</p>
          <input type="file" ref={files} accept=".txt,text/plain" multiple aria-label="Transcript files" />
          <textarea value={newText} onChange={(e) => setNewText(e.target.value)}
            placeholder={"Agent: Thank you for calling, how can I help?\n\nPatient: Hi, I have a question about my bill."} />
          <div className="warn">{S.cfg.llm
            ? "Transcripts are saved to the team database and analyzed right away, which sends them to the AI review provider (Groq). Redacted text only."
            : "Transcripts are saved to the team database and analyzed right away. The AI review isn't set up on this server, so only sentiment is computed."}</div>
          <div className="row"><span id="new-err">{newErr}</span>
            <span><button type="button" className="btn" onClick={() => newDlg.current?.close()}>Cancel</button>{" "}
              <button type="submit" className="btn primary" disabled={uploading}>
                {uploading ? <><span className="spin" />Uploading…</> : "Upload"}</button></span></div>
        </form>
      </dialog>
    </div>
  );

  function CallListView() {
    if (S.listError) return <div id="list"><div className="empty">{S.listError}</div></div>;
    const q = S.q.toLowerCase();
    const groups = GROUPS.map(([src, title]) => [title, S.calls.filter((c) => c.source === src && (!q || c.file_id.toLowerCase().includes(q)))] as const)
      .filter(([, rows]) => rows.length);
    return (
      <div id="list">
        {S.calls.length > 0 && !groups.length && <div className="empty">No calls match.</div>}
        {groups.map(([title, rows]) => (
          <div key={title}>
            <div className="grp">{title} · {rows.length}</div>
            {rows.map((c) => {
              const when = c.added_at ? new Date(c.added_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "";
              const st = state(c);
              return (
                <button type="button" key={c.call_id} aria-current={c.call_id === S.cur} onClick={() => show(c.call_id)}>
                  <span className={`src ${c.source}`}>{c.source}</span>
                  <span style={{ minWidth: 0 }}>
                    <div className="fid">{c.source === "new" ? when : c.file_id}</div>
                    {c.source === "new" && <div className="meta">{st === "processing" ? "Processing…" : c.added_by ? "by " + cap(c.added_by) : ""}</div>}
                    {c.source === "salud" && c.added_at && <div className="meta">{new Date(c.added_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}</div>}
                  </span>
                  {st === "processing" ? <span className="rowspin" title="Processing: sentiment and AI review" />
                    : st === "incomplete" ? <span className="rowwarn" title="Not fully analyzed yet. Open the call to finish it.">!</span>
                    : <span className={`sdot ${c.end_label ?? ""}`} title={c.end_label ? "Customer ended " + c.end_label : "Analyzed"} />}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    );
  }

  function Kpis() {
    const c = sd?.call;
    const answers = a ? Object.values(a.checklist || {}).map((x) => x && x.answer) : [];
    const yes = answers.filter((x) => x === "Yes").length, no = answers.filter((x) => x === "No").length;
    const tile = (k: string, v: React.ReactNode, s: React.ReactNode, color?: string) => (
      <div className="kpi"><div className="k">{k}</div><div className="v" style={color ? { color } : undefined}>{v}</div><div className="s">{s}</div></div>);
    return (
      <div id="kpis"><div className="kpis">
        {tile("Customer ended", c && c.caller_turns ? cap(c.caller_end_label) : "—", c && c.caller_turns ? fmt(c.caller_end) : "no caller turns", c ? sVar(c.caller_end) : "")}
        {tile("Sentiment shift", c && c.caller_turns ? fmt(c.caller_shift) : "—", c && c.caller_turns ? `from ${fmt(c.caller_start)} at the start` : "", c ? sVar(c.caller_shift) : "")}
        {tile("QA score", yes + no ? Math.round((100 * yes) / (yes + no)) + "%" : "—", a ? `${yes} of ${yes + no} items met` : "no AI review")}
        {tile("Outcome", a && a.outcome?.label ? cap(a.outcome.label.replaceAll("_", " ")) : "—", a ? "AI's call outcome" : "no AI review")}
      </div></div>
    );
  }

  function Turns() {
    if (!S.cur) return <div id="turns"><div className="empty">Pick a call on the left, or paste a new one.</div></div>;
    if (S.turnsMsg || !t) return <div id="turns"><div className="empty">{S.turnsMsg ?? "Loading transcript…"}</div></div>;
    const scores = new Map((sd?.turns ?? []).map((x) => [x.turn, x]));
    return (
      <div id="turns">
        {t.turns.map((x) => {
          const kind = /^agent/i.test(x.speaker) ? "agent" : /automated|system|note/i.test(x.speaker) ? "system" : "caller";
          const s = scores.get(x.n);
          const cls = ["turn", kind, s ? "scored" : "", S.flash === x.n ? "flash" : ""].filter(Boolean).join(" ");
          return (
            <div className={cls} id={`turn-${x.n}`} key={x.n} style={s ? ({ "--tint": sColor(s.score) } as React.CSSProperties) : undefined}>
              <div className="tn">T{x.n}</div>
              <div className="bubble">
                {x.speaker !== "Note" && (
                  <div className="who">{x.speaker}
                    {s && <span className={`sscore ${Math.abs(s.score) < 0.15 ? "neu" : s.score > 0 ? "pos" : "neg"}`}
                      title={`Sentiment ${fmt(s.score)} (${s.label})`}>{fmt(s.score)}</span>}
                  </div>
                )}
                <div className="txt">{highlight(x.text, S.mark && S.mark.turn === x.n ? S.mark.quote : undefined)}</div>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  function Results() {
    if (!r) {
      return <div className="empty">{cur && state(cur) === "processing"
        ? <><span className="spin" />Processing: the QA checklist, outcome and coaching tips appear here when they're ready.</>
        : S.cur ? "Loading this call's results…" : "Pick a call to see its QA checklist, call outcome and coaching tips."}</div>;
    }
    if (!a) return <div className="empty">No AI review for this call{(r.notes || []).length ? " (see the note on the left)" : ""}.</div>;
    const defs = S.cfg.definitions || {};
    const answers = Object.values(a.checklist || {}).map((x) => x && x.answer);
    const yes = answers.filter((x) => x === "Yes").length, no = answers.filter((x) => x === "No").length;
    const chips = (evs?: { turn: number; quote: string }[]) => (evs || []).map((e, i) => (
      <button type="button" key={i} onClick={() => jumpTo(e.turn, e.quote)}><b>T{e.turn}</b>“{e.quote}”</button>));
    return (
      <>
        <section>
          <h2>QA checklist <span>{yes} Yes · {no} No · {answers.length - yes - no} N/A</span></h2>
          {(S.cfg.items || Object.keys(a.checklist || {})).map((k) => {
            const x = a.checklist?.[k] || { answer: null }, ans = x.answer || "—";
            return (
              <div className="card" key={k}>
                <div className="card-head"><span>{LABEL[k] || k}<span className="info" title={defs[k] || ""}>i</span></span>
                  <span className={`ans ${ans === "N/A" ? "NA" : ans}`}>{ans}</span></div>
                {x.reason && <div className="reason">{x.reason}</div>}
                <div className="evs">{chips(x.evidence)}</div>
              </div>
            );
          })}
        </section>
        <section>
          <h2>Coaching for the agent</h2>
          {(a.coaching || []).length ? (a.coaching || []).map((tip, i) => (
            <div className="card tip" key={i}><span className="item">{LABEL[tip.item] || tip.item}</span>
              {tip.turn ? <button type="button" className="linkbtn go" onClick={() => jumpTo(tip.turn!)}>Go to T{tip.turn} →</button> : null}
              <div>{tip.tip}</div></div>
          )) : <div className="reason">No coaching tips for this call.</div>}
        </section>
        <div className="reason" style={{ fontSize: 11 }}>AI review: {a.model || ""}{sd ? " · Sentiment: " + sd.model : ""}</div>
      </>
    );
  }
}

/** The turn's text with the cited quote marked (whitespace-tolerant, case-insensitive), as in the original page. */
function highlight(text: string, quote?: string) {
  if (!quote) return text;
  const re = new RegExp(reEsc(quote).replace(/\s+/g, "\\s+"), "i");
  const m = re.exec(text);
  if (!m) return text;
  return <>{text.slice(0, m.index)}<mark>{m[0]}</mark>{text.slice(m.index + m[0].length)}</>;
}
