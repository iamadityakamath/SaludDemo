import { useLayoutEffect, useRef, useState } from "react";
import type { Sentiment } from "../api/types";
import { cap, fmt, sColor, sVar } from "../lib/format";

/** Caller sentiment card: how the caller ended, start / shift / overall, lowest and highest turn, and a line chart
 *  of caller (solid, dots) and agent (dashed) turn scores. Clicking a dot or a lowest/highest link jumps to the turn. */
export function SentimentCard({ data, onJump }: { data: Sentiment; onJump: (turn: number) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const c = data.call!, turns = data.turns ?? [];
  const n = Math.max(c.call_turns, 2);
  const W = Math.max(width - 28, 200), H = 46, P = 5;
  const x = (t: number) => P + ((W - 2 * P) * (t - 1)) / (n - 1), y = (v: number) => H / 2 - (H / 2 - P) * v;
  const line = (role: string) => turns.filter((t) => t.role === role).map((t) => `${x(t.turn).toFixed(1)},${y(t.score).toFixed(1)}`).join(" ");
  return (
    <div id="senti">
      <div className="senti-head">
        <span className="tag">Caller sentiment</span>
        <span className="big" style={{ color: sVar(c.caller_end) }}>Ended {cap(c.caller_end_label)} ({fmt(c.caller_end)})</span>
        <span className="muted">Started {fmt(c.caller_start)} · Shift {fmt(c.caller_shift)} · Overall {c.caller_label} ({fmt(c.caller_overall)})</span>
        <span className="right"><span className="key"><i />Caller</span><span className="key"><i className="agent" />Agent</span>
          <button type="button" onClick={() => onJump(c.caller_min_turn)}>Lowest T{c.caller_min_turn} ({fmt(c.caller_min_score)}) →</button>
          <button type="button" onClick={() => onJump(c.caller_max_turn)}>Highest T{c.caller_max_turn} ({fmt(c.caller_max_score)}) →</button></span>
      </div>
      <div className="senti-chart" ref={box}>
        {width > 0 && (
          <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Caller sentiment over the call">
            <line x1={P} x2={W - P} y1={y(0)} y2={y(0)} style={{ stroke: "#cfd6c6", strokeWidth: 1 }} />
            <polyline points={line("agent")} style={{ fill: "none", stroke: "var(--faint)", strokeWidth: 1.5, strokeDasharray: "4 3", opacity: 0.8 }} />
            <polyline points={line("caller")} style={{ fill: "none", stroke: "var(--ink)", strokeWidth: 2.2, opacity: 0.8 }} />
            {turns.filter((t) => t.role === "caller").map((t) => (
              <circle key={t.turn} cx={x(t.turn).toFixed(1)} cy={y(t.score).toFixed(1)} r={3.6} onClick={() => onJump(t.turn)}
                style={{ fill: sColor(t.score), stroke: "var(--panel)", strokeWidth: 1.2 }}><title>T{t.turn}: {fmt(t.score)}</title></circle>
            ))}
          </svg>
        )}
      </div>
    </div>
  );
}
