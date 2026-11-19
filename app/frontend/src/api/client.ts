import type { CallList, CallResult, Health, Phase2Summary, Transcript } from "./types";

// All requests go to /api; the Vite dev server forwards them to the FastAPI backend.
async function request<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`/api${path}`, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || d.error || `HTTP ${r.status}`);
  return d as T;
}

const q = (id: string) => "?id=" + encodeURIComponent(id);

export const api = {
  health: () => request<Health>("/health"),
  list: () => request<CallList>("/analyze/list"),
  call: (id: string) => request<Transcript>("/call" + q(id)),
  result: (id: string) => request<CallResult>("/result" + q(id)),
  run: (id: string, redo: boolean) => request<CallResult>("/analyze/run", { id, redo }),
  upload: (transcripts: string[]) => request<{ ids: string[] }>("/analyze/new", { transcripts }),
  phase2Summary: () => request<Phase2Summary>("/phase2/summary"),
};

export const LOGO = "/salud-logo.png";   // app/frontend/public/salud-logo.png
