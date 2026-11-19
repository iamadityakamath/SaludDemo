import type { CallDetail, CallList, Health, Phase2Summary, Source } from "./types";

// All requests go to /api; the Vite dev server forwards them to the FastAPI backend.
async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/health"),
  calls: (params: { source?: Source; search?: string; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && q.set(k, String(v)));
    return get<CallList>(`/calls?${q}`);
  },
  // Keys can contain "/" (Phase 1 call ids), so they are always encoded.
  call: (key: string) => get<CallDetail>(`/calls/${encodeURIComponent(key)}`),
  phase2Summary: () => get<Phase2Summary>("/phase2/summary"),
};
