// Number and colour helpers, the same rules as the team dashboard.
export const cap = (s?: string | null) => (s ? s[0].toUpperCase() + s.slice(1) : s ?? "");
export const fmt = (x: number) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(2);
/** Text colour for a score: muted near zero, otherwise the positive or negative colour. */
export const sVar = (x: number) => (Math.abs(x) < 0.15 ? "var(--muted)" : `var(${x > 0 ? "--pos" : "--neg"})`);
/** Tint for a scored turn or chart dot: stronger the further from zero. */
export const sColor = (x: number) => (Math.abs(x) < 0.15 ? "var(--line)"
  : `color-mix(in srgb, var(${x > 0 ? "--pos" : "--neg"}) ${Math.round(35 + 65 * Math.min(1, Math.abs(x)))}%, var(--line))`);
