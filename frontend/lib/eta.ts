/**
 * Phase-window progress + ETA helpers (Sprint 2 / A8).
 *
 * The pipeline emits progress in known percentage bands per phase
 * (init 0-8, phase1a 8-15, phase1b 18-30, phase1c 33-45, phase2 48-60,
 * phase3 63-93, highlight 95, details 97, done 100). We use those bands
 * to project a within-phase progress bar separate from the overall bar,
 * and we compute rate-based ETAs from `started_at` (overall) and
 * `phase_started_at` (current phase).
 */

export type PhaseRange = { start: number; end: number; label: string };

const PHASES: Record<string, PhaseRange> = {
  start: { start: 0, end: 5, label: "Starting" },
  init: { start: 0, end: 8, label: "Initialising" },
  phase1a: { start: 8, end: 15, label: "Native text" },
  phase1b: { start: 18, end: 30, label: "OCR" },
  phase1c: { start: 33, end: 45, label: "Classifying pages" },
  phase2: { start: 48, end: 60, label: "ADE detection" },
  phase3: { start: 63, end: 93, label: "Measurements" },
  highlight: { start: 93, end: 95, label: "Highlighting PDF" },
  details: { start: 95, end: 99, label: "Element details" },
  done: { start: 100, end: 100, label: "Done" },
};

export function phaseRange(phase: string | null | undefined): PhaseRange | null {
  if (!phase) return null;
  return PHASES[phase] ?? null;
}

export function withinPhasePct(
  phase: string | null | undefined,
  overallPct: number,
): number {
  const r = phaseRange(phase);
  if (!r || r.end <= r.start) return 0;
  const local = ((overallPct - r.start) / (r.end - r.start)) * 100;
  return Math.max(0, Math.min(100, local));
}

/**
 * Estimate seconds remaining from a rate observation.
 * Returns null when the input is too sparse to make a sensible guess
 * (no start time, zero pct, or negative remaining).
 */
export function etaSeconds(
  startedAtIso: string | null | undefined,
  pct: number,
): number | null {
  if (!startedAtIso) return null;
  if (pct <= 0 || pct >= 100) return null;
  const started = new Date(startedAtIso).getTime();
  if (Number.isNaN(started)) return null;
  const elapsed = (Date.now() - started) / 1000;
  if (elapsed < 1) return null;
  const rate = pct / elapsed; // %/s
  if (rate <= 0) return null;
  const remaining = (100 - pct) / rate;
  if (!Number.isFinite(remaining) || remaining <= 0) return null;
  return remaining;
}

export function formatEta(secs: number | null): string | null {
  if (secs == null) return null;
  if (secs < 1) return "~now";
  if (secs < 60) return `~${Math.round(secs)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs - m * 60);
  if (m < 60) return s ? `~${m}m ${s}s` : `~${m}m`;
  const h = Math.floor(m / 60);
  const mm = m - h * 60;
  return `~${h}h ${mm}m`;
}
