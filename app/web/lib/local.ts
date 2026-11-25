// Types and helpers shared by the local-evaluation pages: Local, Finder,
// Coverage, Buffers, Suspicious and the game. The numbers are the per-anomaly
// AutoVUS (chance-adjusted local VUS-PR at the gear L, per-side buffers, variant C) of app/export_gear.py.

import type { DetectionType } from "./model";

export type LocalSeries = {
  file: string; anomalies: [number, number][]; detectors: string[];
  auto: number[][]; autoG: number[]; vusG: number[][];
};
export type PairsTable = { detectors: string[]; series: string[]; s: number[]; d: number[]; K: number[]; a: number[]; m: number[]; n: number[] };
export type StitchedTable = { L: number; score: Record<string, number> };
export type CoverTable = { detectors: string[]; series: string[]; K: number[]; M: number[][] };
export type BuffersTable = { L: number; types: DetectionType[]; series: string[]; detectors: string[]; rows: number[][] };
export type OddItem = {
  file: string; dataset: string; K: number; D: number; k: number; len: number; worst: number;
  this: number; others: number; q: number; lo: number; hi: number; start: number; end: number; values: number[]; score: number[];
};
export type OddTable = { q: number; gap: number; tested: Record<string, { series: number; anomalies: number }>; items: OddItem[] };
export type GameRound = {
  file: string; detector: string; k: number; L: number; lo: number; hi: number; start: number; end: number;
  b_left: number; b_right: number; type: DetectionType; seconds: number; autovus?: number; values: number[]; score: number[];
  gain_left: number[]; gain_right: number[];       // the gain at every buffer 0..L per side; b_left / b_right are their first argmax
  labels: [number, number][];                      // every label of the window, [start, end] inclusive
};
// Profiles (app/export_profiles.py): the anomalies of one dataset, their scores (x100) and the groupings.
export type ProfileGraph = { label: string; pos: [number, number][]; edges: [number, number, number][]; group: number[]; counts: number[]; stability: number };
export type ProfilesIndex = { dataset: string; anomalies: number; series: number; detectors?: number; profiles?: number; stability?: number; too_many?: boolean }[];
export type ProfilesTable = {
  dataset: string; k: number; found: number; dropped: number; left_out: string[]; detectors: string[]; series: string[];
  s: number[]; a: number[]; start: number[]; end: number[]; X: number[][];
  graph: ProfileGraph;
  series_graph: { pos: [number, number][]; edges: [number, number, number][] };
  cover: { all: number; rows: {
    detector: string; score: number; seconds: number | null; best: number;
    similar: { detector: string; score: number; seconds: number | null; corr: number } | null;
  }[] };
};
export type GameDoc = { L: number; pool?: number; rounds: GameRound[] };
// The Draw mode (POST /draw): the buffers AutoVUS picks on a painted score; the two values are null while nothing is painted.
export type DrawDoc = {
  seconds: number; autovus: number | null; vus_full: number | null;
  anomalies: { k: number; start: number; end: number; b_left: number; b_right: number; type: DetectionType }[];
};

export const datasetOf = (file: string) => file.split("_")[1] ?? "";
export const seriesName = (file: string) => file.replace(/\.csv$/, "");
export const MAX_SCORED = 20;        // anomalies in view above this: no per-anomaly numbers

// Local AutoVUS colour: dark orange (0), light grey (0.5), dark blue (1).
const LOW = [180, 83, 9], MID = [243, 244, 246], HIGH = [31, 79, 191];
export function gradient(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const [p, q, u] = t < 0.5 ? [LOW, MID, t / 0.5] : [MID, HIGH, (t - 0.5) / 0.5];
  return `rgb(${p.map((x, i) => Math.round(x + (q[i] - x) * u)).join(",")})`;
}
export function chipColors(v: number): { bg: string; fg: string } {
  const t = Math.max(0, Math.min(1, v));
  return { bg: gradient(t), fg: t >= 0.9 || t <= 0.03 ? "#FFFFFF" : "#152033" };
}
export const GRADIENT_CSS = "linear-gradient(90deg, rgb(180,83,9), rgb(243,244,246), rgb(31,79,191))";

// Detector colours in a set (Coverage): mid and dark tones that read on white.
export const SET_COLORS = ["#1F4FBF", "#E07B1A", "#178F6F", "#8E5BD0", "#8A2D1F", "#C2417A", "#4D6B12", "#5B6B82"];

// The oracle: each anomaly scored by the best detector on it. Ties go to the
// detector that reaches the best score on the most anomalies of the series.
// M[d][a] is detector d's local AutoVUS on anomaly a (NaN = no run).
export type Oracle = { cov: number; set: number[]; wins: Map<number, number>; winner: number[]; alone: number[]; single: number };
export function oracle(M: number[][]): Oracle {
  const D = M.length, K = M[0]?.length ?? 0;
  const v = (d: number, a: number) => (Number.isFinite(M[d][a]) ? M[d][a] : 0);
  const best = Array.from({ length: K }, (_, a) => Math.max(...M.map((_, d) => v(d, a))));
  const reach = M.map((_, d) => best.reduce((s, b, a) => s + (v(d, a) === b ? 1 : 0), 0));
  const winner = best.map((b, a) => {
    let w = -1;
    for (let d = 0; d < D; d++) if (v(d, a) === b && (w < 0 || reach[d] > reach[w])) w = d;
    return w;
  });
  const wins = new Map<number, number>();
  for (const w of winner) wins.set(w, (wins.get(w) ?? 0) + 1);
  const set = [...wins.keys()].sort((x, y) => (wins.get(y) ?? 0) - (wins.get(x) ?? 0));
  const alone = M.map((_, d) => (K ? Array.from({ length: K }, (_, a) => v(d, a)).reduce((s, x) => s + x, 0) / K : 0));
  const single = alone.reduce((bi, x, d) => (x > alone[bi] ? d : bi), 0);
  return { cov: K ? best.reduce((s, x) => s + x, 0) / K : 0, set, wins, winner, alone, single };
}

// Coverage of a set: the mean over anomalies of the best score in the set.
export function coverage(M: number[][], S: number[]): number {
  const K = M[0]?.length ?? 0;
  let s = 0;
  for (let a = 0; a < K; a++) {
    let m = -Infinity;
    for (const d of S) { const x = Number.isFinite(M[d][a]) ? M[d][a] : 0; if (x > m) m = x; }
    s += m;
  }
  return K ? s / K : 0;
}

// The best set of each size k = 1 .. maxK: exact search up to k = 4, then one detector added at a time.
export function bestSets(M: number[][], maxK: number): { k: number; S: number[]; cov: number }[] {
  const D = M.length;
  const rows: { k: number; S: number[]; cov: number }[] = [];
  let prev: number[] = [];
  for (let k = 1; k <= maxK; k++) {
    let bestS: number[] = [], bestV = -Infinity;
    if (k <= 4) {
      const idx: number[] = [];
      const rec = (start: number) => {
        if (idx.length === k) { const c = coverage(M, idx); if (c > bestV) { bestV = c; bestS = idx.slice(); } return; }
        for (let d = start; d < D; d++) { idx.push(d); rec(d + 1); idx.pop(); }
      };
      rec(0);
    } else {
      for (let d = 0; d < D; d++) {
        if (prev.includes(d)) continue;
        const c = coverage(M, [...prev, d]);
        if (c > bestV) { bestV = c; bestS = [...prev, d]; }
      }
    }
    rows.push({ k, S: bestS, cov: bestV });
    prev = bestS;
  }
  return rows;
}

// The minimum set: detector d finds anomaly a when its local AutoVUS is at
// least tau (compared in hundredths, as stored). The fewest detectors such
// that every anomaly found by some detector is found by one of the set; among
// the sets of that size, the one with the highest coverage. Exact search:
// sizes 0, 1, 2, ... and, at each size, every set, each one once.
// S is ordered by the anomalies each detector takes; finder[a] is the set's
// best detector among those that find a, or -1 when no detector finds it.
export type MinSet = { S: number[]; finder: number[]; none: number[] };
export function minSet(M: number[][], tau: number): MinSet {
  const D = M.length, K = M[0]?.length ?? 0, cut = Math.round(tau * 100);
  const finds = (d: number, a: number) => Math.round(M[d][a] * 100) >= cut;
  // The distinct groups of detectors that find an anomaly; a group that contains another is implied by it.
  const groups = new Map<string, number[]>();
  const none: number[] = [];
  for (let a = 0; a < K; a++) {
    const g: number[] = [];
    for (let d = 0; d < D; d++) if (finds(d, a)) g.push(d);
    if (g.length) groups.set(g.join(","), g); else none.push(a);
  }
  const all = [...groups.values()].sort((x, y) => x.length - y.length);
  const need = all.filter((g, i) => !all.some((h, j) => j !== i && h.length <= g.length && (h.length < g.length || j < i) && h.every((d) => g.includes(d))));

  const chosen = new Uint8Array(D), barred = new Uint8Array(D);
  const sets: number[][] = [];
  const rec = (left: number) => {
    // The group not yet found that has the fewest detectors still allowed.
    let pick: number[] | null = null, fewest = Infinity;
    for (const g of need) {
      let hit = false, free = 0;
      for (const d of g) { if (chosen[d]) { hit = true; break; } if (!barred[d]) free++; }
      if (!hit && free < fewest) { pick = g; fewest = free; }
    }
    if (!pick) { const S: number[] = []; for (let d = 0; d < D; d++) if (chosen[d]) S.push(d); sets.push(S); return; }
    if (!left || !fewest) return;
    const tried: number[] = [];
    for (const d of pick) {
      if (barred[d]) continue;
      chosen[d] = 1; rec(left - 1); chosen[d] = 0;
      barred[d] = 1; tried.push(d);
    }
    for (const d of tried) barred[d] = 0;
  };
  for (let k = 0; k <= D && !sets.length; k++) rec(k);

  let S = sets[0] ?? [], best = -Infinity;
  if (sets.length > 1) for (const c of sets) { const v = coverage(M, c); if (v > best) { best = v; S = c; } }
  const v = (d: number, a: number) => (Number.isFinite(M[d][a]) ? M[d][a] : 0);
  const finder = Array.from({ length: K }, (_, a) => S.reduce((w, d) => (finds(d, a) && (w < 0 || v(d, a) > v(w, a)) ? d : w), -1));
  const takes = new Map<number, number>();
  for (const w of finder) if (w >= 0) takes.set(w, (takes.get(w) ?? 0) + 1);
  return { S: S.slice().sort((x, y) => (takes.get(y) ?? 0) - (takes.get(x) ?? 0)), finder, none };
}

// Anomaly markers above a strip: centres in percent of the view, pushed apart by at least `gap` percent.
export function spreadMarkers(centres: number[], gap: number): number[] {
  const order = centres.map((_, i) => i).sort((a, b) => centres[a] - centres[b]);
  const pos = centres.slice();
  let prev = -gap / 2;
  for (const i of order) { pos[i] = Math.max(centres[i], prev + gap); prev = pos[i]; }
  let next = 100 - gap / 2;
  for (const i of [...order].reverse()) { pos[i] = Math.min(pos[i], next); next = pos[i] - gap; }
  return pos;
}
