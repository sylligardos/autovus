// Types, constants and URL state shared by server and client code.

export const PAGES = [
  { path: "/", label: "Intro" },
  { path: "/play", label: "Play" },
  { path: "/local", label: "Explore" },
  { path: "/rankings", label: "Rankings" },
  { path: "/coverage", label: "Coverage" },
  { path: "/profiles", label: "Profiles" },
  { path: "/coverage-old", label: "Coverage old" },
] as const;

export const BENCHMARKS = [{ id: "tsb-ad", label: "TSB-AD" }];
export const DEFAULT_BENCHMARK = "tsb-ad";
export const BENCHMARK_ID = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
export const REFERENCES = ["Delayed oracle", "Random"];

export type State = { benchmark: string };
type Params = Record<string, string | string[] | undefined>;

export function stateFrom(params: Params): State {
  const b = typeof params.b === "string" ? params.b : DEFAULT_BENCHMARK;
  return { benchmark: BENCHMARK_ID.test(b) ? b : DEFAULT_BENCHMARK };
}
export function param(params: Params, key: string): string | null {
  const v = params[key];
  return typeof v === "string" ? v : null;
}

// Every link carries the shared state, plus what the page asks for.
export function href(pathname: string, state: State, extra: Record<string, string> = {}): string {
  const q = new URLSearchParams({ b: state.benchmark, ...extra });
  return `${pathname}?${q}`;
}

// The metrics the tables hold. AutoVUS (paper rule, one buffer for both sides) is defined over VUS-PR only.
// Adjusted VUS is VUS-PR with the fixed buffer, chance-adjusted, no selection (skill_full in the export).
export const METRICS: { key: MetricKey; label: string; accent?: boolean }[] = [
  { key: "autovus", label: "AutoVUS", accent: true },
  { key: "adj_vus", label: "Adjusted VUS" },
  { key: "vus_pr", label: "VUS-PR" },
  { key: "vus_roc", label: "VUS-ROC" },
  { key: "auc_pr", label: "AUC-PR" },
  { key: "auc_roc", label: "AUC-ROC" },
];
export type MetricKey = "autovus" | "adj_vus" | "vus_pr" | "vus_roc" | "auc_pr" | "auc_roc";
export const metricLabel = (k: string) => METRICS.find((m) => m.key === k)?.label ?? k;

// An anomaly's label from its selected buffer: wide = a buffer > 0, exact / missed = none.
// early / late only arise with per-side buffers.
export type DetectionType = "exact" | "early" | "late" | "wide" | "missed";
export const TYPE_COLOR: Record<DetectionType, string> = {
  exact: "var(--exact)", early: "var(--early)", late: "var(--late)", wide: "var(--wide)", missed: "var(--missed)",
};

export type Summary = {
  benchmark: string; label: string;
  n_series: number; n_datasets: number; n_detectors: number;
  n_anomalies: number | null; n_points: number | null;
  datasets: string[]; detectors: string[]; references: string[];
  exported: boolean; gpu: boolean;
};
export type SeriesMeta = { file: string; dataset: string; n: number | null; anomalies: number | null };
export type SeriesDoc = { file: string; n: number; start?: number; end?: number; bins: number; lo: number[]; hi: number[]; labels: [number, number][] };
export type ScoreDoc = { file: string; detector: string; n: number; start?: number; end?: number; bins: number; lo: number[]; hi: number[] };
export type Anomaly = { k: number; start: number; end: number; window: [number, number]; b_left: number; b_right: number; type: DetectionType; core_ratio: number };
export type SelectDoc = {
  file: string; detector: string; L: number; seconds: number;
  autovus: number; vus_adapt: number; vus_full: number; skill_full: number; random_full: number; anomalies: Anomaly[];
};
export type SideCurves = { b: number; idx: number; ratio: number; vus_curve: number[]; random_curve: number[]; rule_curve: number[] };
export type CurvesDoc = { k: number; buffers: number[]; sides: Record<string, SideCurves>; zoom: { from: number; to: number; bins: number; lo: number[]; hi: number[] } };
export type RankRow = { detector: string } & Record<MetricKey, number | null>;
export type ToyDoc = {
  case: string; cases: string[]; label: number[]; score: number[]; seconds: number; autovus: number; vus_full: number;
  anomaly: Anomaly; buffers: number[]; sides: Record<string, { b: number; idx: number; vus_curve: number[]; random_curve: number[]; rule_curve: number[] }>;
};
export type TimeRow = { metric: string; seconds: number | null; value: number | null; note: string | null };
export type TimeDoc = { file: string; detector: string; n: number; L: number; repeats: number; results: TimeRow[] };

// One (series, detector) of the exported tables.
export type PairRow = {
  file: string; dataset: string; detector: string; n: number;
  autovus: number | null; adj_vus: number | null; vus_pr: number | null; vus_roc: number | null; auc_pr: number | null; auc_roc: number | null;
  n_anomalies: number; n_exact: number; n_early: number; n_late: number; n_wide: number; n_missed: number;
  b_left: number; b_right: number; t_vus: number; t_autovus: number;
};
export type Tables = {
  L: number; variants: Record<string, [string, string]>; detectors: string[]; references: string[];
  rows: PairRow[]; series: Record<string, { dataset: string; n: number; anomalies: number }>;
  summary: Record<string, unknown> | null;
  baselines: { L: number; rows: Record<string, number | string | null>[] } | null;
  gpu: { L: number; detector: string; seconds: Record<string, number | null> } | null;
};

export const fmt = (n: number) => n.toLocaleString("en-US");
export const f2 = (v: number | null | undefined) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(2));
export const f3 = (v: number | null | undefined) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(3));
export const fsec = (s: number | null | undefined) => {
  if (s === null || s === undefined) return "–";
  if (s < 1e-3) return `${(s * 1e6).toFixed(0)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(s < 0.01 ? 1 : 0)} ms`;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  if (s < 3600) return `${(s / 60).toFixed(1)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} h`;
  return `${(s / 86400).toFixed(1)} days`;
};
export const shortFile = (f: string) => {
  const m = f.match(/^(\d+)_([^_]+)_id_(\d+)_([^_]+)/);
  return m ? `${m[2]} ${m[1]} · ${m[4]}` : f.replace(/\.csv$/, "");
};
