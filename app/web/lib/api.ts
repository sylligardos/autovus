// The Python side (app/api, FastAPI). Server components call it directly;
// the browser goes through the /api/py rewrite in next.config.ts.

import type { CurvesDoc, RankRow, ScoreDoc, SelectDoc, SeriesDoc, SeriesMeta, Summary, TimeDoc, ToyDoc } from "./model";

export const API_URL = process.env.AUTOVUS_API ?? "http://127.0.0.1:8765";
// In the browser, relative; on the server, absolute.
const base = () => (typeof window === "undefined" ? API_URL : "/api/py");

async function get<T>(path: string): Promise<T | null> {
  try {
    const r = await fetch(`${base()}${path}`, { cache: "no-store" });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}
const q = (o: Record<string, string | number | boolean | undefined>) =>
  new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)])).toString();

export const loadSummary = (b: string) => get<Summary>(`/summary?${q({ b })}`);
export const loadSeriesList = (b: string, dataset: string) => get<SeriesMeta[]>(`/series?${q({ b, dataset })}`);
export const loadSeriesDoc = (b: string, file: string, bins = 2000, range?: [number, number]) =>
  get<SeriesDoc>(`/series/${encodeURIComponent(file)}?${q({ b, bins, ...(range ? { start: range[0], end: range[1] } : {}) })}`);
export const loadScoreDoc = (b: string, file: string, detector: string, bins = 2000, range?: [number, number]) =>
  get<ScoreDoc>(`/score?${q({ b, file, detector, bins, ...(range ? { start: range[0], end: range[1] } : {}) })}`);
export const loadSelect = (b: string, file: string, detector: string, L: number) => get<SelectDoc>(`/select?${q({ b, file, detector, L })}`);
export const loadCurves = (b: string, file: string, detector: string, k: number, L: number) => get<CurvesDoc>(`/curves?${q({ b, file, detector, k, L })}`);
export const loadRank = (b: string, file: string, L: number, metric: string) => get<RankRow[]>(`/rank?${q({ b, file, L, metric })}`);
export const loadToy = (c: string, L: number) => get<ToyDoc>(`/toy?${q({ case: c, L })}`);
export const loadTime = (b: string, file: string, detector: string, L: number, metrics: string, repeats = 3) =>
  get<TimeDoc>(`/time?${q({ b, file, detector, L, metrics, repeats })}`);

export async function upload(label: File, score: File): Promise<{ id: string; file: string; n: number; anomalies: number; name: string } | { error: string }> {
  const fd = new FormData();
  fd.append("label", label);
  fd.append("score", score);
  try {
    const r = await fetch(`${base()}/upload`, { method: "POST", body: fd });
    const j = await r.json();
    return r.ok ? j : { error: j.detail ?? "upload failed" };
  } catch {
    return { error: "the Python side is not running" };
  }
}

export type Notes = Record<string, { note: string; updated: string }>;
export const loadNotes = () => get<Notes>("/notes");
export const saveNotes = (notes: Notes) =>
  fetch(`${base()}/notes`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ notes }) }).catch(() => {});

// The local-evaluation pages (tables from app/export_local.py, the game live).
export const loadLocalTable = <T>(b: string, name: string) => get<T>(`/local/table/${name}?${q({ b })}`);
export const loadLocalSeries = <T>(b: string, file: string) => get<T>(`/local/series/${encodeURIComponent(file)}?${q({ b })}`);
export const loadProfilesIndex = <T>(b: string) => get<T>(`/local/profiles?${q({ b })}`);
export const loadProfiles = <T>(b: string, dataset: string) => get<T>(`/local/profiles/${encodeURIComponent(dataset)}?${q({ b })}`);
export const loadGame = <T>(b: string, n?: number, set?: string) => get<T>(`/game/rounds?${q({ b, n, set })}`);
export type OracleNorm = "median";
export type OracleDoc = ScoreDoc & { winners: string[]; norm: OracleNorm; vusG: number[]; autoG: number; aucpr: number; auto: number[]; seconds: number };
export const loadOracle = (b: string, file: string, winners: string[], norm: OracleNorm = "median", bins = 2000, range?: [number, number]) =>
  get<OracleDoc>(`/local/oracle?${q({ b, file, winners: winners.join(","), norm, bins, ...(range ? { start: range[0], end: range[1] } : {}) })}`);
export const loadAucPr = (b: string, file: string) => get<Record<string, number | null>>(`/local/aucpr/${encodeURIComponent(file)}?${q({ b })}`);
