// The Python side (app/api, FastAPI). Server components call it directly;
// the browser goes through the /api/py rewrite in next.config.ts.

import type { CurvesDoc, RankRow, ScoreDoc, SelectDoc, SeriesDoc, SeriesMeta, Summary, TimeDoc, ToyDoc } from "./model";

export const API_URL = process.env.AUTOVUS_API ?? "http://127.0.0.1:8765";
// In the browser, relative; on the server, absolute.
const base = () => (typeof window === "undefined" ? API_URL : "/api/py");

async function get<T>(path: string): Promise<T | null> {
  try {
    let r = await fetch(`${base()}${path}`, { cache: "no-store" });
    // 410: the Python side no longer holds your data. Send the file again, once for all waiting requests, and retry.
    if (r.status === 410 && own && (await sendAgain())) r = await fetch(`${base()}${path}`, { cache: "no-store" });
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

// Your own data on the Explore page: one CSV (the series, the label, one column per detector). The Python
// side keeps it in memory under the hash of its content and may forget it; this tab keeps the file and
// sends it again when that happens, so nothing has to survive on the other side between two requests.
export type OwnDoc = { file: string; name: string; n: number; anomalies: number; detectors: string[] };
let own: { csv: File; doc: OwnDoc } | null = null;
let again: Promise<boolean> | null = null;
export const ownDoc = () => own?.doc ?? null;
async function register(csv: File): Promise<OwnDoc | { error: string }> {
  const fd = new FormData();
  fd.append("file", csv);
  try {
    const r = await fetch(`${base()}/own`, { method: "POST", body: fd });
    const j = await r.json();
    return r.ok ? j : { error: typeof j.detail === "string" ? j.detail : "the upload failed" };
  } catch {
    return { error: "the Python side is not running" };
  }
}
export async function uploadOwn(csv: File) {
  const d = await register(csv);
  if (!("error" in d)) own = { csv, doc: d };
  return d;
}
function sendAgain() {
  again ??= register(own!.csv).then((d) => !("error" in d)).finally(() => { again = null; });
  return again;
}

export type Notes = Record<string, { note: string; updated: string }>;
export const loadNotes = () => get<Notes>("/notes");
export const saveNotes = (notes: Notes) =>
  fetch(`${base()}/notes`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ notes }) }).catch(() => {});

// The local-evaluation pages (tables from app/export_gear.py and app/export_local.py, the game live).
// L is the gear (the switch of the top bar); the tables that do not depend on it ignore it.
export const loadLocalTable = <T>(b: string, name: string, L?: number) => get<T>(`/local/table/${name}?${q({ b, L })}`);
export const loadLocalSeries = <T>(b: string, file: string, L: number) => get<T>(`/local/series/${encodeURIComponent(file)}?${q({ b, L })}`);
export const loadProfilesIndex = <T>(b: string, L: number) => get<T>(`/local/profiles?${q({ b, L })}`);
export const loadProfiles = <T>(b: string, dataset: string, L: number) => get<T>(`/local/profiles/${encodeURIComponent(dataset)}?${q({ b, L })}`);
export const loadGame = <T>(b: string, L: number, n?: number, set?: string) => get<T>(`/game/rounds?${q({ b, L, n, set })}`);
// The Draw mode of the Play page: AutoVUS on one window of a series with the score the visitor painted.
export async function scoreDrawing<T>(b: string, file: string, lo: number, hi: number, L: number, score: number[]): Promise<T | null> {
  try {
    const r = await fetch(`${base()}/draw`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ b, file, lo, hi, L, score }) });
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}
// Find the buffer: the AutoVUS value of the series with the player's buffers on anomaly k, next to AutoVUS's own.
export type BuffersDoc = { yours: number; autovus: number; seconds: number };
export const scoreBuffers = (b: string, file: string, detector: string, k: number, L: number, b_left: number, b_right: number) =>
  post<BuffersDoc>("/game/score", { b, file, detector, k, L, b_left, b_right });
// Who played (the Played so far strip of the Play page): app/data/games.json.
export type Played = {
  name: string; mode: "game" | "draw"; when: string;
  matched?: number; rounds?: number; peak?: number;          // Find the buffer
  labels?: number; types?: string[]; autovus?: number;       // Draw your detection
};
export const loadGames = () => get<{ games: Played[] }>("/games");
// keepalive: the request outlives the page (the player leaves mid-game or closes the tab).
export const saveGame = (x: Omit<Played, "when">, keepalive = false) => post<{ ok: boolean }>("/games", x, keepalive).then((d) => d?.ok === true);
async function post<T>(path: string, body: unknown, keepalive = false): Promise<T | null> {
  try {
    const r = await fetch(`${base()}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), keepalive });
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}
export type OracleNorm = "median";
export type OracleDoc = ScoreDoc & { winners: string[]; norm: OracleNorm; vusG: number[]; autoG: number; aucpr: number; auto: number[]; seconds: number };
export const loadOracle = (b: string, file: string, L: number, winners: string[], norm: OracleNorm = "median", bins = 2000, range?: [number, number]) =>
  get<OracleDoc>(`/local/oracle?${q({ b, file, L, winners: winners.join(","), norm, bins, ...(range ? { start: range[0], end: range[1] } : {}) })}`);
export const loadAucPr = (b: string, file: string) => get<Record<string, number | null>>(`/local/aucpr/${encodeURIComponent(file)}?${q({ b })}`);
