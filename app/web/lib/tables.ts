// The exported tables (app/export.py), read from app/data/<benchmark>/.
// Server only. Cached per process; the export rewrites the files rarely.

import { promises as fs } from "fs";
import path from "path";
import type { PairRow, Tables } from "./model";

export const DATA_ROOT = process.env.AUTOVUS_DATA ?? path.join(process.cwd(), "..", "data");
const cache = new Map<string, { mtime: number; tables: Tables }>();

async function readJson<T>(...parts: string[]): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(DATA_ROOT, ...parts), "utf8")) as T;
  } catch {
    return null;
  }
}

type ScoresFile = {
  L: number; variants: Record<string, [string, string]>; detectors: string[]; references: string[];
  files: Record<string, Record<string, Record<string, number | string | null>>>;
};

export async function loadTables(benchmark: string, scoresFile = "scores.json"): Promise<Tables | null> {
  let mtime = 0;
  try { mtime = (await fs.stat(path.join(DATA_ROOT, benchmark, scoresFile))).mtimeMs; } catch { return null; }
  const key = `${benchmark}/${scoresFile}`;
  const hit = cache.get(key);
  if (hit && hit.mtime === mtime) return hit.tables;
  const scores = await readJson<ScoresFile>(benchmark, scoresFile);
  const series = await readJson<Tables["series"]>(benchmark, "series.json");
  if (!scores || !series) return null;
  const rows: PairRow[] = [];
  for (const [file, dets] of Object.entries(scores.files)) {
    const s = series[file];
    if (!s) continue;
    for (const [detector, v] of Object.entries(dets)) {
      if (v.error !== undefined || v.autovus === undefined) continue;
      const num = (k: string) => (typeof v[k] === "number" ? (v[k] as number) : null);
      rows.push({
        file, dataset: s.dataset, detector, n: s.n,
        autovus: num("autovus"), adj_vus: num("skill_full"), vus_pr: num("vus_pr"), vus_roc: num("vus_roc"), auc_pr: num("auc_pr"), auc_roc: num("auc_roc"),
        n_anomalies: num("n_anomalies") ?? 0, n_exact: num("n_exact") ?? 0, n_early: num("n_early") ?? 0, n_late: num("n_late") ?? 0, n_wide: num("n_wide") ?? 0, n_missed: num("n_missed") ?? 0,
        b_left: num("b_left") ?? 0, b_right: num("b_right") ?? 0, t_vus: num("t_vus") ?? 0, t_autovus: num("t_autovus") ?? 0,
      });
    }
  }
  const tables: Tables = {
    L: scores.L, variants: scores.variants ?? {}, detectors: scores.detectors, references: scores.references,
    rows, series,
    summary: await readJson<Record<string, unknown>>(benchmark, "summary.json"),
    baselines: await readJson<Tables["baselines"]>(benchmark, "times_baselines.json"),
    gpu: await readJson<Tables["gpu"]>(benchmark, "times_gpu.json"),
  };
  cache.set(key, { mtime, tables });
  return tables;
}
