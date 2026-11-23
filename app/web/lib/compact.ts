// The tables, packed for the trip from the server to the browser: one
// array per row, names replaced by indexes, numbers rounded. Client-safe.

import type { PairRow, Tables } from "./model";

export type CompactTables = Omit<Tables, "rows"> & {
  files: string[]; datasets: string[]; dets: string[];
  // [file, dataset, detector, autovus, vus_pr, vus_roc, auc_pr, auc_roc, n_anomalies, n_exact, n_early, n_late, n_missed, t_vus, t_autovus, adj_vus,
  //  n_wide]
  data: (number | null)[][];
};

const r4 = (v: number | null) => (v === null ? null : Math.round(v * 1e4) / 1e4);
const r6 = (v: number) => Math.round(v * 1e6) / 1e6;

export function compact(t: Tables): CompactTables {
  const files: string[] = [], datasets: string[] = [], dets: string[] = [];
  const fi = new Map<string, number>(), di = new Map<string, number>(), ti = new Map<string, number>();
  const idx = (m: Map<string, number>, a: string[], k: string) => { let i = m.get(k); if (i === undefined) { i = a.length; a.push(k); m.set(k, i); } return i; };
  const data = t.rows.map((r) => [
    idx(fi, files, r.file), idx(di, datasets, r.dataset), idx(ti, dets, r.detector),
    r4(r.autovus), r4(r.vus_pr), r4(r.vus_roc), r4(r.auc_pr), r4(r.auc_roc),
    r.n_anomalies, r.n_exact, r.n_early, r.n_late, r.n_missed, r6(r.t_vus), r6(r.t_autovus), r4(r.adj_vus),
    r.n_wide,
  ]);
  const { rows: _rows, ...rest } = t;
  void _rows;
  return { ...rest, files, datasets, dets, data };
}

export function expand(c: CompactTables): Tables {
  const rows: PairRow[] = c.data.map((d) => {
    const file = c.files[d[0] as number];
    return {
      file, dataset: c.datasets[d[1] as number], detector: c.dets[d[2] as number], n: c.series[file]?.n ?? 0,
      autovus: d[3], adj_vus: d[15] ?? null, vus_pr: d[4], vus_roc: d[5], auc_pr: d[6], auc_roc: d[7],
      n_anomalies: d[8] as number, n_exact: d[9] as number, n_early: d[10] as number, n_late: d[11] as number, n_wide: (d[16] ?? 0) as number, n_missed: d[12] as number,
      b_left: 0, b_right: 0, t_vus: d[13] as number, t_autovus: d[14] as number,
    };
  });
  const { files: _f, datasets: _d, dets: _t, data: _x, ...rest } = c;
  void _f; void _d; void _t; void _x;
  return { ...rest, rows };
}
