// Rankings and agreement from the pair rows. Client-safe, no I/O.

import type { MetricKey, PairRow } from "./model";

export function mean(xs: number[]): number | null {
  const v = xs.filter((x) => Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
export function median(xs: number[]): number | null {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** Detector -> aggregate of a metric over the rows (mean or median), best first. */
export function rankDetectors(rows: PairRow[], metric: MetricKey, agg: "mean" | "median" = "mean"): { detector: string; value: number; n: number }[] {
  const by = new Map<string, number[]>();
  for (const r of rows) {
    const v = r[metric];
    if (v === null) continue;
    if (!by.has(r.detector)) by.set(r.detector, []);
    by.get(r.detector)!.push(v);
  }
  const out = Array.from(by.entries()).map(([detector, xs]) => ({ detector, value: (agg === "mean" ? mean(xs) : median(xs)) ?? NaN, n: xs.length }));
  return out.filter((x) => Number.isFinite(x.value)).sort((a, b) => b.value - a.value);
}

/** Kendall's tau-b between two rankings given as detector -> value maps (common keys only). */
export function kendall(a: Map<string, number>, b: Map<string, number>): number | null {
  const keys = Array.from(a.keys()).filter((k) => b.has(k));
  if (keys.length < 2) return null;
  let conc = 0, disc = 0, tieA = 0, tieB = 0;
  for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++) {
    const da = a.get(keys[i])! - a.get(keys[j])!, db = b.get(keys[i])! - b.get(keys[j])!;
    if (da === 0 && db === 0) continue;
    if (da === 0) tieA++; else if (db === 0) tieB++; else if (Math.sign(da) === Math.sign(db)) conc++; else disc++;
  }
  const den = Math.sqrt((conc + disc + tieA) * (conc + disc + tieB));
  return den ? (conc - disc) / den : null;
}

/** Weighted Kendall's tau (Vigna 2015) with hyperbolic weights 1/(r+1), additive, as scipy.stats.weightedtau defaults:
 * a pair weighs w(r_i) + w(r_j), r the 0-based rank by decreasing (x, y); the result averages the (x, y) and (y, x) orders. */
export function weightedKendall(a: Map<string, number>, b: Map<string, number>): number | null {
  const keys = Array.from(a.keys()).filter((k) => b.has(k));
  if (keys.length < 2) return null;
  const x = keys.map((k) => a.get(k)!), y = keys.map((k) => b.get(k)!);
  const one = (p: number[], q: number[]) => {
    const order = keys.map((_, i) => i).sort((i, j) => p[j] - p[i] || q[j] - q[i]);
    const w = new Array<number>(keys.length);
    order.forEach((i, r) => { w[i] = 1 / (r + 1); });
    let num = 0, dp = 0, dq = 0;
    for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++) {
      const wij = w[i] + w[j], sp = Math.sign(p[i] - p[j]), sq = Math.sign(q[i] - q[j]);
      num += wij * sp * sq; if (sp) dp += wij; if (sq) dq += wij;
    }
    return dp && dq ? num / Math.sqrt(dp * dq) : null;
  };
  const t1 = one(x, y), t2 = one(y, x);
  return t1 === null || t2 === null ? null : (t1 + t2) / 2;
}

export const toMap = (r: { detector: string; value: number }[]) => new Map(r.map((x) => [x.detector, x.value]));

/** Per series, the rank of every detector under a metric (1 = best). */
export function ranksPerSeries(rows: PairRow[], metric: MetricKey): Map<string, Map<string, number>> {
  const by = new Map<string, PairRow[]>();
  for (const r of rows) {
    if (r[metric] === null) continue;
    if (!by.has(r.file)) by.set(r.file, []);
    by.get(r.file)!.push(r);
  }
  const out = new Map<string, Map<string, number>>();
  for (const [file, rs] of by) {
    rs.sort((a, b) => (b[metric] as number) - (a[metric] as number));
    out.set(file, new Map(rs.map((r, i) => [r.detector, i + 1])));
  }
  return out;
}
