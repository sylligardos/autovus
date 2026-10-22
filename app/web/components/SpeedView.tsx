"use client";
// The Speed page: every metric timed on the whole benchmark, every series
// against its length, by dataset, and your own data timed live.

import Link from "next/link";
import { useMemo, useState } from "react";
import { loadTime, upload } from "@/lib/api";
import { fmt, fsec, href, type State, type Tables, type TimeDoc } from "@/lib/model";
import { median } from "@/lib/stats";
import { expand, type CompactTables } from "@/lib/compact";

type Bar = { label: string; seconds: number | null; tag: string; color: string; note?: string };
const LOG_MIN = 60, LOG_MAX = 10 * 86400;
const logw = (s: number | null, lo = LOG_MIN, hi = LOG_MAX) => (s === null || s <= 0 ? 0 : Math.max(0, Math.min(100, ((Math.log10(s) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * 100)));

export function SpeedView({ state, tables: packed }: { state: State; tables: CompactTables }) {
  const tables = useMemo(() => expand(packed), [packed]);
  const [scope, setScope] = useState("");
  const datasets = useMemo(() => Array.from(new Set(tables.rows.map((r) => r.dataset))).sort(), [tables.rows]);
  const rows = useMemo(() => tables.rows.filter((r) => !scope || r.dataset === scope), [tables.rows, scope]);
  const nDet = new Set(rows.map((r) => r.detector)).size || 1;
  const files = useMemo(() => Array.from(new Set(rows.map((r) => r.file))), [rows]);

  // Whole-benchmark totals, from the export's own timings.
  const sum = (k: "t_vus" | "t_autovus") => rows.reduce((s, r) => s + (r[k] ?? 0), 0);
  const gpuTotal = tables.gpu ? files.reduce((s, f) => s + (tables.gpu!.seconds[f] ?? 0), 0) * nDet : null;
  const bars: Bar[] = [
    { label: "AutoVUS · FastVUS GPU", seconds: null, tag: "not timed", color: "var(--accent)" },
    { label: "AutoVUS · FastVUS CPU", seconds: sum("t_autovus"), tag: "exact", color: "var(--accent)" },
    { label: "VUS-PR · FastVUS GPU", seconds: gpuTotal, tag: "approximate, one detector per series times " + nDet, color: "var(--ink)" },
    { label: "VUS-PR · FastVUS CPU", seconds: sum("t_vus"), tag: "exact", color: "var(--ink)" },
  ].filter((b) => b.seconds !== null);

  // The slow baselines, on the subset they were timed on, with FastVUS on the same subset.
  const sub = useMemo(() => {
    const b = tables.baselines;
    if (!b) return null;
    const keys = new Set(b.rows.map((r) => `${r.file}|${r.detector}`));
    const same = rows.filter((r) => keys.has(`${r.file}|${r.detector}`));
    const tot = (k: string) => b.rows.filter((r) => !scope || same.some((x) => x.file === r.file)).reduce((s, r) => s + ((r[k] as number) ?? 0), 0);
    const cnt = (k: string) => b.rows.filter((r) => (!scope || same.some((x) => x.file === r.file)) && typeof r[k] === "number").length;
    const nfiles = new Set(b.rows.filter((r) => !scope || same.some((x) => x.file === r.file)).map((r) => r.file)).size;
    const out: Bar[] = [
      { label: "AutoVUS · FastVUS CPU", seconds: same.reduce((s, r) => s + r.t_autovus, 0), tag: `exact · ${same.length} pairs`, color: "var(--accent)" },
      { label: "VUS-PR · FastVUS CPU", seconds: same.reduce((s, r) => s + r.t_vus, 0), tag: `exact · ${same.length} pairs`, color: "var(--ink)" },
      { label: "VUS-PR · upstream", seconds: tot("t_vus"), tag: `exact, 250 thresholds · ${cnt("t_vus")} pairs`, color: "var(--missed)" },
      { label: "Range-F1", seconds: tot("t_rf"), tag: `one threshold · ${cnt("t_rf")} pairs`, color: "var(--missed)" },
      { label: "Affiliation-F1", seconds: tot("t_affiliation"), tag: `one threshold · ${cnt("t_affiliation")} pairs`, color: "var(--missed)" },
      { label: "PATE", seconds: cnt("t_pate") ? tot("t_pate") : null, tag: `threshold-free · ${cnt("t_pate")} pairs`, color: "var(--missed)" },
    ].filter((x) => x.seconds !== null);
    return { out, nfiles, nPairs: same.length, maxT: b.rows.reduce((m, r) => Math.max(m, (r.n as number) ?? 0), 0) };
  }, [tables.baselines, rows, scope]);

  // Per series: median over the detectors.
  const perSeries = useMemo(() => files.map((f) => {
    const rs = rows.filter((r) => r.file === f);
    return { file: f, n: rs[0]?.n ?? 0, cpu: median(rs.map((r) => r.t_vus)), auto: median(rs.map((r) => r.t_autovus)), gpu: tables.gpu?.seconds[f] ?? null };
  }), [files, rows, tables.gpu]);
  const W = 720, H = 300;
  const sx = (n: number) => 60 + ((Math.log10(Math.max(n, 1000)) - 3) / 3) * 600;
  const sy = (t: number) => 20 + ((3 - Math.log10(Math.max(t, 1e-3))) / 6) * 260;

  const byDataset = useMemo(() => datasets.map((d) => {
    const rs = tables.rows.filter((r) => r.dataset === d);
    const fs = Array.from(new Set(rs.map((r) => r.file)));
    return { dataset: d, n: fs.length, len: median(fs.map((f) => tables.series[f]?.n ?? 0)) ?? 0, cpu: rs.reduce((s, r) => s + r.t_vus, 0), auto: rs.reduce((s, r) => s + r.t_autovus, 0), gpu: tables.gpu ? fs.reduce((s, f) => s + (tables.gpu!.seconds[f] ?? 0), 0) * nDet : null };
  }).sort((a, b) => b.len - a.len), [datasets, tables, nDet]);

  // Your data.
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [scoreFile, setScoreFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [yours, setYours] = useState<{ file: string; n: number; anomalies: number; name: string; time: TimeDoc | null; error?: string } | null>(null);
  const run = async () => {
    if (!labelFile || !scoreFile) return;
    setBusy(true);
    const u = await upload(labelFile, scoreFile);
    if ("error" in u) { setYours({ file: "", n: 0, anomalies: 0, name: "", time: null, error: u.error }); setBusy(false); return; }
    const t = await loadTime(state.benchmark, u.file, "yours", tables.L, "autovus,vus_cpu,vus_gpu,vus_upstream,auc_pr,rf,affiliation,pate", 3);
    setYours({ file: u.file, n: u.n, anomalies: u.anomalies, name: u.name, time: t });
    setBusy(false);
  };
  const tl: Record<string, string> = { autovus: "AutoVUS · FastVUS CPU", vus_cpu: "VUS-PR · FastVUS CPU", vus_gpu: "VUS-PR · FastVUS GPU", vus_upstream: "VUS-PR · upstream", auc_pr: "AUC-PR", rf: "Range-F1", affiliation: "Affiliation-F1", pate: "PATE" };

  return (
    <>
      <aside className="rail" aria-label="Controls">
        <div className="field"><span>Scope</span>
          <select className="select" aria-label="Scope" value={scope} onChange={(e) => setScope(e.target.value)}><option value="">whole benchmark</option>{datasets.map((d) => <option key={d} value={d}>{d}</option>)}</select>
        </div>
        <div className="field" style={{ gap: 6, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <span>Your data</span>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12 }}>label, 0 or 1 per point<input type="file" accept=".csv,.npy,.txt" onChange={(e) => setLabelFile(e.target.files?.[0] ?? null)} style={{ fontSize: 12 }} /></label>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12 }}>score, one number per point<input type="file" accept=".csv,.npy,.txt" onChange={(e) => setScoreFile(e.target.files?.[0] ?? null)} style={{ fontSize: 12 }} /></label>
          <button type="button" className="btn" style={{ height: 34, borderColor: "var(--accent)", color: "var(--accent)", fontWeight: 600 }} disabled={!labelFile || !scoreFile || busy} onClick={run}>{busy ? "Timing…" : "Run and time"}</button>
        </div>
      </aside>

      <main className="content">
        <section className="panel" aria-label="Whole benchmark">
          <div className="panel-head">
            <h2>{scope || "TSB-AD"}, every detector on every series</h2>
            <span className="muted" style={{ fontSize: 12 }}>{fmt(files.length)} series · {nDet} detectors · {fmt(rows.length)} evaluations · L = {tables.L} · log scale</span>
          </div>
          <BarList bars={bars} lo={1} hi={LOG_MAX} ticks={["1 s", "1 min", "1 h", "1 day", "10 days"]} />
        </section>

        {sub && (
          <section className="panel" aria-label="Against the baselines">
            <div className="panel-head">
              <h2>Against the baselines, on the same {sub.nfiles} series</h2>
              <span className="muted" style={{ fontSize: 12 }}>up to {fmt(sub.maxT)} points each · log scale</span>
            </div>
            <BarList bars={sub.out} lo={0.1} hi={86400} ticks={["0.1 s", "10 s", "10 min", "1 day"]} />
          </section>
        )}

        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "stretch" }}>
          <section className="panel" style={{ flex: "1 1 520px", minWidth: 0 }} aria-label="Per series">
            <div className="panel-head"><h2>Every series, time against length</h2><span className="muted" style={{ fontSize: 12 }}>median over the detectors · L = {tables.L}</span></div>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="auto" style={{ display: "block", background: "var(--subtle)", borderRadius: 4 }} role="img" aria-label="Runtime of each series against its length">
              {[20, 106.7, 193.3].map((y) => <line key={y} x1={60} y1={y} x2={660} y2={y} stroke="var(--border-soft)" />)}
              <line x1={60} y1={280} x2={660} y2={280} stroke="var(--border)" /><line x1={60} y1={20} x2={60} y2={280} stroke="var(--border)" />
              {[["1000 s", 24], ["10 s", 110], ["0.1 s", 197], ["1 ms", 284]].map(([t, y]) => <text key={t} x={54} y={y as number} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="end">{t}</text>)}
              {[["1k", 60], ["10k", 260], ["100k", 460], ["1M points", 660]].map(([t, x]) => <text key={t} x={x as number} y={296} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="middle">{t}</text>)}
              {perSeries.map((s) => <g key={s.file}>
                {s.cpu !== null && <circle cx={sx(s.n)} cy={sy(s.cpu)} r={2.6} fill="var(--ink)" opacity={0.55}><title>{`${s.file} · VUS-PR CPU ${fsec(s.cpu)}`}</title></circle>}
                {s.auto !== null && <circle cx={sx(s.n)} cy={sy(s.auto)} r={2.6} fill="var(--accent)" opacity={0.55}><title>{`${s.file} · AutoVUS CPU ${fsec(s.auto)}`}</title></circle>}
                {s.gpu !== null && <circle cx={sx(s.n)} cy={sy(s.gpu)} r={2.6} fill="var(--exact)" opacity={0.55}><title>{`${s.file} · VUS-PR GPU ${fsec(s.gpu)}`}</title></circle>}
              </g>)}
              {yours?.time && yours.time.results.filter((r) => r.seconds).map((r) => <circle key={r.metric} cx={sx(yours.n)} cy={sy(r.seconds!)} r={6} fill="none" stroke="var(--late)" strokeWidth={2}><title>{`yours · ${tl[r.metric]} ${fsec(r.seconds)}`}</title></circle>)}
            </svg>
            <div className="muted" style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12 }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ borderRadius: "50%", width: 10, height: 10, background: "var(--accent)" }} />AutoVUS, FastVUS CPU</span>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ borderRadius: "50%", width: 10, height: 10, background: "var(--ink)" }} />VUS-PR, FastVUS CPU</span>
              {tables.gpu && <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ borderRadius: "50%", width: 10, height: 10, background: "var(--exact)" }} />VUS-PR, FastVUS GPU, one detector</span>}
              {yours?.time && <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ borderRadius: "50%", width: 10, height: 10, border: "2px solid var(--late)" }} />your data</span>}
            </div>
          </section>
          <section className="panel" style={{ flex: "1 1 380px", minWidth: 0 }} aria-label="By dataset">
            <div className="panel-head"><h2>By dataset</h2><span className="muted" style={{ fontSize: 12 }}>all detectors, longest series first</span></div>
            <div style={{ overflowX: "auto" }}>
              <table className="table">
                <thead><tr><th>dataset</th><th className="r">series</th><th className="r">median length</th><th className="r">VUS-PR CPU</th><th className="r" style={{ color: "var(--accent)" }}>AutoVUS</th>{tables.gpu && <th className="r">GPU</th>}</tr></thead>
                <tbody className="mono">
                  {byDataset.map((d) => <tr key={d.dataset}><td className="sans" style={{ fontWeight: 600 }}>{d.dataset}</td><td className="r">{d.n}</td><td className="r">{fmt(Math.round(d.len))}</td><td className="r">{fsec(d.cpu)}</td><td className="r">{fsec(d.auto)}</td>{tables.gpu && <td className="r">{fsec(d.gpu)}</td>}</tr>)}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        {yours && (
          <section className="panel panel-accent" aria-label="Your data" style={{ gap: 8 }}>
            <div className="panel-head">
              <h2 style={{ color: "var(--accent)" }}>Your data, timed just now</h2>
              {yours.error ? <span style={{ color: "var(--late)", fontSize: 12 }}>{yours.error}</span> : <span className="mono" style={{ fontSize: 12 }}>{yours.name} · {fmt(yours.n)} points · {yours.anomalies} anomalies · L = {tables.L}</span>}
              {!yours.error && <Link href={href("/series", state, { s: yours.file })} style={{ marginLeft: "auto", fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Open in Series as &quot;yours&quot;</Link>}
            </div>
            {yours.time && <BarList bars={yours.time.results.map((r) => ({ label: tl[r.metric] ?? r.metric, seconds: r.seconds, tag: r.note ?? (r.value !== null ? `value ${r.value.toFixed(3)}` : ""), color: r.metric === "autovus" ? "var(--accent)" : r.metric.startsWith("vus_") && !r.metric.endsWith("upstream") ? "var(--ink)" : "var(--missed)" }))} lo={1e-3} hi={100} ticks={["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"]} />}
          </section>
        )}
      </main>
    </>
  );
}

function BarList({ bars, lo, hi, ticks }: { bars: Bar[]; lo: number; hi: number; ticks: string[] }) {
  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "220px minmax(0, 1fr) 72px 200px", gap: "8px 10px", alignItems: "center", fontSize: 12, gridAutoRows: 24 }}>
        {bars.map((b) => (
          <Bar key={b.label} b={b} lo={lo} hi={hi} />
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "220px minmax(0, 1fr) 72px 200px", gap: "0 10px", fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)" }}>
        <span /><span style={{ display: "flex", justifyContent: "space-between" }}>{ticks.map((t) => <span key={t}>{t}</span>)}</span><span /><span />
      </div>
    </>
  );
}
function Bar({ b, lo, hi }: { b: Bar; lo: number; hi: number }) {
  return (
    <>
      <span style={{ fontWeight: b.color === "var(--accent)" ? 600 : 400, color: b.color === "var(--accent)" ? "var(--accent)" : undefined }}>{b.label}</span>
      <span style={{ height: 14, background: "var(--border-soft)", borderRadius: 3, overflow: "hidden" }}><span style={{ display: "block", width: `${logw(b.seconds, lo, hi)}%`, height: "100%", background: b.color }} /></span>
      <span className="mono" style={{ textAlign: "right" }}>{b.seconds === null ? "–" : fsec(b.seconds)}</span>
      <span className="muted" style={{ fontSize: 11 }}>{b.tag}</span>
    </>
  );
}
