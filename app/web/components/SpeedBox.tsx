"use client";
// The speed box on the Intro: a box plot of the time per (series, detector)
// under each measure, with the total and the ratio to the original VUS
// (local/speed.json, the paper's runtime experiments), and your own label
// and score timed live on this machine.

import { useState } from "react";
import { loadTime, upload } from "@/lib/api";
import { fmt, fsec, type State, type TimeDoc } from "@/lib/model";

// box: lower whisker, first quartile, median, third quartile, upper whisker, in seconds.
export type SpeedRow = { key: string; label: string; tag: string; seconds: number; ratio: number | null; box: [number, number, number, number, number] };
export type SpeedTable = { rows: SpeedRow[] };

const COLOR: Record<string, string> = { vus: "var(--missed)", fastvus: "var(--ink)", fastvus_gpu: "var(--ink)", autovus: "var(--accent)" };
// The live rows, in the order of the box: the /time metric, its row in the box.
const LIVE: [string, string, string][] = [["vus_upstream", "VUS", "vus"], ["vus_cpu", "FastVUS", "fastvus"], ["vus_gpu", "FastVUS-GPU", "fastvus_gpu"], ["autovus", "AutoVUS", "autovus"]];
const LIVE_L = 20;
// The plot, in SVG units: the left margin holds the time axis.
const W = 600, H = 250, ML = 56, MT = 10, MB = 10;
const decade = (e: number) => (e < -3 ? `${10 ** (e + 6)} µs` : e < 0 ? `${10 ** (e + 3)} ms` : `${10 ** e} s`);

// 27512 s -> "7 h 38 min", 462 s -> "7 min 41 s", 56 s -> "56 s" (truncated, as written in the paper).
const hms = (s: number) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${Math.floor(s % 60)} s` : `${Math.floor(s)} s`);

export function SpeedBox({ state, speed }: { state: State; speed: SpeedTable }) {
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [scoreFile, setScoreFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [yours, setYours] = useState<{ n: number; anomalies: number; name: string; time: TimeDoc | null; error?: string } | null>(null);
  const run = async () => {
    if (!labelFile || !scoreFile) return;
    setBusy(true);
    const u = await upload(labelFile, scoreFile);
    if ("error" in u) { setYours({ n: 0, anomalies: 0, name: "", time: null, error: u.error }); setBusy(false); return; }
    const t = await loadTime(state.benchmark, u.file, "yours", LIVE_L, LIVE.map(([m]) => m).join(","), 3);
    setYours({ n: u.n, anomalies: u.anomalies, name: u.name, time: t });
    setBusy(false);
  };
  const live = (m: string) => yours?.time?.results.find((r) => r.metric === m) ?? null;
  // The time axis: whole decades around the whiskers.
  const e0 = Math.floor(Math.log10(Math.min(...speed.rows.map((r) => r.box[0])))), e1 = Math.ceil(Math.log10(Math.max(...speed.rows.map((r) => r.box[4]))));
  const y = (sec: number) => MT + (1 - (Math.log10(sec) - e0) / Math.max(e1 - e0, 1)) * (H - MT - MB);
  const ticks = Array.from({ length: e1 - e0 + 1 }, (_, k) => e0 + k);
  const col = (W - ML) / Math.max(speed.rows.length, 1);

  return (
    <section className="panel" aria-label="Speed" style={{ flexDirection: "row", flexWrap: "wrap", gap: 28 }}>
      <div style={{ flex: "3 1 460px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="panel-head"><h2>TSB-AD, every detector on every series</h2><span className="muted" style={{ fontSize: 12 }}>time per series and detector · log scale</span></div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }} role="img" aria-label="Box plot of the time per series and detector, per measure, log scale">
          {ticks.map((e) => (
            <g key={e}>
              <line x1={ML} y1={y(10 ** e)} x2={W} y2={y(10 ** e)} stroke="var(--border-soft)" />
              <text x={ML - 8} y={y(10 ** e) + 3.5} fontFamily="var(--mono)" fontSize={11} fill="var(--muted)" textAnchor="end">{decade(e)}</text>
            </g>
          ))}
          {speed.rows.map((r, k) => {
            const c = COLOR[r.key] ?? "var(--ink)", cx = ML + (k + 0.5) * col, bw = Math.min(64, col * 0.45);
            const [lo, q1, med, q3, hi] = r.box;
            return (
              <g key={r.key} stroke={c} strokeWidth={1.5}>
                <line x1={cx} y1={y(hi)} x2={cx} y2={y(q3)} /><line x1={cx} y1={y(q1)} x2={cx} y2={y(lo)} />
                <line x1={cx - bw / 4} y1={y(hi)} x2={cx + bw / 4} y2={y(hi)} /><line x1={cx - bw / 4} y1={y(lo)} x2={cx + bw / 4} y2={y(lo)} />
                <rect x={cx - bw / 2} y={y(q3)} width={bw} height={Math.max(y(q1) - y(q3), 1)} fill={c} fillOpacity={0.18} rx={2} />
                <line x1={cx - bw / 2} y1={y(med)} x2={cx + bw / 2} y2={y(med)} strokeWidth={3} />
                <title>{`${r.label}: median ${fsec(med)}, quartiles ${fsec(q1)} to ${fsec(q3)}, whiskers ${fsec(lo)} to ${fsec(hi)}`}</title>
              </g>
            );
          })}
        </svg>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${speed.rows.length}, minmax(0, 1fr))`, marginLeft: `${(ML / W) * 100}%`, textAlign: "center", fontSize: 13, rowGap: 2 }}>
          {speed.rows.map((r) => <span key={r.key} style={{ fontWeight: 600, color: r.key === "autovus" ? "var(--accent)" : undefined }}>{r.label}</span>)}
          {speed.rows.map((r) => <span key={r.key} className="muted" style={{ fontSize: 11 }}>{r.tag}</span>)}
          {speed.rows.map((r) => <span key={r.key} className="mono" style={{ marginTop: 6 }}>{hms(r.seconds)}</span>)}
          {speed.rows.map((r) => <span key={r.key} className="mono" style={{ fontWeight: 600, color: r.key === "autovus" ? "var(--accent)" : undefined }}>{r.ratio === null ? "total" : `${Math.round(r.ratio)}× faster`}</span>)}
        </div>
      </div>

      <div style={{ flex: "2 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="panel-head"><h2>Your data</h2>{yours && !yours.error && <span className="mono muted" style={{ fontSize: 12 }}>{yours.name} · {fmt(yours.n)} points · {yours.anomalies} anomalies</span>}</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "flex-end" }}>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, minWidth: 0 }}>label, 0 or 1 per point<input type="file" accept=".csv,.npy,.txt" onChange={(e) => setLabelFile(e.target.files?.[0] ?? null)} style={{ fontSize: 12, maxWidth: 190 }} /></label>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, minWidth: 0 }}>score, one number per point<input type="file" accept=".csv,.npy,.txt" onChange={(e) => setScoreFile(e.target.files?.[0] ?? null)} style={{ fontSize: 12, maxWidth: 190 }} /></label>
          <button type="button" className="btn btn-primary" disabled={!labelFile || !scoreFile || busy} onClick={run}>{busy ? "Timing…" : "Run and time"}</button>
        </div>
        {yours?.error && <span style={{ color: "var(--late)", fontSize: 12 }}>{yours.error}</span>}
        {yours?.time && (
          <div style={{ display: "grid", gridTemplateColumns: "max-content max-content", gap: "8px 28px", alignItems: "baseline", fontSize: 13 }}>
            {LIVE.map(([m, label, key]) => {
              const r = live(m);
              return [
                <span key={m + "l"} style={{ fontWeight: 600, color: key === "autovus" ? "var(--accent)" : undefined }}>{label}</span>,
                <span key={m + "t"} className="mono" style={{ textAlign: "right" }} title={r?.note ?? undefined}>{r?.seconds ? fsec(r.seconds) : "–"}</span>,
              ];
            })}
          </div>
        )}
      </div>
    </section>
  );
}
