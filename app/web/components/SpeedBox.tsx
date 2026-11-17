"use client";
// The speed box on the Intro: every detector on every series under each
// measure, against the original VUS (local/speed.json, the paper's runtime
// experiments), and your own label and score timed live on this machine.

import { useState } from "react";
import { loadTime, upload } from "@/lib/api";
import { fmt, fsec, type State, type TimeDoc } from "@/lib/model";

export type SpeedRow = { key: string; label: string; tag: string; seconds: number; ratio: number | null };
export type SpeedTable = { rows: SpeedRow[] };

const COLOR: Record<string, string> = { vus: "var(--missed)", fastvus: "var(--ink)", fastvus_gpu: "var(--ink)", autovus: "var(--accent)" };
// The live rows, in the order of the box: the /time metric, its row in the box.
const LIVE: [string, string, string][] = [["vus_upstream", "VUS", "vus"], ["vus_cpu", "FastVUS", "fastvus"], ["vus_gpu", "FastVUS-GPU", "fastvus_gpu"], ["autovus", "AutoVUS", "autovus"]];
const LIVE_L = 20;
const COLS = "110px minmax(0, 1fr) 96px 56px";

const logw = (s: number | null, lo: number, hi: number) => (s === null || s <= 0 ? 0 : Math.round(100 * Math.max(0, Math.min(100, ((Math.log10(s) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * 100))) / 100);
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

  return (
    <section className="panel" aria-label="Speed" style={{ flexDirection: "row", flexWrap: "wrap", gap: 28 }}>
      <div style={{ flex: "3 1 460px", minWidth: 0, display: "flex", flexDirection: "column", gap: 10 }}>
        <div className="panel-head"><h2>TSB-AD, every detector on every series</h2></div>
        <div style={{ display: "grid", gridTemplateColumns: COLS, gap: "8px 12px", alignItems: "center", fontSize: 13 }}>
          <span /><span /><span className="eyebrow" style={{ textAlign: "right" }}>total</span><span className="eyebrow" style={{ textAlign: "right" }}>vs VUS</span>
          {speed.rows.map((r) => {
            const accent = r.key === "autovus";
            return [
              <span key={r.key + "l"} style={{ display: "flex", flexDirection: "column", lineHeight: 1.2 }}><span style={{ fontWeight: 600, color: accent ? "var(--accent)" : undefined }}>{r.label}</span><span className="muted" style={{ fontSize: 11 }}>{r.tag}</span></span>,
              <span key={r.key + "b"} style={{ height: 14, background: "var(--border-soft)", borderRadius: 3, overflow: "hidden" }}><span style={{ display: "block", width: `${logw(r.seconds, 10, 36000)}%`, height: "100%", background: COLOR[r.key] ?? "var(--ink)" }} /></span>,
              <span key={r.key + "t"} className="mono" style={{ textAlign: "right" }}>{hms(r.seconds)}</span>,
              <span key={r.key + "r"} className="mono" style={{ textAlign: "right", fontWeight: 600, color: accent ? "var(--accent)" : undefined }}>{r.ratio === null ? "" : `${Math.round(r.ratio)}×`}</span>,
            ];
          })}
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
          <div style={{ display: "grid", gridTemplateColumns: "110px minmax(0, 1fr) 96px", gap: "8px 12px", alignItems: "center", fontSize: 13 }}>
            <span /><span /><span className="eyebrow" style={{ textAlign: "right" }}>time</span>
            {LIVE.map(([m, label, key]) => {
              const r = live(m);
              return [
                <span key={m + "l"} style={{ fontWeight: 600, color: key === "autovus" ? "var(--accent)" : undefined }}>{label}</span>,
                <span key={m + "b"} style={{ height: 14, background: "var(--border-soft)", borderRadius: 3, overflow: "hidden" }}><span style={{ display: "block", width: `${logw(r?.seconds ?? null, 1e-4, 100)}%`, height: "100%", background: COLOR[key] }} /></span>,
                <span key={m + "t"} className="mono" style={{ textAlign: "right" }} title={r?.note ?? undefined}>{r?.seconds ? fsec(r.seconds) : "–"}</span>,
              ];
            })}
          </div>
        )}
      </div>
    </section>
  );
}
