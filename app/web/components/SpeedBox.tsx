"use client";
// The speed box on the Intro: a box plot of the time per (series, detector)
// under each measure, with the total and the ratio to the original VUS
// (local/speed.json, the paper's runtime experiments). Your own data is
// on the Explore page.

import { fsec } from "@/lib/model";

// box: lower whisker, first quartile, median, third quartile, upper whisker, in seconds.
export type SpeedRow = { key: string; label: string; tag: string; seconds: number; ratio: number | null; box: [number, number, number, number, number] };
export type SpeedTable = { rows: SpeedRow[] };

const COLOR: Record<string, string> = { vus: "var(--missed)", fastvus: "var(--ink)", fastvus_gpu: "var(--ink)", autovus: "var(--accent)" };
// The plot, in SVG units: the left margin holds the time axis.
const W = 600, H = 250, ML = 56, MT = 10, MB = 10;
const decade = (e: number) => (e < -3 ? `${10 ** (e + 6)} µs` : e < 0 ? `${10 ** (e + 3)} ms` : `${10 ** e} s`);

// 27512 s -> "7 h 38 min", 462 s -> "7 min 41 s", 56 s -> "56 s" (truncated, as written in the paper).
const hms = (s: number) => (s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${Math.floor(s % 60)} s` : `${Math.floor(s)} s`);

export function SpeedBox({ speed }: { speed: SpeedTable }) {
  // The time axis: whole decades around the whiskers.
  const e0 = Math.floor(Math.log10(Math.min(...speed.rows.map((r) => r.box[0])))), e1 = Math.ceil(Math.log10(Math.max(...speed.rows.map((r) => r.box[4]))));
  const y = (sec: number) => MT + (1 - (Math.log10(sec) - e0) / Math.max(e1 - e0, 1)) * (H - MT - MB);
  const ticks = Array.from({ length: e1 - e0 + 1 }, (_, k) => e0 + k);
  const col = (W - ML) / Math.max(speed.rows.length, 1);

  return (
    <section className="panel" aria-label="Speed">
      <div style={{ width: "100%", maxWidth: 760, display: "flex", flexDirection: "column", gap: 10 }}>
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

    </section>
  );
}
