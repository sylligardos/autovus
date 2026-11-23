"use client";
// Coverage: for every threshold on AutoVUS Local, how many series have one
// detector that detects every anomaly, how many need several detectors
// together, and how many hold an anomaly no detector detects. Series with at
// least two anomalies. A detector detects an anomaly when its AutoVUS Local on
// it is at least the threshold. Below, every series as a point: how far one
// detector is enough against how far the detectors together are.

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { loadLocalTable } from "@/lib/api";
import { href, type State } from "@/lib/model";
import { seriesName, type CoverTable } from "@/lib/local";
import { GapScatter } from "./GapScatter";

const H = 380, PAD = { l: 52, r: 16, t: 12, b: 40 };
const BANDS = [
  { key: "one", label: "One detector detects every anomaly", color: "var(--exact)" },
  { key: "several", label: "Only several detectors together", color: "var(--accent)" },
  { key: "none", label: "An anomaly no detector detects", color: "var(--missed)" },
] as const;

export function CoverageCurve({ state }: { state: State }) {
  const b = state.benchmark;
  const [t, setT] = useState<CoverTable | null>(null);
  const [error, setError] = useState(false);
  const [W, setW] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    let live = true;
    loadLocalTable<CoverTable>(b, "cover").then((x) => { if (live) { if (x) setT(x); else setError(true); } });
    return () => { live = false; };
  }, [b]);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.floor(el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Per series, in hundredths: alone = the best detector's lowest score over the anomalies (one detector
  // detects all up to this threshold), together = the lowest best score of an anomaly (a set does, up to this one).
  const view = useMemo(() => {
    if (!t) return null;
    const alone: number[] = [], together: number[] = [];
    const cases: { file: string; K: number; alone: number; together: number; dets: string[] }[] = [];
    t.series.forEach((file, i) => {
      const K = t.K[i], M = t.M[i];
      if (K < 2) return;
      let single = -Infinity, who = -1, weak = 0;          // the best detector alone, and its lowest anomaly
      const best = new Array<number>(K).fill(-Infinity), winner = new Array<number>(K).fill(-1);
      for (let d = 0; d < t.detectors.length; d++) {
        if (M[d * K] < -900) continue;                      // no score for this detector
        let low = Infinity, at = 0;
        for (let k = 0; k < K; k++) { const v = M[d * K + k]; if (v < low) { low = v; at = k; } if (v > best[k]) { best[k] = v; winner[k] = d; } }
        if (low > single) { single = low; who = d; weak = at; }
      }
      const all = best.reduce((p, q) => Math.min(p, q), Infinity);
      alone.push(single); together.push(all);
      // To open the series with: its best detector alone, and the detector that is best on that one's lowest anomaly.
      cases.push({ file, K, alone: Math.max(0, single) / 100, together: Math.max(0, all) / 100, dets: Array.from(new Set([who, winner[weak]])).map((d) => t.detectors[d]) });
    });
    // Series at the same position are spread on a small spiral around it, folded back into the plot.
    const seen = new Map<string, number>();
    const fold = (v: number) => (v > 1 ? 2 - v : v < 0 ? -v : v);
    const points = cases.map((c) => {
      const key = `${c.alone}|${c.together}`, j = seen.get(key) ?? 0;
      seen.set(key, j + 1);
      const r = 0.008 * Math.sqrt(j), a = j * 2.39996;
      return { x: fold(c.alone + r * Math.cos(a)), y: fold(c.together + r * Math.sin(a)), g: c.together - c.alone, title: seriesName(c.file), sub: `${c.K} anomalies` };
    });
    const rows = Array.from({ length: 100 }, (_, j) => {
      const T = j + 1, one = alone.filter((v) => v >= T).length, cov = together.filter((v) => v >= T).length;
      return { T: T / 100, one, several: cov - one, none: alone.length - cov };
    });
    return { n: alone.length, rows, cases, points };
  }, [t]);

  const sx = (T: number) => PAD.l + T * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - (v / (view?.n ?? 1)) * (H - PAD.t - PAD.b);
  const band = (lo: (r: { one: number; several: number; none: number }) => number, hi: (r: { one: number; several: number; none: number }) => number) => {
    const rs = view!.rows;
    return `M${rs.map((r) => `${sx(r.T).toFixed(1)},${sy(hi(r)).toFixed(1)}`).join("L")}L${[...rs].reverse().map((r) => `${sx(r.T).toFixed(1)},${sy(lo(r)).toFixed(1)}`).join("L")}Z`;
  };
  const at = hover === null || !view ? null : view.rows[hover];
  const yTicks = view ? [0, 100, 200, 300, 400, view.n].filter((v) => v <= view.n) : [];

  return (
    <main className="content">
      <section className="panel" aria-label="Series covered at each threshold">
        <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 18px", fontSize: 12 }}>
          {BANDS.map((x) => <span key={x.key} style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ background: x.color }} />{x.label}</span>)}
        </div>
        {error && <span className="muted">No local tables. Run <code>conda run -n autovus python app/export_local.py</code>.</span>}
        <div ref={box} style={{ position: "relative", height: H }}>
          {view && W > 0 && (
            <svg width={W} height={H} role="img" aria-label="Time series by how their anomalies are covered, at each threshold" style={{ display: "block" }}
              onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const T = (e.clientX - r.left - PAD.l) / (W - PAD.l - PAD.r); setHover(Math.min(99, Math.max(0, Math.round(T * 100) - 1))); }}
              onMouseLeave={() => setHover(null)}>
              {yTicks.map((v) => (<g key={v}><line x1={PAD.l} x2={W - PAD.r} y1={sy(v)} y2={sy(v)} stroke="var(--border-soft)" /><text x={PAD.l - 8} y={sy(v)} textAnchor="end" dominantBaseline="middle" fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">{v}</text></g>))}
              <path d={band(() => 0, (r) => r.one)} fill="var(--exact)" />
              <path d={band((r) => r.one, (r) => r.one + r.several)} fill="var(--accent)" />
              <path d={band((r) => r.one + r.several, (r) => r.one + r.several + r.none)} fill="var(--missed)" />
              {/* surface gaps between the bands */}
              <path d={`M${view.rows.map((r) => `${sx(r.T).toFixed(1)},${sy(r.one).toFixed(1)}`).join("L")}`} fill="none" stroke="var(--panel)" strokeWidth={2} />
              <path d={`M${view.rows.map((r) => `${sx(r.T).toFixed(1)},${sy(r.one + r.several).toFixed(1)}`).join("L")}`} fill="none" stroke="var(--panel)" strokeWidth={2} />
              {[0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => <text key={v} x={sx(v)} y={H - PAD.b + 14} textAnchor="middle" fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">{v.toFixed(1)}</text>)}
              <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle" fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">AutoVUS Local needed to count an anomaly as detected</text>
              <text transform={`translate(11, ${(PAD.t + H - PAD.b) / 2}) rotate(-90)`} textAnchor="middle" fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">Time series</text>
              {at && <line x1={sx(at.T)} x2={sx(at.T)} y1={PAD.t} y2={H - PAD.b} stroke="var(--ink)" strokeWidth={1} />}
            </svg>
          )}
          {at && (
            <div style={{ position: "absolute", left: Math.min(Math.max(sx(at.T), 150), W - 150), top: PAD.t + 4, transform: "translateX(-50%)", pointerEvents: "none",
              padding: "7px 10px", background: "var(--tooltip-bg)", color: "var(--tooltip-ink)", borderRadius: 6, fontSize: 12, display: "flex", flexDirection: "column", gap: 3, whiteSpace: "nowrap" }}>
              <span className="mono" style={{ fontWeight: 600 }}>threshold {at.T.toFixed(2)}</span>
              {BANDS.map((x) => <span key={x.key} style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ background: x.color, width: 10, height: 10 }} /><span className="mono" style={{ width: 28, textAlign: "right" }}>{at[x.key]}</span>{x.label}</span>)}
            </div>
          )}
        </div>
      </section>
      {view && (
        <section className="panel" aria-label="Every series">
          <GapScatter points={view.points} xlabel="One detector is enough up to this threshold" ylabel="Several together are enough up to this threshold"
            glabel="Thresholds where several detectors are needed" far={0.5} above="--accent" oneSided mark={[]}
            tip={(i) => { const c = view.cases[i]; return `one detector up to ${c.alone.toFixed(2)} (${c.dets[0]}) · several up to ${c.together.toFixed(2)}`; }}
            onPick={(i) => router.push(href("/local", state, { s: view.cases[i].file, det: view.cases[i].dets.join(",") }))} />
        </section>
      )}
    </main>
  );
}
