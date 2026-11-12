"use client";
// The game: for each round, drag the buffer edges before and after the label
// to what the detection deserves, lock in, and see the buffers AutoVUS picks
// (per side, L = 20). The rounds live in app/data/<benchmark>/local/game.json.

import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { loadGame } from "@/lib/api";
import { type DetectionType, type State } from "@/lib/model";
import { seriesName, type GameDoc } from "@/lib/local";

const NAMES: Record<DetectionType, string> = { early: "Before the label", exact: "On the label", late: "After the label", wide: "Both sides", missed: "Missed" };
const BEFORE = "var(--early)", AFTER = "var(--down)";
type Phase = "drag" | "reveal" | "end";

export function GameView({ state }: { state: State }) {
  const [g, setG] = useState<GameDoc | null>(null);
  const [error, setError] = useState(false);
  const [i, setI] = useState(0);
  const [phase, setPhase] = useState<Phase>("drag");
  const [bl, setBl] = useState(0);
  const [br, setBr] = useState(0);
  const [pts, setPts] = useState<number[]>([]);
  const [t0, setT0] = useState(0);
  const [t1, setT1] = useState(0);
  const plot = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<"l" | "r" | null>(null);

  useEffect(() => {
    let live = true;
    loadGame<GameDoc>(state.benchmark).then((d) => { if (live) { if (d && d.rounds.length) { setG(d); setT0(Date.now()); } else setError(true); } });
    return () => { live = false; };
  }, [state.benchmark]);

  if (error) return <main className="content"><div className="panel"><span className="muted">No game rounds. Run <code>conda run -n autovus python app/export_local.py</code> and start the Python side.</span></div></main>;
  if (!g) return <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}><div className="panel" style={{ height: 420 }} /></main>;

  const n = g.rounds.length;
  const total = pts.reduce((a, b) => a + b, 0);
  const L = g.L;
  const restart = () => { setI(0); setPhase("drag"); setBl(0); setBr(0); setPts([]); setT0(Date.now()); setT1(0); };
  const header = (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
      <div style={{ display: "flex", gap: 8 }} aria-label={`Round ${Math.min(i + 1, n)} of ${n}`}>
        {g.rounds.map((_, k) => <span key={k} style={{ width: 28, height: 6, borderRadius: 3, background: k < i || phase === "end" ? "var(--ink)" : k === i ? "var(--accent)" : "var(--border)" }} />)}
      </div>
      <div className="mono" style={{ display: "flex", alignItems: "baseline", gap: 18 }}>
        <span className="muted" style={{ fontSize: 13 }}>You <span style={{ fontSize: 20, color: "var(--ink)" }}>{total.toFixed(1)}</span></span>
        <span className="muted" style={{ fontSize: 13 }}>AutoVUS <span style={{ fontSize: 20, color: "var(--accent)" }}>{(phase === "end" ? n : pts.length).toFixed(1)}</span></span>
      </div>
    </div>
  );

  if (phase === "end") {
    const ms = g.rounds.reduce((a, r) => a + r.seconds * 1000, 0);
    return (
      <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
        {header}
        <section className="panel" aria-label="Result" style={{ alignItems: "center", padding: "40px 24px", gap: 22, textAlign: "center" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 220px))", gap: 24 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="muted" style={{ fontSize: 13, fontWeight: 600 }}>You</span>
              <span className="mono" style={{ fontSize: 44 }}>{total.toFixed(1)}</span>
              <span className="mono muted" style={{ fontSize: 13 }}>{Math.round((t1 - t0) / 1000)} s</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: "var(--accent)" }}>AutoVUS</span>
              <span className="mono" style={{ fontSize: 44, color: "var(--accent)" }}>{n.toFixed(1)}</span>
              <span className="mono muted" style={{ fontSize: 13 }}>{ms.toFixed(1)} ms</span>
            </div>
          </div>
          <button type="button" className="btn btn-primary" onClick={restart} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Play again</button>
        </section>
      </main>
    );
  }

  const r = g.rounds[i];
  const N = r.hi - r.lo;
  const at = (p: number) => `${(Math.max(0, Math.min(N, p - r.lo)) / N) * 100}%`;
  const zone = (a: number, b: number, color: string, dashed: boolean) => {
    const x0 = Math.max(r.lo, a), x1 = Math.min(r.hi, b);
    if (x1 <= x0) return null;
    const style = { position: "absolute" as const, top: dashed ? -6 : 0, bottom: dashed ? -6 : 0, left: at(x0), width: `${((x1 - x0) / N) * 100}%` };
    return dashed ? <div style={{ ...style, boxSizing: "border-box", border: `2px dashed ${color}`, borderRadius: 3 }} /> : <div style={{ ...style, background: color, opacity: 0.22 }} />;
  };
  const line = (a: number[]) => a.map((v, k) => `${(((k + 0.5) / a.length) * 1000).toFixed(1)},${(96 - v * 92).toFixed(1)}`).join(" ");
  const yourType: DetectionType = bl > 0 && br > 0 ? "wide" : bl > 0 ? "early" : br > 0 ? "late" : "exact";
  const sameType = yourType === r.type || (r.type === "missed" && yourType === "exact");
  const roundPts = Math.round(Math.max(0, 1 - (Math.abs(bl - r.b_left) + Math.abs(br - r.b_right)) / 20) * 10) / 10;
  const last = i === n - 1;

  const setFrom = (side: "l" | "r", clientX: number) => {
    const box = plot.current?.getBoundingClientRect();
    if (!box) return;
    const p = r.lo + ((clientX - box.left) / box.width) * N;
    const clamp = (v: number) => Math.max(0, Math.min(L, Math.round(v)));
    if (side === "l") setBl(clamp(r.start - p)); else setBr(clamp(p - (r.end + 1)));
  };
  const handle = (side: "l" | "r") => {
    const left = side === "l";
    return (
      <button key={side} type="button" aria-label={left ? `Buffer before the label: ${bl} points` : `Buffer after the label: ${br} points`}
        onPointerDown={(e: PointerEvent<HTMLButtonElement>) => { setDrag(side); e.currentTarget.setPointerCapture(e.pointerId); }}
        onPointerMove={(e: PointerEvent<HTMLButtonElement>) => { if (drag === side) setFrom(side, e.clientX); }}
        onPointerUp={() => setDrag(null)}
        onKeyDown={(e: KeyboardEvent<HTMLButtonElement>) => {
          const out = left ? "ArrowLeft" : "ArrowRight", inn = left ? "ArrowRight" : "ArrowLeft";
          const set = left ? setBl : setBr;
          if (e.key === out) { e.preventDefault(); set((v) => Math.min(L, v + 1)); }
          if (e.key === inn) { e.preventDefault(); set((v) => Math.max(0, v - 1)); }
        }}
        style={{ position: "absolute", top: -10, bottom: -10, left: left ? at(r.start - bl) : at(r.end + 1 + br), width: 18, transform: "translateX(-50%)", padding: 0, border: 0, background: "transparent", cursor: "ew-resize", touchAction: "none", display: "flex", justifyContent: "center" }}>
        <span style={{ width: 4, height: "100%", borderRadius: 2, background: left ? BEFORE : AFTER }} />
        <span style={{ position: "absolute", top: "50%", width: 16, height: 36, transform: "translateY(-50%)", borderRadius: 5, background: left ? BEFORE : AFTER, boxShadow: "0 1px 3px rgba(21,32,51,0.35)" }} />
      </button>
    );
  };

  return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      {header}
      <section className="panel" aria-label="Round" style={{ padding: "20px 24px 24px", gap: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", fontSize: 12 }}>
          <span className="mono muted">{seriesName(r.file)}</span>
          <span style={{ fontWeight: 600 }}>{r.detector}</span>
        </div>
        <h2 style={{ fontSize: 17, textTransform: "none", letterSpacing: 0 }}>{phase === "drag" ? "Drag the edges: how much buffer does this detection deserve?" : "AutoVUS picked the dashed buffer"}</h2>
        <div ref={plot} style={{ position: "relative", height: 280, touchAction: "none", userSelect: "none" }}>
          {zone(r.start - bl, r.start, BEFORE, false)}
          {zone(r.end + 1, r.end + 1 + br, AFTER, false)}
          <div style={{ position: "absolute", top: 0, bottom: 0, left: at(r.start), width: `${((r.end - r.start + 1) / N) * 100}%`, minWidth: 3, background: "var(--label)", opacity: 0.55 }} />
          {phase === "reveal" && zone(r.start - r.b_left, r.start, BEFORE, true)}
          {phase === "reveal" && zone(r.end + 1, r.end + 1 + r.b_right, AFTER, true)}
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 0, width: "100%", height: 118, pointerEvents: "none" }}><polyline points={line(r.values)} fill="none" stroke="var(--ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" /></svg>
          <div style={{ position: "absolute", left: 0, right: 0, top: 136, borderTop: "1px solid var(--border-soft)" }} />
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 150, width: "100%", height: 130, pointerEvents: "none" }}><polyline points={line(r.score)} fill="none" stroke="var(--accent)" strokeWidth={1.75} vectorEffect="non-scaling-stroke" /></svg>
          {phase === "drag" && [handle("l"), handle("r")]}
        </div>

        {phase === "drag" ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <span className="mono" style={{ fontSize: 14 }}><span style={{ color: BEFORE }}>{bl}</span> · <span style={{ color: AFTER }}>{br}</span></span>
            <button type="button" className="btn btn-primary" onClick={() => { setPts((p) => [...p, roundPts]); setPhase("reveal"); }} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Lock in</button>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <table className="table">
              <thead><tr><th /><th>Detection</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th><th className="r">Points</th></tr></thead>
              <tbody>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600 }}>You</td><td className="sans" style={{ color: sameType ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yourType]}</td><td className="mono">{bl}</td><td className="mono">{br}</td><td className="r mono">{(pts[i] ?? 0).toFixed(1)}</td></tr>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600, color: "var(--accent)" }}>AutoVUS</td><td className="sans" style={{ fontWeight: 600 }}>{NAMES[r.type]}</td><td className="mono">{r.b_left}</td><td className="mono">{r.b_right}</td><td className="r mono" style={{ color: "var(--accent)" }}>1.0</td></tr>
              </tbody>
            </table>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
              <span className="mono muted" style={{ fontSize: 13 }}>AutoVUS {(r.seconds * 1000).toFixed(1)} ms</span>
              <button type="button" className="btn" onClick={() => { if (last) { setPhase("end"); setT1(Date.now()); } else { setI(i + 1); setPhase("drag"); setBl(0); setBr(0); } }}
                style={{ height: 44, padding: "0 22px", fontSize: 14, fontWeight: 600, background: "var(--ink)", color: "var(--panel)", borderColor: "var(--ink)" }}>{last ? "See result" : "Next round"}</button>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
