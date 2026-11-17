"use client";
// The game: for each round, drag the buffer edges before and after the label
// to what the detection deserves, lock in, and see the buffers AutoVUS picks
// (per side, L = 20). Every game is a new draw of ROUNDS rounds from
// app/data/<benchmark>/local/game.json. A round opens in Review.

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { loadGame } from "@/lib/api";
import { href, type DetectionType, type State } from "@/lib/model";
import { seriesName, type GameDoc, type GameRound } from "@/lib/local";

const ROUNDS = 6;

const NAMES: Record<DetectionType, string> = { early: "Before the label", exact: "On the label", late: "After the label", wide: "Both sides", missed: "Missed" };
const BEFORE = "var(--early)", AFTER = "var(--down)";
type Phase = "drag" | "reveal" | "end";
const typeOf = (bl: number, br: number): DetectionType => (bl > 0 && br > 0 ? "wide" : bl > 0 ? "early" : br > 0 ? "late" : "exact");

export function GameView({ state }: { state: State }) {
  const [g, setG] = useState<GameDoc | null>(null);
  const [error, setError] = useState(false);
  const [i, setI] = useState(0);
  const [phase, setPhase] = useState<Phase>("drag");
  const [bl, setBl] = useState(0);
  const [br, setBr] = useState(0);
  const [picks, setPicks] = useState<[number, number][]>([]);
  const [t0, setT0] = useState(0);
  const [t1, setT1] = useState(0);
  const plot = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<"l" | "r" | null>(null);

  // A new draw: on arrival and on Play again.
  const draw = useCallback(() => {
    setG(null); setI(0); setPhase("drag"); setBl(0); setBr(0); setPicks([]); setT1(0);
    loadGame<GameDoc>(state.benchmark, ROUNDS).then((d) => { if (d && d.rounds.length) { setG(d); setT0(Date.now()); } else setError(true); });
  }, [state.benchmark]);
  useEffect(() => { draw(); }, [draw]);
  // The round in Review: the detector pinned, zoomed on the window, the anomaly's buffer explained.
  const review = (x: GameRound) => href("/local", state, { s: x.file, det: x.detector, v: `${x.lo},${x.hi}`, a: String(x.k) });

  if (error) return <main className="content"><div className="panel"><span className="muted">No game rounds. Run <code>conda run -n autovus python app/export_local.py</code> and start the Python side.</span></div></main>;
  if (!g) return <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}><div className="panel" style={{ height: 420 }} /></main>;

  const n = g.rounds.length;
  const L = g.L;
  const header = (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
      <div style={{ display: "flex", gap: 8 }} aria-label={`Round ${Math.min(i + 1, n)} of ${n}`}>
        {g.rounds.map((_, k) => <span key={k} style={{ width: 28, height: 6, borderRadius: 3, background: k < i || phase === "end" ? "var(--ink)" : k === i ? "var(--accent)" : "var(--border)" }} />)}
      </div>
    </div>
  );

  if (phase === "end") {
    const ms = g.rounds.reduce((a, r) => a + r.seconds * 1000, 0);
    return (
      <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
        {header}
        <section className="panel" aria-label="Result" style={{ alignItems: "center", padding: "32px 24px", gap: 22 }}>
          <table className="table" style={{ maxWidth: 720 }}>
            <thead>
              <tr><th>Detector</th><th>You</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th><th style={{ color: "var(--accent)" }}>AutoVUS</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th><th /></tr>
            </thead>
            <tbody>
              {g.rounds.map((x, k) => {
                const [pl, pr] = picks[k] ?? [0, 0];
                const yt = typeOf(pl, pr);
                return (
                  <tr key={k} style={{ cursor: "default" }}>
                    <td className="sans" style={{ fontWeight: 600 }}>{x.detector}</td>
                    <td className="sans" style={{ color: yt === x.type ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yt]}</td>
                    <td className="mono">{pl}</td><td className="mono">{pr}</td>
                    <td className="sans" style={{ fontWeight: 600 }}>{NAMES[x.type]}</td>
                    <td className="mono">{x.b_left}</td><td className="mono">{x.b_right}</td>
                    <td className="r"><Link href={review(x)} style={{ fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Review</Link></td>
                  </tr>
                );
              })}
              <tr style={{ cursor: "default" }}>
                <td className="sans muted">Time</td><td className="mono" colSpan={3}>{Math.round((t1 - t0) / 1000)} s</td>
                <td className="mono" colSpan={4} style={{ color: "var(--accent)" }}>{ms.toFixed(1)} ms</td>
              </tr>
            </tbody>
          </table>
          <button type="button" className="btn btn-primary" onClick={draw} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Play again</button>
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
  const yourType = typeOf(bl, br);
  const sameType = yourType === r.type || (r.type === "missed" && yourType === "exact");
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
            <button type="button" className="btn btn-primary" onClick={() => { setPicks((p) => [...p, [bl, br]]); setPhase("reveal"); }} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Lock in</button>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <table className="table">
              <thead><tr><th /><th>Detection</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th></tr></thead>
              <tbody>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600 }}>You</td><td className="sans" style={{ color: sameType ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yourType]}</td><td className="mono">{bl}</td><td className="mono">{br}</td></tr>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600, color: "var(--accent)" }}>AutoVUS</td><td className="sans" style={{ fontWeight: 600 }}>{NAMES[r.type]}</td><td className="mono">{r.b_left}</td><td className="mono">{r.b_right}</td></tr>
              </tbody>
            </table>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
              <span className="mono muted" style={{ fontSize: 13 }}>AutoVUS {(r.seconds * 1000).toFixed(1)} ms</span>
              <Link href={review(r)} className="btn" style={{ marginLeft: "auto", height: 44, padding: "0 18px", fontSize: 14, display: "inline-flex", alignItems: "center", textDecoration: "none" }}>Why this buffer</Link>
              <button type="button" className="btn" onClick={() => { if (last) { setPhase("end"); setT1(Date.now()); } else { setI(i + 1); setPhase("drag"); setBl(0); setBr(0); } }}
                style={{ height: 44, padding: "0 22px", fontSize: 14, fontWeight: 600, background: "var(--ink)", color: "var(--panel)", borderColor: "var(--ink)" }}>{last ? "See result" : "Next round"}</button>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
