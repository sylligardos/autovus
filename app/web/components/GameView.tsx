"use client";
// The game: for each round, drag the buffer edges before and after the label
// to what the detection deserves, lock in, and see the buffers AutoVUS picks
// (per side, L = 20). Every game is a new draw of ROUNDS rounds from
// app/data/<benchmark>/local/game.json. After the lock-in, the ratio curves
// that explain AutoVUS's buffers show under the round; a round opens in Review.

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { loadCurves, loadGame, loadSelect } from "@/lib/api";
import { href, type Anomaly, type CurvesDoc, type DetectionType, type State } from "@/lib/model";
import { seriesName, type GameDoc, type GameRound } from "@/lib/local";
import { Drawer } from "./Drawer";

const ROUNDS = 6;
const TAB = 26;                                    // room under the plot for the tabs of the two edges

const NAMES: Record<DetectionType, string> = { early: "Before the label", exact: "On the label", late: "After the label", wide: "Both sides", missed: "Missed" };
const BEFORE = "var(--early)", AFTER = "var(--down)";
type Phase = "intro" | "drag" | "reveal" | "end";
const typeOf = (bl: number, br: number): DetectionType => (bl > 0 && br > 0 ? "wide" : bl > 0 ? "early" : br > 0 ? "late" : "exact");

// candidates: every round of game_candidates.json in order, numbered, no intro (/play?set=candidates), to choose what joins the pool.
export function GameView({ state, candidates = false }: { state: State; candidates?: boolean }) {
  const [g, setG] = useState<GameDoc | null>(null);
  const [error, setError] = useState(false);
  const [i, setI] = useState(0);
  const [phase, setPhase] = useState<Phase>(candidates ? "drag" : "intro");
  const [bl, setBl] = useState(0);
  const [br, setBr] = useState(0);
  const [picks, setPicks] = useState<[number, number][]>([]);
  const plot = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<"l" | "r" | null>(null);
  const [why, setWhy] = useState<{ a: Anomaly; curves: CurvesDoc } | null>(null);

  // A new draw: on arrival (behind the intro) and on Play again (straight into round 1).
  const draw = useCallback((again: boolean) => {
    setG(null); setI(0); setBl(0); setBr(0); setPicks([]);
    if (again) setPhase("drag");
    setError(false);
    // The Python side can be restarting: ask a few times before giving up.
    const ask = (left: number) => loadGame<GameDoc>(state.benchmark, candidates ? undefined : ROUNDS, candidates ? "candidates" : undefined).then((d) => {
      if (d && d.rounds.length) setG(d);
      else if (left > 0) setTimeout(() => ask(left - 1), 1500);
      else setError(true);
    });
    ask(6);
  }, [state.benchmark, candidates]);
  useEffect(() => { draw(false); }, [draw]);
  // Why this buffer: the curves of the round's anomaly, once the player has locked in.
  const cur = g?.rounds[i] ?? null;
  useEffect(() => {
    setWhy(null);
    if (phase !== "reveal" || !cur) return;
    let live = true;
    Promise.all([loadSelect(state.benchmark, cur.file, cur.detector, cur.L), loadCurves(state.benchmark, cur.file, cur.detector, cur.k, cur.L)]).then(([sel, c]) => {
      if (live && sel && c && sel.anomalies[cur.k]) setWhy({ a: sel.anomalies[cur.k], curves: c });
    });
    return () => { live = false; };
  }, [phase, cur, state.benchmark]);
  // The round in Review: the detector pinned, zoomed on the window, the anomaly's buffer explained.
  const review = (x: GameRound) => href("/local", state, { s: x.file, det: x.detector, v: `${x.lo},${x.hi}`, a: String(x.k) });

  if (error) return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      <div className="panel" style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
        <span className="muted">The Python side did not answer.</span>
        <button type="button" className="btn btn-primary" onClick={() => draw(phase !== "intro")}>Try again</button>
      </div>
    </main>
  );
  if (phase === "intro") {
    return (
      <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
        <section className="panel" aria-label="The game" style={{ padding: "36px 40px 40px", gap: 24 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <h1 style={{ fontSize: 30, fontWeight: 700, letterSpacing: "-0.015em", lineHeight: 1.15 }}>How much buffer does a detection need?</h1>
            <p className="muted" style={{ margin: 0, fontSize: 15, maxWidth: "46em" }}>A detector doesn&apos;t always fire exactly on the anomaly. It may fire a little early or a little late. A range-aware metric forgives that with a buffer around the label. Here you set that buffer yourself, one detection at a time, and compare it with the one AutoVUS picks.</p>
          </div>
          <svg viewBox="0 0 640 172" width="100%" style={{ display: "block" }} role="img" aria-label="A labelled anomaly, a detector score that fires late, and a buffer after the label">
            <rect x={0} y={0} width={640} height={150} rx={6} fill="var(--subtle)" />
            <rect x={270} y={0} width={70} height={150} fill="var(--label)" opacity={0.55} />
            <rect x={340} y={0} width={52} height={150} fill={AFTER} opacity={0.22} />
            <polyline points="0,50 250,50 262,46 270,32 300,28 340,34 352,48 640,50" fill="none" stroke="var(--ink)" strokeWidth={1.5} />
            <polyline points="0,128 300,128 306,84 386,80 392,128 640,128" fill="none" stroke="var(--accent)" strokeWidth={1.75} />
            <rect x={269} y={0} width={2} height={152} fill={BEFORE} opacity={0.7} /><rect x={256} y={152} width={14} height={18} rx={4} fill={BEFORE} fillOpacity={0.18} stroke={BEFORE} strokeWidth={1.5} />
            <rect x={391} y={0} width={2} height={152} fill={AFTER} opacity={0.7} /><rect x={392} y={152} width={14} height={18} rx={4} fill={AFTER} fillOpacity={0.18} stroke={AFTER} strokeWidth={1.5} />
            <g fontFamily="var(--mono)" fontSize={11} fill="var(--muted)">
              <text x={8} y={16}>series</text><text x={8} y={112}>detector score</text>
              <text x={305} y={16} textAnchor="middle">label</text><text x={400} y={16}>buffer after</text><text x={262} y={16} textAnchor="end">no buffer before</text>
            </g>
          </svg>
          <p style={{ margin: 0, fontSize: 14 }}>Drag the blue edge for the buffer before the label and the red edge for the buffer after it, then lock in.</p>
          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <button type="button" className="btn btn-primary" disabled={!g} onClick={() => setPhase("drag")} style={{ height: 48, padding: "0 34px", fontSize: 16 }}>{g ? "Start" : "Loading…"}</button>
            <span className="mono muted" style={{ fontSize: 12 }}>{ROUNDS} rounds</span>
          </div>
        </section>
      </main>
    );
  }
  if (!g) return <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}><div className="panel" style={{ height: 420 }} /></main>;

  const n = g.rounds.length;
  const L = g.L;
  const header = (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
      <div style={{ display: "flex", gap: 8 }} aria-label={`Round ${Math.min(i + 1, n)} of ${n}`}>
        {g.rounds.map((_, k) => <span key={k} style={{ width: 28, height: 6, borderRadius: 3, background: k < i || phase === "end" ? "var(--ink)" : k === i ? "var(--accent)" : "var(--border)" }} />)}
      </div>
      {phase !== "end" && <span className="mono muted" style={{ fontSize: 12 }}>{candidates ? "Candidate" : "Round"} {i + 1} of {n}</span>}
    </div>
  );

  if (phase === "end") {
    return (
      <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
        {header}
        <section className="panel" aria-label="Result" style={{ alignItems: "center", padding: "32px 24px", gap: 22 }}>
          <h2 style={{ fontSize: 17, textTransform: "none", letterSpacing: 0 }}>You vs AutoVUS</h2>
          <table className="table">
            <thead>
              <tr>{candidates && <th>#</th>}<th>Series</th><th>Detector</th><th>You</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th><th style={{ color: "var(--accent)" }}>AutoVUS</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th><th /></tr>
            </thead>
            <tbody>
              {g.rounds.map((x, k) => {
                const [pl, pr] = picks[k] ?? [0, 0];
                const yt = typeOf(pl, pr);
                return (
                  <tr key={k} style={{ cursor: "default" }}>
                    {candidates && <td className="mono">{k + 1}</td>}
                    <td className="mono" style={{ fontSize: 11 }}>{seriesName(x.file)}</td>
                    <td className="sans" style={{ fontWeight: 600 }}>{x.detector}</td>
                    <td className="sans" style={{ color: yt === x.type ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yt]}</td>
                    <td className="mono">{pl}</td><td className="mono">{pr}</td>
                    <td className="sans" style={{ fontWeight: 600 }}>{NAMES[x.type]}</td>
                    <td className="mono">{x.b_left}</td><td className="mono">{x.b_right}</td>
                    <td className="r"><Link href={review(x)} style={{ fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Explore</Link></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <button type="button" className="btn btn-primary" onClick={() => draw(true)} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Play again</button>
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
    const left = side === "l", color = left ? BEFORE : AFTER;
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
        style={{ position: "absolute", top: 0, bottom: -TAB, left: left ? at(r.start - bl) : at(r.end + 1 + br), width: 18, transform: "translateX(-50%)", padding: 0, border: 0, background: "transparent", cursor: "ew-resize", touchAction: "none" }}>
        <span style={{ position: "absolute", top: 0, bottom: TAB - 4, left: left ? 7 : 9, width: 2, background: color, opacity: 0.7 }} />
        <span style={{ position: "absolute", bottom: 0, [left ? "right" : "left"]: 9, width: 14, height: TAB - 4, boxSizing: "border-box", borderRadius: 4, border: `1.5px solid ${color}`, background: `color-mix(in srgb, ${color} 18%, transparent)` }} />
      </button>
    );
  };

  return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      {header}
      <section className="panel" aria-label="Round" style={{ padding: "18px 24px 20px", gap: 14 }}>
        <h2 style={{ fontSize: 17, textTransform: "none", letterSpacing: 0 }}>{phase === "drag" ? "How much buffer does this detection need? Drag the edges." : "AutoVUS picked the dashed buffer"}</h2>
        <div ref={plot} style={{ position: "relative", height: 230, marginBottom: TAB, touchAction: "none", userSelect: "none" }}>
          {zone(r.start - bl, r.start, BEFORE, false)}
          {zone(r.end + 1, r.end + 1 + br, AFTER, false)}
          <div style={{ position: "absolute", top: 0, bottom: 0, left: at(r.start), width: `${((r.end - r.start + 1) / N) * 100}%`, minWidth: 3, background: "var(--label)", opacity: 0.55 }} />
          {phase === "reveal" && zone(r.start - r.b_left, r.start, BEFORE, true)}
          {phase === "reveal" && zone(r.end + 1, r.end + 1 + r.b_right, AFTER, true)}
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 14, width: "100%", height: 88, pointerEvents: "none" }}><polyline points={line(r.values)} fill="none" stroke="var(--ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" /></svg>
          <div style={{ position: "absolute", left: 0, right: 0, top: 108, borderTop: "1px solid var(--border-soft)" }} />
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 128, width: "100%", height: 102, pointerEvents: "none" }}><polyline points={line(r.score)} fill="none" stroke="var(--accent)" strokeWidth={1.75} vectorEffect="non-scaling-stroke" /></svg>
          <span className="mono muted" style={{ position: "absolute", left: 0, top: 0, fontSize: 10, lineHeight: "12px", pointerEvents: "none" }}>{seriesName(r.file)}</span>
          <span className="mono muted" style={{ position: "absolute", left: 0, top: 113, fontSize: 10, lineHeight: "12px", pointerEvents: "none" }}>{r.detector} score</span>
          {phase === "drag" && [handle("l"), handle("r")]}
        </div>
        <div className="muted" style={{ display: "flex", flexWrap: "wrap", gap: 16, fontSize: 12 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, background: "var(--label)", opacity: 0.7 }} />label</span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, background: BEFORE, opacity: 0.4 }} />your buffer before</span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, background: AFTER, opacity: 0.4 }} />your buffer after</span>
          {phase === "reveal" && <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, boxSizing: "border-box", border: "2px dashed var(--ink)" }} />AutoVUS</span>}
        </div>

        {phase === "drag" ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <span className="mono" style={{ fontSize: 14 }}><span style={{ color: BEFORE }}>{bl}</span> · <span style={{ color: AFTER }}>{br}</span></span>
            <button type="button" className="btn btn-primary" onClick={() => { setPicks((p) => [...p, [bl, br]]); setPhase("reveal"); }} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Lock in</button>
          </div>
        ) : (
          <div style={{ display: "flex", alignItems: "flex-end", gap: 24 }}>
            <table className="table" style={{ flex: 1 }}>
              <thead><tr><th /><th>Detection</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th></tr></thead>
              <tbody>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600 }}>You</td><td className="sans" style={{ color: sameType ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yourType]}</td><td className="mono">{bl}</td><td className="mono">{br}</td></tr>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600, color: "var(--accent)" }}>AutoVUS</td><td className="sans" style={{ fontWeight: 600 }}>{NAMES[r.type]}</td><td className="mono">{r.b_left}</td><td className="mono">{r.b_right}</td></tr>
              </tbody>
            </table>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
              <span className="mono muted" style={{ fontSize: 12 }}>AutoVUS {(r.seconds * 1000).toFixed(1)} ms</span>
              <button type="button" className="btn" onClick={() => { if (last) setPhase("end"); else { setI(i + 1); setPhase("drag"); setBl(0); setBr(0); } }}
                style={{ height: 44, padding: "0 22px", fontSize: 14, fontWeight: 600, background: "var(--ink)", color: "var(--panel)", borderColor: "var(--ink)" }}>{last ? "See result" : "Next round"}</button>
            </div>
          </div>
        )}
      </section>
      {phase === "reveal" && (
        <section className="panel panel-accent" aria-label="Why this buffer" style={{ padding: "12px 24px 16px", gap: 8 }}>
          <div className="panel-head">
            <h2>Why this buffer</h2>
            <span className="mono" style={{ fontSize: 12 }}>{NAMES[r.type]} · b* = ({r.b_left}, {r.b_right})</span>
            <Link href={review(r)} className="btn" style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", textDecoration: "none" }}>Open in Explore</Link>
          </div>
          {why ? <Drawer curves={why.curves} a={why.a} L={L} height={110} /> : <div style={{ height: 134 }} />}
        </section>
      )}
    </main>
  );
}
