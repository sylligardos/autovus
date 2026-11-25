"use client";
// Find the buffer: for each round, drag the buffer edges before and after the
// label to what the detection deserves, lock in, and see the buffers AutoVUS
// picks (per side, at the gear L). Every game is a new draw of ROUNDS rounds
// from app/data/<benchmark>/local/game.json. After the lock-in, the ratio
// curves that explain AutoVUS's buffers show under the round with the
// player's pick on them, and the AutoVUS value of the series with each pair of
// buffers. The game starts in round 1; the landing page is PlayIntro. A game is
// recorded for Played so far (POST /games) under the name from the landing when
// its last round is done, or when the player leaves after at least one lock-in.

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { loadCurves, loadGame, loadSelect, saveGame, scoreBuffers, type BuffersDoc } from "@/lib/api";
import { f3, href, type Anomaly, type CurvesDoc, type DetectionType, type State } from "@/lib/model";
import { seriesName, type GameDoc, type GameRound } from "@/lib/local";
import { ANON, getPlayer } from "@/lib/player";
import { Drawer } from "./Drawer";

const ROUNDS = 6;
const TAB = 26;                                    // room under the plot for the tabs of the two edges
const GAIN = 28;                                   // room above the plot for the live gain of the two edges

export const NAMES: Record<DetectionType, string> = { early: "Before the label", exact: "On the label", late: "After the label", wide: "Both sides", missed: "Missed" };
const BEFORE = "var(--early)", AFTER = "var(--down)";
type Phase = "drag" | "reveal" | "end";
const typeOf = (bl: number, br: number): DetectionType => (bl > 0 && br > 0 ? "wide" : bl > 0 ? "early" : br > 0 ? "late" : "exact");
// The same verdict as AutoVUS: a missed detection gets no buffer, so "on the label" counts.
const agrees = (yours: DetectionType, auto: DetectionType) => yours === auto || (auto === "missed" && yours === "exact");
// The gain of one side at buffer b, and how close it is to the best of that side (0 = the lowest, 1 = the peak).
const gainAt = (curve: number[] | undefined, b: number) => {
  if (!curve?.length) return null;                 // an API that does not send the curves yet: no number
  const v = curve[Math.min(b, curve.length - 1)], lo = Math.min(...curve), hi = Math.max(...curve);
  return { v, t: hi > lo ? (v - lo) / (hi - lo) : 1 };
};
// How close a pick is to the peak gain, the mean of its two sides (a side without a curve counts as reached).
const peakOf = (r: GameRound, bl: number, br: number) => {
  const sides = [gainAt(r.gain_left, bl), gainAt(r.gain_right, br)].filter((x): x is { v: number; t: number } => x !== null);
  return sides.length ? sides.reduce((s, x) => s + x.t, 0) / sides.length : 1;
};
// The rounds played so far against AutoVUS: verdicts that agree, and the mean closeness to the peak gain.
const summary = (rounds: GameRound[], picks: [number, number][]) => {
  const played = rounds.slice(0, picks.length);
  const matched = played.filter((x, k) => agrees(typeOf(picks[k][0], picks[k][1]), x.type)).length;
  const peak = played.length ? played.reduce((s, x, k) => s + peakOf(x, picks[k][0], picks[k][1]), 0) / played.length : 0;
  return { matched, peak, rounds: played.length };
};

// candidates: every round of game_candidates.json in order, numbered (/play?set=candidates), to choose what joins the pool.
export function GameView({ state, candidates = false }: { state: State; candidates?: boolean }) {
  const [g, setG] = useState<GameDoc | null>(null);
  const [error, setError] = useState(false);
  const [i, setI] = useState(0);
  const [phase, setPhase] = useState<Phase>("drag");
  const [bl, setBl] = useState(0);
  const [br, setBr] = useState(0);
  const [picks, setPicks] = useState<[number, number][]>([]);
  const plot = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<"l" | "r" | null>(null);
  const [why, setWhy] = useState<{ a: Anomaly; curves: CurvesDoc } | null>(null);
  const [yours, setYours] = useState<BuffersDoc | null>(null);
  const [player, setPlayerName] = useState(ANON);
  // Recording for Played so far: the latest rounds and picks, and whether this game was sent already.
  const latest = useRef<{ rounds: GameRound[]; picks: [number, number][] }>({ rounds: [], picks: [] });
  const recorded = useRef(false);
  useEffect(() => { latest.current = { rounds: g?.rounds ?? [], picks }; }, [g, picks]);
  const record = useCallback((keepalive: boolean) => {
    const { rounds, picks } = latest.current;
    if (candidates || recorded.current || !picks.length || !rounds.length) return;
    recorded.current = true;
    const s = summary(rounds, picks);
    saveGame({ name: getPlayer() || ANON, mode: "game", matched: s.matched, rounds: s.rounds, peak: s.peak }, keepalive);
  }, [candidates]);
  useEffect(() => {
    setPlayerName(getPlayer() || ANON);
    const leave = () => record(true);
    window.addEventListener("pagehide", leave);
    return () => { window.removeEventListener("pagehide", leave); record(true); };   // leaving the game mid-way records the rounds played
  }, [record]);

  // A new draw: on arrival and on Play again, straight into round 1.
  const gen = useRef(0);
  const draw = useCallback(() => {
    record(false);   // a gear switch mid-game ends that game: record the rounds played (no-op on arrival and after the end)
    setG(null); setI(0); setBl(0); setBr(0); setPicks([]); setPhase("drag"); recorded.current = false;
    setError(false);
    // The Python side can be restarting: ask a few times before giving up.
    const my = ++gen.current;   // a newer draw (gear switch, Try again) wins over this one and its retries
    const ask = (left: number) => loadGame<GameDoc>(state.benchmark, state.L, candidates ? undefined : ROUNDS, candidates ? "candidates" : undefined).then((d) => {
      if (my !== gen.current) return;
      if (d && d.rounds.length) setG(d);
      else if (left > 0) setTimeout(() => ask(left - 1), 1500);
      else setError(true);
    });
    ask(6);
  }, [state.benchmark, state.L, candidates, record]);
  useEffect(() => { draw(); }, [draw]);
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
  // The AutoVUS value of the series with the player's buffers on this anomaly, next to AutoVUS's own.
  const pick = picks[i];
  useEffect(() => {
    setYours(null);
    if (phase !== "reveal" || !cur || !pick) return;
    let live = true;
    scoreBuffers(state.benchmark, cur.file, cur.detector, cur.k, cur.L, pick[0], pick[1]).then((d) => { if (live && d) setYours(d); });
    return () => { live = false; };
  }, [phase, cur, pick, state.benchmark]);
  // The last round is done: the game goes to Played so far.
  useEffect(() => { if (phase === "end") record(false); }, [phase, record]);
  // The round in Explore: the detector pinned, zoomed on the window, the anomaly's buffer explained.
  const review = (x: GameRound) => href("/local", state, { s: x.file, det: x.detector, v: `${x.lo},${x.hi}`, a: String(x.k) });

  if (error) return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      <div className="panel" style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
        <span className="muted">The Python side did not answer.</span>
        <button type="button" className="btn btn-primary" onClick={draw}>Try again</button>
      </div>
    </main>
  );
  if (!g) return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      <div className="panel" style={{ height: 420, alignItems: "center", justifyContent: "center" }}><span className="muted" style={{ fontSize: 14 }}>Loading the rounds at L = {state.L}…</span></div>
    </main>
  );

  const n = g.rounds.length;
  const L = g.L;
  const header = (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{ display: "flex", gap: 8 }} aria-label={`Round ${Math.min(i + 1, n)} of ${n}`}>
          {g.rounds.map((_, k) => <span key={k} style={{ width: 28, height: 6, borderRadius: 3, background: k < i || phase === "end" ? "var(--ink)" : k === i ? "var(--accent)" : "var(--border)" }} />)}
        </div>
        {phase !== "end" && <span className="mono muted" style={{ fontSize: 12 }}>{candidates ? "Candidate" : "Round"} {i + 1} of {n}</span>}
      </div>
      <div style={{ display: "flex", gap: 18, fontSize: 12, fontWeight: 600 }}>
        <Link href={href("/play", state)} className="muted" style={{ textDecoration: "none" }}>All games</Link>
        {!candidates && <Link href={href("/play", state, { mode: "draw" })} className="muted" style={{ textDecoration: "none" }}>Draw your detection</Link>}
      </div>
    </div>
  );

  if (phase === "end") {
    const { matched, peak } = summary(g.rounds, picks);
    return (
      <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
        {header}
        <section className="panel" aria-label="Result" style={{ padding: "28px 24px 24px", gap: 20 }}>
          <h2 style={{ fontSize: 22, fontWeight: 700, textTransform: "none", letterSpacing: "-0.01em", lineHeight: 1.25 }}>
            {n} rounds: your verdict matched AutoVUS&apos;s on {matched} of {n}, and your buffers reached {Math.round(peak * 100)}&thinsp;% of the peak gain.
          </h2>
          <table className="table">
            <thead>
              <tr><th>{candidates ? "#" : "Round"}</th><th>Series</th><th>Detector</th><th>You</th><th style={{ color: "var(--accent)" }}>AutoVUS</th><th /></tr>
            </thead>
            <tbody>
              {g.rounds.map((x, k) => {
                const [pl, pr] = picks[k] ?? [0, 0];
                const yt = typeOf(pl, pr);
                return (
                  <tr key={k} style={{ cursor: "default" }}>
                    <td className="mono">{k + 1}</td>
                    <td className="mono" style={{ fontSize: 11 }}>{seriesName(x.file)}</td>
                    <td className="sans" style={{ fontWeight: 600 }}>{x.detector}</td>
                    <td className="sans"><span style={{ color: agrees(yt, x.type) ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yt]}</span> <span className="mono muted">(<span style={{ color: BEFORE }}>{pl}</span>, <span style={{ color: AFTER }}>{pr}</span>)</span></td>
                    <td className="sans"><span style={{ fontWeight: 600 }}>{NAMES[x.type]}</span> <span className="mono muted">(<span style={{ color: BEFORE }}>{x.b_left}</span>, <span style={{ color: AFTER }}>{x.b_right}</span>)</span></td>
                    <td className="r"><Link href={review(x)} style={{ fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Explore</Link></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
            {!candidates && <span className="muted" style={{ fontSize: 13 }}>Added to Played so far as <b style={{ color: "var(--ink)" }}>{player}</b>.</span>}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginLeft: "auto" }}>
              {!candidates && <Link href={href("/play", state, { mode: "draw" })} style={{ fontSize: 13, fontWeight: 600, color: "var(--accent)", textDecoration: "none" }}>Draw your detection</Link>}
              <button type="button" className="btn btn-primary" onClick={draw} style={{ height: 44, padding: "0 22px", fontSize: 14 }}>Play again</button>
            </div>
          </div>
        </section>
      </main>
    );
  }

  const r = g.rounds[i];
  const N = r.hi - r.lo;
  const at = (p: number) => `${(Math.max(0, Math.min(N, p - r.lo)) / N) * 100}%`;
  const zone = (a: number, b: number, color: string, dashed: boolean, opacity = 0.22) => {
    const x0 = Math.max(r.lo, a), x1 = Math.min(r.hi, b);
    if (x1 <= x0) return null;
    const style = { position: "absolute" as const, top: dashed ? -6 : 0, bottom: dashed ? -6 : 0, left: at(x0), width: `${((x1 - x0) / N) * 100}%` };
    return dashed ? <div style={{ ...style, boxSizing: "border-box", border: `2px dashed ${color}`, borderRadius: 3 }} /> : <div style={{ ...style, background: color, opacity }} />;
  };
  // The live gain of the player's two buffers: the number above each edge, brighter the closer to the peak.
  const gl = gainAt(r.gain_left, bl), gr = gainAt(r.gain_right, br);
  const glow = (g: { t: number } | null) => (g ? 0.08 + 0.4 * g.t : 0.22);
  const line = (a: number[]) => a.map((v, k) => `${(((k + 0.5) / a.length) * 1000).toFixed(1)},${(96 - v * 92).toFixed(1)}`).join(" ");
  const yourType = typeOf(bl, br);
  const sameType = agrees(yourType, r.type);
  const last = i === n - 1;

  const setFrom = (side: "l" | "r", clientX: number) => {
    const box = plot.current?.getBoundingClientRect();
    if (!box) return;
    const p = r.lo + ((clientX - box.left) / box.width) * N;
    const clamp = (v: number) => Math.max(0, Math.min(L, Math.round(v)));
    if (side === "l") setBl(clamp(r.start - p)); else setBr(clamp(p - (r.end + 1)));
  };
  const handle = (side: "l" | "r") => {
    const left = side === "l", color = left ? BEFORE : AFTER, gn = left ? gl : gr;
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
        {gn && <span className="mono" style={{ position: "absolute", top: -GAIN, [left ? "right" : "left"]: 9, height: GAIN - 6, boxSizing: "border-box", padding: "0 7px", borderRadius: 4, fontSize: 12, lineHeight: `${GAIN - 9}px`, fontWeight: 600, whiteSpace: "nowrap", color: "var(--ink)",
          border: `1.5px solid color-mix(in srgb, ${color} ${Math.round(25 + 75 * gn.t)}%, transparent)`, background: `color-mix(in srgb, ${color} ${Math.round(6 + 54 * gn.t)}%, transparent)` }}>gain {gn.v.toFixed(3)}</span>}
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
        <div ref={plot} style={{ position: "relative", height: 230, marginTop: phase === "drag" ? GAIN : 0, marginBottom: TAB, touchAction: "none", userSelect: "none" }}>
          {zone(r.start - bl, r.start, BEFORE, false, phase === "drag" ? glow(gl) : 0.22)}
          {zone(r.end + 1, r.end + 1 + br, AFTER, false, phase === "drag" ? glow(gr) : 0.22)}
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
              <thead><tr><th /><th>Detection</th><th style={{ color: BEFORE }}>Before</th><th style={{ color: AFTER }}>After</th><th className="r">AutoVUS of the series</th></tr></thead>
              <tbody>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600 }}>You</td><td className="sans" style={{ color: sameType ? "var(--exact)" : "var(--late)", fontWeight: 600 }}>{NAMES[yourType]}</td><td className="mono">{bl}</td><td className="mono">{br}</td><td className="mono r">{yours ? f3(yours.yours) : "…"}</td></tr>
                <tr style={{ cursor: "default" }}><td className="sans" style={{ fontWeight: 600, color: "var(--accent)" }}>AutoVUS</td><td className="sans" style={{ fontWeight: 600 }}>{NAMES[r.type]}</td><td className="mono">{r.b_left}</td><td className="mono">{r.b_right}</td><td className="mono r" style={{ color: "var(--accent)" }}>{f3(r.autovus ?? yours?.autovus)}</td></tr>
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
            <span className="mono" style={{ fontSize: 12 }}>{NAMES[r.type]} · b* = ({r.b_left}, {r.b_right}) · you ({bl}, {br})</span>
            <Link href={review(r)} className="btn" style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", textDecoration: "none" }}>Open in Explore</Link>
          </div>
          {why ? <Drawer curves={why.curves} a={why.a} L={L} height={110} you={{ left: bl, right: br }} /> : <div style={{ height: 134 }} />}
        </section>
      )}
    </main>
  );
}
