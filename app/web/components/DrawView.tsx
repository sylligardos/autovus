"use client";
// Draw your detection (the Play page, /play?mode=draw): the window of a game round with its labels,
// and an empty score lane the visitor paints. Every stroke is scored live
// (POST /draw): the buffers AutoVUS picks before and after every label, the
// detection type, the AutoVUS value and the time it took.

import Link from "next/link";
import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import { loadGame, saveGame, scoreDrawing } from "@/lib/api";
import { TYPE_COLOR, f3, href, type State } from "@/lib/model";
import { ANON, getPlayer } from "@/lib/player";
import { seriesName, type DrawDoc, type GameDoc, type GameRound } from "@/lib/local";
import { AFTER, BEFORE } from "./Drawer";
import { NAMES } from "./GameView";

const POOL = 16;                                   // rounds asked per draw; the windows come from them
const MAX_LABELS = 4;                              // a window with more labels has no room for their names
const CHIPS = 48;                                  // room above the plot for the names, on two rows
const SERIES = 96;                                 // height of the series lane
const LANE = 190;                                  // height of the lane the visitor paints
const H = SERIES + 16 + LANE;
const HEIGHT = 0.8;                                // score of the starting shapes

type Shape = "on" | "early" | "late" | "wide" | "everywhere" | "noise" | "real";

export function DrawView({ state }: { state: State }) {
  const [rounds, setRounds] = useState<GameRound[] | null>(null);
  const [error, setError] = useState(false);
  const [i, setI] = useState(0);
  const [score, setScore] = useState<number[]>([]);
  const [res, setRes] = useState<DrawDoc | null>(null);
  // Recording for Played so far: the latest scored drawing of this window; sent when the visitor leaves the window.
  const latestRes = useRef<{ r: GameRound; res: DrawDoc } | null>(null);
  const record = useCallback((keepalive: boolean) => {
    const job = latestRes.current;
    if (!job || job.res.autovus == null) return;
    latestRes.current = null;
    saveGame({ name: getPlayer() || ANON, mode: "draw", labels: job.res.anomalies.length, types: job.res.anomalies.map((a) => a.type), autovus: job.res.autovus }, keepalive);
  }, []);
  useEffect(() => {
    const leave = () => record(true);
    window.addEventListener("pagehide", leave);
    return () => { window.removeEventListener("pagehide", leave); record(true); };
  }, [record]);
  const lane = useRef<HTMLDivElement>(null);
  const last = useRef<{ i: number; v: number } | null>(null);
  const busy = useRef(false), dirty = useRef(false), latest = useRef<{ r: GameRound; score: number[] } | null>(null);
  // The window each result was computed for, and the window last sent: a result or a drawing never crosses windows.
  const resFor = useRef<GameRound | null>(null), sentFor = useRef<GameRound | null>(null);

  // A new set of windows: one per series, the ones whose labels have room for their names.
  const gen = useRef(0);
  const draw = useCallback(() => {
    setRounds(null); setError(false);
    const my = ++gen.current;   // a newer draw (gear switch, Try again) wins over this one and its retries
    const ask = (left: number) => loadGame<GameDoc>(state.benchmark, state.L, POOL).then((d) => {
      if (my !== gen.current) return;
      const seen = new Set<string>();
      const ok = (d?.rounds ?? []).filter((x) => x.labels?.length > 0 && x.labels.length <= MAX_LABELS && !seen.has(x.file) && seen.add(x.file));
      if (ok.length) { setRounds(ok); setI(0); }
      else if (left > 0) setTimeout(() => ask(left - 1), 1500);
      else setError(true);
    });
    ask(6);
  }, [state.benchmark, state.L]);
  useEffect(() => { draw(); }, [draw]);

  const r = rounds?.[i] ?? null;
  const N = r ? r.hi - r.lo : 0;
  useEffect(() => { setScore(r ? new Array(r.hi - r.lo).fill(0) : []); setRes(null); }, [r]);
  useEffect(() => { if (r && res && resFor.current === r) latestRes.current = { r, res }; }, [r, res]);
  useEffect(() => () => record(false), [r, record]);              // another window: the drawing of the last one is recorded

  // Score the latest drawing; one request at a time, the newest drawing always goes last.
  const send = useCallback(async () => {
    if (busy.current) { dirty.current = true; return; }
    busy.current = true;
    do {
      dirty.current = false;
      const job = latest.current;
      if (!job) break;
      const d = await scoreDrawing<DrawDoc>(state.benchmark, job.r.file, job.r.lo, job.r.hi, state.L, job.score);
      if (d && latest.current?.r === job.r) { resFor.current = job.r; setRes(d); }
    } while (dirty.current);
    busy.current = false;
  }, [state.benchmark, state.L]);
  useEffect(() => {
    if (!r || score.length !== r.hi - r.lo) return;
    if (sentFor.current !== r) { sentFor.current = r; if (latest.current && latest.current.r !== r) return; }   // new window: wait for its own blank score
    latest.current = { r, score };
    send();
  }, [r, score, send]);

  if (error) return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      <div className="panel" style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
        <span className="muted">The Python side did not answer.</span>
        <button type="button" className="btn btn-primary" onClick={draw}>Try again</button>
      </div>
    </main>
  );
  if (!r) return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      <div className="panel" style={{ height: 480, alignItems: "center", justifyContent: "center" }}><span className="muted" style={{ fontSize: 14 }}>Loading the windows at L = {state.L}…</span></div>
    </main>
  );

  const L = state.L;
  const at = (p: number) => `${(Math.max(0, Math.min(N, p - r.lo)) / N) * 100}%`;
  const band = (key: string, a: number, b: number, color: string, opacity: number) => {
    const x0 = Math.max(r.lo, a), x1 = Math.min(r.hi, b);
    return x1 > x0 ? <div key={key} style={{ position: "absolute", top: 0, bottom: 0, left: at(x0), width: `${((x1 - x0) / N) * 100}%`, minWidth: 3, background: color, opacity, pointerEvents: "none" }} /> : null;
  };
  const line = (a: number[], h: number) => a.map((v, k) => `${(((k + 0.5) / a.length) * 1000).toFixed(1)},${(h - v * h).toFixed(1)}`).join(" ");
  const empty = !score.some((v) => v > 0);

  // Painting: the pointer sets the score of the points under it, a straight line between two pointer positions.
  const paint = (e: PointerEvent<HTMLDivElement>, start: boolean) => {
    const box = lane.current?.getBoundingClientRect();
    if (!box) return;
    const x = Math.max(0, Math.min(box.width - 1, e.clientX - box.left));
    const raw = 1 - (e.clientY - box.top) / box.height;
    const v = raw < 0.03 ? 0 : Math.min(1, Math.round(raw * 100) / 100);         // along the bottom: erase
    const a = Math.floor((x / box.width) * N), b = Math.max(a + 1, Math.floor(((x + 1) / box.width) * N));
    const from = start || !last.current ? { i: a, v } : last.current;
    setScore((old) => {
      const next = old.slice();
      const lo = Math.min(from.i, a), hi = Math.max(from.i, b - 1);
      for (let k = lo; k <= hi && k < N; k++) {
        const t = a === from.i ? 1 : (k - from.i) / (a - from.i);
        next[k] = Math.max(0, Math.min(1, from.v + (v - from.v) * Math.max(0, Math.min(1, t))));
      }
      return next;
    });
    last.current = { i: a, v };
  };

  // The starting shapes: the same block on every label of the window.
  const shape = (s: Shape) => {
    if (s === "real") { setScore(r.score.slice()); return; }
    if (s === "noise") { setScore(Array.from({ length: N }, () => Math.round(Math.random() * 40) / 100)); return; }
    const next = new Array(N).fill(s === "everywhere" ? HEIGHT : 0);
    const half = Math.round(L / 2);
    const [dl, dr] = s === "early" ? [-half, -half] : s === "late" ? [half, half] : s === "wide" ? [-half, half] : [0, 0];
    if (s !== "everywhere") for (const [a, b] of r.labels) for (let k = a + dl - r.lo; k <= b + dr - r.lo; k++) if (k >= 0 && k < N) next[k] = HEIGHT;
    setScore(next);
  };
  const SHAPES: [Shape, string][] = [["on", "On the label"], ["early", "Early"], ["late", "Late"], ["wide", "Wide"], ["everywhere", "Everywhere"], ["noise", "Noise"], ["real", r.detector]];

  const anomalies = res?.anomalies ?? r.labels.map(([start, end], k) => ({ k, start, end, b_left: 0, b_right: 0, type: "missed" as const }));
  return (
    <main className="content" style={{ maxWidth: 980, margin: "0 auto" }}>
      <section className="panel" aria-label="Draw a detection" style={{ padding: "18px 24px 20px", gap: 14 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <h2 style={{ fontSize: 17, textTransform: "none", letterSpacing: 0 }}>Draw your detection. AutoVUS buffers it as you draw.</h2>
          <div style={{ display: "flex", gap: 18, fontSize: 12, fontWeight: 600, alignItems: "baseline" }}>
            <span className="mono muted" style={{ fontWeight: 400 }}>L = {L}</span>
            <Link href={href("/play", state)} className="muted" style={{ textDecoration: "none" }}>All games</Link>
            <Link href={href("/play", state, { mode: "game" })} className="muted" style={{ textDecoration: "none" }}>Find the buffer</Link>
          </div>
        </div>
        <div style={{ position: "relative", height: H, marginTop: CHIPS, userSelect: "none" }}>
          {r.labels.map(([a, b], k) => band(`l${k}`, a, b + 1, "var(--label)", 0.55))}
          {anomalies.map((a) => band(`b${a.k}`, a.start - a.b_left, a.start, BEFORE, 0.3))}
          {anomalies.map((a) => band(`a${a.k}`, a.end + 1, a.end + 1 + a.b_right, AFTER, 0.3))}
          {anomalies.map((a) => (
            <span key={`n${a.k}`} style={{ position: "absolute", top: -CHIPS + (a.k % 2) * 24, left: at((a.start + a.end + 1) / 2), transform: "translateX(-50%)", height: 20, padding: "0 8px", borderRadius: 4, display: "flex", alignItems: "center", gap: 8, whiteSpace: "nowrap", fontSize: 12, fontWeight: 600, color: TYPE_COLOR[a.type], border: `1.5px solid ${TYPE_COLOR[a.type]}`, background: "var(--panel)", pointerEvents: "none" }}>
              {NAMES[a.type]}
              <span className="mono"><span style={{ color: BEFORE }}>{a.b_left}</span><span className="muted"> · </span><span style={{ color: AFTER }}>{a.b_right}</span></span>
            </span>
          ))}
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 14, width: "100%", height: SERIES - 18, pointerEvents: "none" }}><polyline points={line(r.values, 100)} fill="none" stroke="var(--ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" /></svg>
          <div style={{ position: "absolute", left: 0, right: 0, top: SERIES + 8, borderTop: "1px solid var(--border-soft)" }} />
          <div ref={lane} role="img" aria-label="The score lane: draw the detector score here"
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); paint(e, true); }}
            onPointerMove={(e) => { if (e.buttons & 1) paint(e, false); }}
            onPointerUp={() => { last.current = null; }}
            style={{ position: "absolute", left: 0, right: 0, top: SERIES + 16, height: LANE, cursor: "crosshair", touchAction: "none", borderBottom: "1px solid var(--border)" }}>
            <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
              <polygon points={`0,100 ${line(score, 100)} 1000,100`} fill="var(--accent)" opacity={0.16} />
              <polyline points={line(score, 100)} fill="none" stroke="var(--accent)" strokeWidth={1.75} vectorEffect="non-scaling-stroke" />
            </svg>
            {empty && <span className="muted" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, pointerEvents: "none" }}>draw the detector score here</span>}
          </div>
          <span className="mono muted" style={{ position: "absolute", left: 0, top: 0, fontSize: 10, lineHeight: "12px", pointerEvents: "none" }}>{seriesName(r.file)}</span>
          <span className="mono muted" style={{ position: "absolute", left: 0, top: SERIES + 20, fontSize: 10, lineHeight: "12px", pointerEvents: "none" }}>your score</span>
        </div>
        <div className="muted" style={{ display: "flex", flexWrap: "wrap", gap: 16, fontSize: 12 }}>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, background: "var(--label)", opacity: 0.7 }} />label</span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, background: BEFORE, opacity: 0.45 }} />AutoVUS buffer before</span>
          <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 12, height: 10, background: AFTER, opacity: 0.45 }} />AutoVUS buffer after</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {SHAPES.map(([s, name]) => <button key={s} type="button" className="chip-btn" onClick={() => shape(s)}>{name}</button>)}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button type="button" className="btn" onClick={() => setScore(new Array(N).fill(0))} style={{ height: 36, padding: "0 16px", fontSize: 13 }}>Clear</button>
            <button type="button" className="btn" onClick={() => { if (i + 1 < (rounds?.length ?? 0)) setI(i + 1); else draw(); }}
              style={{ height: 36, padding: "0 16px", fontSize: 13, fontWeight: 600, background: "var(--ink)", color: "var(--panel)", borderColor: "var(--ink)" }}>Another series</button>
          </div>
        </div>
      </section>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 16 }}>
        <div className="tile" style={{ borderColor: "var(--accent)" }}><span className="n" style={{ color: "var(--accent)" }}>{res?.autovus != null ? f3(res.autovus) : "–"}</span><span className="l">AutoVUS</span></div>
        <div className="tile"><span className="n">{res?.vus_full != null ? f3(res.vus_full) : "–"}</span><span className="l">VUS-PR, L = {L}</span></div>
        <div className="tile"><span className="n">{res && res.seconds > 0 ? `${(res.seconds * 1000).toFixed(1)} ms` : "–"}</span><span className="l">AutoVUS time</span></div>
      </div>
    </main>
  );
}
