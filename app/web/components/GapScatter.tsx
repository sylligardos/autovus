"use client";
// A scatter of cases against the diagonal: x and y are two scores of the same
// case, the colour is how far the case sits from y = x (blue below, orange
// above, grey on it). Hover names the nearest case, a click picks it. The
// caller can mark cases: one gets a ring, several are redrawn in ink. With
// oneSided the cases all sit on or above the diagonal and the key starts at 0.

import { useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "@/lib/theme";

export type ScatterPoint = { x: number; y: number; g: number; title: string; sub: string };

const H = 380, PAD = { l: 58, r: 14, t: 12, b: 34 };
const signed = (g: number) => (g >= 0 ? "+" : "−") + Math.abs(g).toFixed(2);

function rgb(hex: string): [number, number, number] {
  const h = hex.trim().replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

export function GapScatter({ points, xlabel, ylabel, glabel, mark, onPick, far: FAR = 0.3, above = "--late", oneSided = false, tip: tipLine }: {
  points: ScatterPoint[]; xlabel: string; ylabel: string; glabel: string; mark: number[]; onPick: (i: number) => void;
  far?: number;            // gap at which the colour is full
  above?: string;          // colour variable of the cases above the diagonal
  oneSided?: boolean;
  tip?: (i: number) => string;   // last line of the tooltip, in place of the two values and the gap
}) {
  const theme = useTheme();
  const box = useRef<HTMLDivElement>(null);
  const base = useRef<HTMLCanvasElement>(null);
  const over = useRef<HTMLCanvasElement>(null);
  const [W, setW] = useState(0);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.floor(el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Both axes share one domain, so the diagonal is y = x.
  const lo = useMemo(() => {
    let m = 0;
    for (const p of points) m = Math.min(m, p.x, p.y);
    return Math.floor(m * 10) / 10;
  }, [points]);
  const sx = (v: number) => PAD.l + ((v - lo) / (1 - lo)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - lo) / (1 - lo)) * (H - PAD.t - PAD.b);

  const colors = () => {
    const css = getComputedStyle(document.documentElement);
    const v = (n: string) => css.getPropertyValue(n);
    return { below: rgb(v("--accent")), above: rgb(v(above)), mid: rgb(v("--missed")), ink: v("--ink"), muted: v("--muted"), grid: v("--border"), mono: getComputedStyle(base.current!).fontFamily };
  };
  const fit = (c: HTMLCanvasElement) => {
    const r = window.devicePixelRatio || 1;
    c.width = W * r; c.height = H * r;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, W, H);
    return ctx;
  };

  useEffect(() => {
    if (!base.current || W === 0) return;
    const ctx = fit(base.current), c = colors();
    ctx.font = `10px ${c.mono}`; ctx.fillStyle = c.muted; ctx.strokeStyle = c.grid; ctx.lineWidth = 1;
    for (let v = Math.ceil(lo * 4) / 4; v <= 1.0001; v += 0.25) {
      ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(sx(v), PAD.t); ctx.lineTo(sx(v), H - PAD.b); ctx.moveTo(PAD.l, sy(v)); ctx.lineTo(W - PAD.r, sy(v)); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(v.toFixed(2), sx(v), H - PAD.b + 4);
      ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(v.toFixed(2), PAD.l - 6, sy(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(xlabel, (PAD.l + W - PAD.r) / 2, H - 2);
    ctx.save(); ctx.translate(9, (PAD.t + H - PAD.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); ctx.restore();
    ctx.strokeStyle = c.muted; ctx.beginPath(); ctx.moveTo(sx(lo), sy(lo)); ctx.lineTo(sx(1), sy(1)); ctx.stroke();
    // Far cases last, so they sit on top.
    const order = points.map((_, i) => i).sort((a, b) => Math.abs(points[a].g) - Math.abs(points[b].g));
    for (const i of order) {
      const p = points[i], t = Math.min(Math.abs(p.g) / FAR, 1);
      const [r, g, b] = mix(c.mid, p.g >= 0 ? c.above : c.below, t);
      ctx.fillStyle = `rgba(${r},${g},${b},${0.25 + 0.75 * t})`;
      ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 2 + t, 0, 2 * Math.PI); ctx.fill();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, W, lo, theme, xlabel, ylabel, FAR, above]);

  useEffect(() => {
    if (!over.current || W === 0) return;
    const ctx = fit(over.current), c = colors();
    ctx.strokeStyle = c.ink; ctx.fillStyle = c.ink; ctx.lineWidth = 2;
    const ring = (i: number | null) => {
      const p = i === null ? undefined : points[i];
      if (p) { ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 6, 0, 2 * Math.PI); ctx.stroke(); }
    };
    if (mark.length === 1) ring(mark[0]);
    else for (const i of mark) { const p = points[i]; if (p) { ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 2.5, 0, 2 * Math.PI); ctx.fill(); } }
    ring(hover);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, W, lo, theme, mark, hover]);

  const nearest = (e: React.MouseEvent) => {
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best: number | null = null, bd = 12 * 12;
    for (let i = 0; i < points.length; i++) {
      const dx = sx(points[i].x) - mx, dy = sy(points[i].y) - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };

  const tip = hover === null ? undefined : points[hover];
  const css = (c: string, a: number) => `color-mix(in srgb, var(${c}) ${a}%, var(--missed))`;
  return (
    <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 8 }}>
      <div ref={box} style={{ position: "relative", height: H }}>
        <canvas ref={base} className="mono" style={{ position: "absolute", inset: 0, width: "100%", height: H }} role="img" aria-label={`${ylabel} against ${xlabel}`} />
        <canvas ref={over} style={{ position: "absolute", inset: 0, width: "100%", height: H, cursor: hover === null ? "default" : "pointer" }}
          onMouseMove={(e) => setHover(nearest(e))} onMouseLeave={() => setHover(null)} onClick={(e) => { const i = nearest(e); if (i !== null) onPick(i); }} />
        {tip && (
          <div style={{ position: "absolute", left: Math.min(Math.max(sx(tip.x), 140), W - 140), top: sy(tip.y) - 12, transform: "translate(-50%, -100%)", pointerEvents: "none", zIndex: 2,
            padding: "7px 10px", background: "var(--tooltip-bg)", color: "var(--tooltip-ink)", borderRadius: 6, fontSize: 12, whiteSpace: "nowrap", display: "flex", flexDirection: "column", gap: 2 }}>
            <span className="mono" style={{ fontSize: 11 }}>{tip.title}</span>
            <span style={{ fontWeight: 600 }}>{tip.sub}</span>
            <span className="mono">{tipLine ? tipLine(hover!) : `${ylabel} ${tip.y.toFixed(2)} · ${xlabel} ${tip.x.toFixed(2)} · ${signed(tip.g)}`}</span>
          </div>
        )}
      </div>
      <figcaption style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11 }} className="muted">
        <span>{glabel}</span>
        <span className="mono">{oneSided ? "0" : `−${FAR.toFixed(2)}`}</span>
        <span aria-hidden="true" style={{ width: 140, height: 8, borderRadius: 2, background: oneSided ? `linear-gradient(to right, var(--missed), ${css(above, 50)}, var(${above}))`
          : `linear-gradient(to right, var(--accent), ${css("--accent", 50)}, var(--missed), ${css(above, 50)}, var(${above}))` }} />
        <span className="mono">+{FAR.toFixed(2)}</span>
      </figcaption>
    </figure>
  );
}
