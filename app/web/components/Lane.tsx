"use client";
// One strip of a series or a score: min-max bins drawn as a filled band
// (a line when the bins hold single points), bands behind it in point
// coordinates, and clickable zones on top. With onZoom, a drag across the
// strip picks a point range and a double click resets it.

import { useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from "react";

export type Band = { from: number; to: number; color: string; opacity?: number; key?: string };
export type Zone = Band & { onEnter?: () => void; onLeave?: () => void; onClick?: () => void; selected?: boolean };

export function Lane({ lo, hi, bins, n, start = 0, end, view, onZoom, height = 56, bands = [], zones = [], ymin, ymax, stroke = "var(--ink)", style, children }: {
  lo: number[]; hi: number[]; bins: number; n: number; start?: number; end?: number; view?: [number, number] | null; onZoom?: (v: [number, number] | null) => void;
  height?: number; bands?: Band[]; zones?: Zone[]; ymin?: number; ymax?: number; stroke?: string; style?: CSSProperties; children?: ReactNode;
}) {
  const stop = end ?? n;
  const [v0, v1] = view ?? [0, Math.max(n, 1)];
  const w = Math.max(v1 - v0, 1e-9);
  const mn = ymin ?? Math.min(...lo), mx = ymax ?? Math.max(...hi);
  const span = mx > mn ? mx - mn : 1;
  const y = (v: number) => 98 - ((v - mn) / span) * 96;
  const px = (i: number) => start + ((i + 0.5) * (stop - start)) / bins;
  const top = hi.map((v, i) => `${px(i)},${y(v).toFixed(2)}`);
  const bottom = lo.map((v, i) => `${px(i)},${y(v).toFixed(2)}`).reverse();
  const line = bins === stop - start;

  // Drag to zoom: the selection lives in fractions of the strip width.
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x0: number; id: number } | null>(null);
  const dragged = useRef(false);
  const [sel, setSel] = useState<[number, number] | null>(null);
  const frac = (e: PointerEvent) => { const r = box.current!.getBoundingClientRect(); return Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1); };
  const down = (e: PointerEvent) => { if (!onZoom || e.button !== 0) return; drag.current = { x0: frac(e), id: e.pointerId }; dragged.current = false; };
  const move = (e: PointerEvent) => {
    const d = drag.current; if (!d) return;
    const f = frac(e);
    if (!dragged.current && Math.abs(f - d.x0) * box.current!.getBoundingClientRect().width > 4) { dragged.current = true; box.current!.setPointerCapture(d.id); }
    if (dragged.current) setSel([Math.min(d.x0, f), Math.max(d.x0, f)]);
  };
  const up = () => {
    if (drag.current && dragged.current && sel && onZoom) {
      const a = v0 + sel[0] * w, b = v0 + sel[1] * w;
      if (b - a >= 2) onZoom([Math.floor(a), Math.ceil(b)]);
    }
    drag.current = null; setSel(null);
  };

  return (
    <div ref={box} className="lane" style={{ position: "relative", minWidth: 0, cursor: onZoom ? "crosshair" : undefined, touchAction: onZoom ? "none" : undefined, ...style }}
      onMouseDown={onZoom ? (e) => e.preventDefault() : undefined} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onDoubleClick={onZoom ? () => onZoom(null) : undefined}>
      <svg viewBox={`${v0} 0 ${w} 100`} width="100%" height={height} preserveAspectRatio="none" style={{ display: "block", background: "var(--subtle)", borderRadius: 4 }} aria-hidden="true">
        {bands.map((b, i) => (
          <rect key={b.key ?? i} x={b.from} y={0} width={Math.max(b.to - b.from, w / 1200)} height={100} fill={b.color} opacity={b.opacity ?? 0.4} />
        ))}
        {line ? (
          <polyline points={top.join(" ")} fill="none" stroke={stroke} strokeWidth={1.1} vectorEffect="non-scaling-stroke" />
        ) : (
          <polygon points={[...top, ...bottom].join(" ")} fill={stroke} stroke={stroke} strokeWidth={0.8} vectorEffect="non-scaling-stroke" opacity={0.9} />
        )}
        {zones.map((z, i) => (
          <rect key={z.key ?? `z${i}`} x={z.from} y={0} width={Math.max(z.to - z.from, w / 400)} height={100}
            fill={z.color} opacity={0} stroke={z.selected ? "var(--accent)" : "none"} strokeWidth={2} vectorEffect="non-scaling-stroke"
            style={{ cursor: z.onClick ? "pointer" : "default" }} onMouseEnter={z.onEnter} onMouseLeave={z.onLeave} onClick={() => { if (!dragged.current) z.onClick?.(); }} />
        ))}
      </svg>
      {sel && <div style={{ position: "absolute", top: 0, bottom: 0, left: `${sel[0] * 100}%`, width: `${(sel[1] - sel[0]) * 100}%`, background: "var(--accent)", opacity: 0.15, pointerEvents: "none" }} />}
      {children}
    </div>
  );
}
