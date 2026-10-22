"use client";
// One strip of a series or a score: min-max bins drawn as a filled band
// (a line when the bins hold single points), bands behind it in point
// coordinates, and clickable zones on top.

import type { CSSProperties, ReactNode } from "react";

export type Band = { from: number; to: number; color: string; opacity?: number; key?: string };
export type Zone = Band & { onEnter?: () => void; onLeave?: () => void; onClick?: () => void; selected?: boolean };

export function Lane({ lo, hi, bins, n, height = 56, bands = [], zones = [], ymin, ymax, stroke = "var(--ink)", style, children }: {
  lo: number[]; hi: number[]; bins: number; n: number; height?: number;
  bands?: Band[]; zones?: Zone[]; ymin?: number; ymax?: number; stroke?: string; style?: CSSProperties; children?: ReactNode;
}) {
  const mn = ymin ?? Math.min(...lo), mx = ymax ?? Math.max(...hi);
  const span = mx > mn ? mx - mn : 1;
  const y = (v: number) => 98 - ((v - mn) / span) * 96;
  const x = (p: number) => (p / Math.max(n, 1)) * bins;
  const top = hi.map((v, i) => `${i + 0.5},${y(v).toFixed(2)}`);
  const bottom = lo.map((v, i) => `${i + 0.5},${y(v).toFixed(2)}`).reverse();
  const line = bins === n;
  return (
    <div className="lane" style={{ position: "relative", minWidth: 0, ...style }}>
      <svg viewBox={`0 0 ${bins} 100`} width="100%" height={height} preserveAspectRatio="none" style={{ display: "block", background: "var(--subtle)", borderRadius: 4 }} aria-hidden="true">
        {bands.map((b, i) => (
          <rect key={b.key ?? i} x={x(b.from)} y={0} width={Math.max(x(b.to) - x(b.from), bins / 1200)} height={100} fill={b.color} opacity={b.opacity ?? 0.4} />
        ))}
        {line ? (
          <polyline points={top.join(" ")} fill="none" stroke={stroke} strokeWidth={1.1} vectorEffect="non-scaling-stroke" />
        ) : (
          <polygon points={[...top, ...bottom].join(" ")} fill={stroke} stroke={stroke} strokeWidth={0.8} vectorEffect="non-scaling-stroke" opacity={0.9} />
        )}
        {zones.map((z, i) => (
          <rect key={z.key ?? `z${i}`} x={x(z.from)} y={0} width={Math.max(x(z.to) - x(z.from), bins / 400)} height={100}
            fill={z.color} opacity={z.selected ? 0.0 : 0.0} stroke={z.selected ? "var(--accent)" : "none"} strokeWidth={2} vectorEffect="non-scaling-stroke"
            style={{ cursor: z.onClick ? "pointer" : "default" }} onMouseEnter={z.onEnter} onMouseLeave={z.onLeave} onClick={z.onClick} />
        ))}
      </svg>
      {children}
    </div>
  );
}
