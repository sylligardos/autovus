"use client";
// Why this buffer: in the middle the score around one anomaly with the label
// and the selected buffers; on each side that side's ratio curve, whose first
// argmax is the selected buffer.

import { fmt, type Anomaly, type CurvesDoc } from "@/lib/model";
import { Lane, type Band } from "./Lane";
import { LineChart } from "./LineChart";

export const BEFORE = "var(--early)", AFTER = "var(--down)";

// you: the player's buffers (the game), drawn as a hollow dot on each side's curve next to AutoVUS's filled one.
export function Drawer({ curves, a, L, height = 150, you }: { curves: CurvesDoc; a: Anomaly; L: number; height?: number; you?: { left: number; right: number } }) {
  const z = curves.zoom;
  // The index of buffer b on the curves' x axis (the nearest one when b is not on it).
  const idx = (b: number) => curves.buffers.reduce((best, v, i) => (Math.abs(v - b) < Math.abs(curves.buffers[best] - b) ? i : best), 0);
  const yours = (b: number | undefined) => (b === undefined ? [] : [{ line: 0, i: idx(b), hollow: true, label: "you" }]);
  const bands: Band[] = [{ from: a.start - z.from, to: a.end + 1 - z.from, color: "var(--label)", opacity: 0.45 }];
  if (a.b_left > 0) bands.push({ from: a.start - a.b_left - z.from, to: a.start - z.from, color: BEFORE, opacity: 0.35 });
  if (a.b_right > 0) bands.push({ from: a.end + 1 - z.from, to: a.end + 1 + a.b_right - z.from, color: AFTER, opacity: 0.35 });
  const before = curves.sides.before ?? curves.sides.both, after = curves.sides.after ?? curves.sides.both;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 3fr) minmax(0, 5fr) minmax(0, 3fr)", gap: 16 }}>
      {before ? (
        <LineChart x={curves.buffers} title={`before · ratio · b* = ${before.b}${you ? ` · you ${you.left}` : ""}`} height={height}
          lines={[{ y: before.rule_curve, color: BEFORE, width: 1.8 }]} mark={before.b} dot={{ line: 0, i: before.idx }} dots={yours(you?.left)} left="b = 0" right={String(L)} />
      ) : <span />}
      <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
        <figcaption className="muted" style={{ fontSize: 12 }}>score · label · buffers · points {fmt(z.from)} to {fmt(z.to)}</figcaption>
        <Lane lo={z.lo} hi={z.hi} bins={z.bins} n={z.to - z.from} height={height} ymin={0} ymax={1} bands={bands} />
      </figure>
      {after ? (
        <LineChart x={curves.buffers} title={`after · ratio · b* = ${after.b}${you ? ` · you ${you.right}` : ""}`} height={height}
          lines={[{ y: after.rule_curve, color: AFTER, width: 1.8 }]} mark={after.b} dot={{ line: 0, i: after.idx }} dots={yours(you?.right)} left="b = 0" right={String(L)} />
      ) : <span />}
    </div>
  );
}
