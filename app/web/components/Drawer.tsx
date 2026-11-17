"use client";
// Why this buffer: in the middle the score around one anomaly with the label
// and the selected buffers; on each side that side's ratio curve, whose first
// argmax is the selected buffer.

import { fmt, type Anomaly, type CurvesDoc } from "@/lib/model";
import { Lane, type Band } from "./Lane";
import { LineChart } from "./LineChart";

export const BEFORE = "var(--early)", AFTER = "var(--down)";

export function Drawer({ curves, a, L }: { curves: CurvesDoc; a: Anomaly; L: number }) {
  const z = curves.zoom;
  const bands: Band[] = [{ from: a.start - z.from, to: a.end + 1 - z.from, color: "var(--label)", opacity: 0.45 }];
  if (a.b_left > 0) bands.push({ from: a.start - a.b_left - z.from, to: a.start - z.from, color: BEFORE, opacity: 0.35 });
  if (a.b_right > 0) bands.push({ from: a.end + 1 - z.from, to: a.end + 1 + a.b_right - z.from, color: AFTER, opacity: 0.35 });
  const before = curves.sides.before ?? curves.sides.both, after = curves.sides.after ?? curves.sides.both;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 3fr) minmax(0, 5fr) minmax(0, 3fr)", gap: 16 }}>
      {before ? (
        <LineChart x={curves.buffers} title={`before · ratio · b* = ${before.b}`} height={150}
          lines={[{ y: before.rule_curve, color: BEFORE, width: 1.8 }]} mark={before.b} dot={{ line: 0, i: before.idx }} left="b = 0" right={String(L)} />
      ) : <span />}
      <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
        <figcaption className="muted" style={{ fontSize: 12 }}>score · label · buffers · points {fmt(z.from)} to {fmt(z.to)}</figcaption>
        <Lane lo={z.lo} hi={z.hi} bins={z.bins} n={z.to - z.from} height={150} ymin={0} ymax={1} bands={bands} />
      </figure>
      {after ? (
        <LineChart x={curves.buffers} title={`after · ratio · b* = ${after.b}`} height={150}
          lines={[{ y: after.rule_curve, color: AFTER, width: 1.8 }]} mark={after.b} dot={{ line: 0, i: after.idx }} left="b = 0" right={String(L)} />
      ) : <span />}
    </div>
  );
}
