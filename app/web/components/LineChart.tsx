"use client";
// A small line chart: several series over one x, a dashed vertical mark,
// dots on a series (filled, or hollow with a label), axis labels at the corners.

export type Line = { y: number[]; color: string; width?: number; dash?: string; label?: string };
export type Dot = { line: number; i: number; hollow?: boolean; label?: string };

export function LineChart({ x, lines, mark, dot, dots = [], height = 150, ymin, ymax, xlabel, left, right, title }: {
  x: number[]; lines: Line[]; mark?: number; dot?: Dot; dots?: Dot[]; height?: number;
  ymin?: number; ymax?: number; xlabel?: string; left?: string; right?: string; title?: string;
}) {
  const W = 330, H = 150, P = 6;
  const all = lines.flatMap((l) => l.y).filter(Number.isFinite);
  const mn = ymin ?? Math.min(...all), mx = ymax ?? Math.max(...all);
  const span = mx > mn ? mx - mn : 1;
  const xmin = x[0] ?? 0, xmax = x[x.length - 1] ?? 1;
  const sx = (v: number) => P + ((v - xmin) / Math.max(xmax - xmin, 1)) * (W - 2 * P);
  const sy = (v: number) => H - P - ((v - mn) / span) * (H - 2 * P - 14);
  return (
    <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
      {title && <figcaption className="muted" style={{ fontSize: 12 }}>{title}</figcaption>}
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={height} preserveAspectRatio="none" style={{ display: "block", background: "var(--subtle)", borderRadius: 4 }} role="img" aria-label={title}>
        {lines.map((l, i) => (
          <polyline key={i} points={l.y.map((v, j) => `${sx(x[j]).toFixed(1)},${sy(v).toFixed(1)}`).join(" ")} fill="none"
            stroke={l.color} strokeWidth={l.width ?? 1.6} strokeDasharray={l.dash} vectorEffect="non-scaling-stroke" />
        ))}
        {mark !== undefined && <line x1={sx(mark)} y1={0} x2={sx(mark)} y2={H} stroke="var(--ink)" strokeWidth={1} strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />}
        {[...(dot ? [dot] : []), ...dots].map((d, k) => lines[d.line] && (
          <g key={k}>
            <circle cx={sx(x[d.i])} cy={sy(lines[d.line].y[d.i])} r={5} fill={d.hollow ? "var(--panel)" : lines[d.line].color} stroke={d.hollow ? "var(--ink)" : "none"} strokeWidth={2} vectorEffect="non-scaling-stroke" />
            {d.label && <text x={sx(x[d.i])} y={sy(lines[d.line].y[d.i]) - 9} fontFamily="var(--mono)" fontSize={10} fontWeight={600} fill="var(--ink)" textAnchor="middle">{d.label}</text>}
          </g>
        ))}
        {left && <text x={4} y={H - 2} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">{left}</text>}
        {right && <text x={W - 4} y={H - 2} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="end">{right}</text>}
        {xlabel && <text x={W / 2} y={H - 2} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="middle">{xlabel}</text>}
      </svg>
    </figure>
  );
}
