"use client";
// The toy cases on the Intro: one anomaly, a block detector, the buffers
// chosen as you click. Same view as the "Why this buffer" drawer: the score
// in the middle, the ratio curve of each side next to it.

import { useEffect, useState } from "react";
import { loadToy } from "@/lib/api";
import { TYPE_COLOR, f3, fsec, type ToyDoc } from "@/lib/model";
import { AFTER, BEFORE } from "./Drawer";
import { LineChart } from "./LineChart";
import { useRemembered } from "@/lib/remember";

const FAMILIES: [string, string[]][] = [
  ["A · same width, shifted", ["A1", "A2", "A3", "A4", "A5"]],
  ["B · half width, shifted", ["B1", "B2", "B3", "B4", "B5"]],
  ["C · wider, shifted", ["C1", "C2", "C3", "C4", "C5"]],
  ["D · inside plus a second block", ["D1-1", "D1-2", "D1-3", "D1-4", "D2-1", "D2-2", "D2-3", "D2-4"]],
  ["E · two blocks, both outside", ["E1"]],
];

export function Toy({ L }: { L: number }) {
  const [c, setC] = useRemembered("toy:case", "A4");
  const [doc, setDoc] = useState<ToyDoc | null>(null);
  useEffect(() => { let live = true; loadToy(c, L).then((d) => { if (live) setDoc(d); }); return () => { live = false; }; }, [c, L]);
  const T = 160, W = 520, H = 150;
  const x = (p: number) => (p / T) * W;
  const a = doc?.anomaly;
  const before = doc ? doc.sides.before ?? doc.sides.both : null, after = doc ? doc.sides.after ?? doc.sides.both : null;
  return (
    <section className="panel" aria-label="How the buffer is chosen">
      <div className="panel-head">
        <h2>How the buffer is chosen, on 24 toy cases</h2>
        <span className="muted" style={{ fontSize: 12 }}>one anomaly of 20 points, a block detector · L = {L}</span>
      </div>
      {FAMILIES.map(([name, cases]) => (
        <div key={name} style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", fontSize: 12 }}>
          <span className="muted" style={{ width: 200 }}>{name}</span>
          {cases.map((k) => <button key={k} type="button" className="chip-btn" aria-pressed={k === c} onClick={() => setC(k)}>{k}</button>)}
        </div>
      ))}
      {doc && a && (
        <>
          <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
            <b style={{ color: "var(--ink)" }}>{doc.case}</b> · <span className="mono" style={{ color: TYPE_COLOR[a.type], fontWeight: 600 }}>{a.type}</span> · b* = ({a.b_left}, {a.b_right}) · AutoVUS {f3(doc.autovus)} · VUS-PR {f3(doc.vus_full)} · {fsec(doc.seconds)}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 3fr) minmax(0, 5fr) minmax(0, 3fr)", gap: 16 }}>
            {before ? (
              <LineChart x={doc.buffers} title={`before · ratio · b* = ${before.b}`} height={H}
                lines={[{ y: before.rule_curve, color: BEFORE, width: 1.8 }]} mark={before.b} dot={{ line: 0, i: before.idx }} left="b = 0" right={String(L)} />
            ) : <span />}
            <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
              <figcaption className="muted" style={{ fontSize: 12 }}>score · label · buffers</figcaption>
              <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" style={{ display: "block", background: "var(--subtle)", borderRadius: 4 }} role="img" aria-label="The label, the score block and the selected buffers">
                <rect x={x(a.start)} y={0} width={x(a.end + 1) - x(a.start)} height={H} fill="var(--label)" opacity={0.45} />
                {a.b_left > 0 && <rect x={x(a.start - a.b_left)} y={0} width={x(a.start) - x(a.start - a.b_left)} height={H} fill={BEFORE} opacity={0.35} />}
                {a.b_right > 0 && <rect x={x(a.end + 1)} y={0} width={x(a.end + 1 + a.b_right) - x(a.end + 1)} height={H} fill={AFTER} opacity={0.35} />}
                <polyline points={doc.score.map((v, i) => `${x(i + 0.5)},${H - 12 - v * (H - 30)}`).join(" ")} fill="none" stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                <text x={6} y={H - 4} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">0</text>
                <text x={W - 6} y={H - 4} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="end">{T}</text>
                <text x={x(a.start) + 4} y={12} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">label {a.start} to {a.end + 1}</text>
              </svg>
            </figure>
            {after ? (
              <LineChart x={doc.buffers} title={`after · ratio · b* = ${after.b}`} height={H}
                lines={[{ y: after.rule_curve, color: AFTER, width: 1.8 }]} mark={after.b} dot={{ line: 0, i: after.idx }} left="b = 0" right={String(L)} />
            ) : <span />}
          </div>
        </>
      )}
    </section>
  );
}
