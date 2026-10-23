"use client";
// The toy cases on the Intro: one anomaly, a block detector, the buffer
// chosen as you click.

import { useEffect, useState } from "react";
import { loadToy } from "@/lib/api";
import { TYPE_COLOR, f3, fsec, type ToyDoc } from "@/lib/model";
import { LineChart } from "./LineChart";
import { useRemembered } from "@/lib/remember";

const FAMILIES: [string, string[]][] = [
  ["A · same width, shifted", ["A1", "A2", "A3", "A4", "A5"]],
  ["B · half width, shifted", ["B1", "B2", "B3", "B4", "B5"]],
  ["C · wider, shifted", ["C1", "C2", "C3", "C4", "C5"]],
  ["D · inside plus a second block", ["D1-1", "D1-2", "D1-3", "D1-4", "D2-1", "D2-2", "D2-3", "D2-4"]],
  ["E · two blocks, both outside", ["E1"]],
];

export function Toy() {
  const [c, setC] = useRemembered("toy:case", "A4");
  const [rule, setRule] = useRemembered<"paper" | "ratio">("toy:rule", "paper");
  const [sides, setSides] = useRemembered<"symmetric" | "per_side">("toy:sides", "symmetric");
  const [doc, setDoc] = useState<ToyDoc | null>(null);
  useEffect(() => { let live = true; loadToy(c, 20, rule, sides).then((d) => { if (live) setDoc(d); }); return () => { live = false; }; }, [c, rule, sides]);
  const T = 160, W = 520, H = 150;
  const x = (p: number) => (p / T) * W;
  const a = doc?.anomaly;
  const sideKeys = doc ? Object.keys(doc.sides) : [];
  return (
    <section className="panel" aria-label="How the buffer is chosen">
      <div className="panel-head">
        <h2>How the buffer is chosen, on 24 toy cases</h2>
        <span className="muted" style={{ fontSize: 12 }}>one anomaly of 20 points, a block detector · L = 20</span>
        <span className="seg" role="group" aria-label="Selection rule" style={{ marginLeft: "auto", width: 200 }}>
          <button type="button" aria-pressed={rule === "paper"} onClick={() => setRule("paper")} style={{ height: 28 }}>Paper rule</button>
          <button type="button" aria-pressed={rule === "ratio"} onClick={() => setRule("ratio")} style={{ height: 28 }}>Ratio rule</button>
        </span>
        <span className="seg" role="group" aria-label="Buffer sides" style={{ width: 200 }}>
          <button type="button" aria-pressed={sides === "symmetric"} onClick={() => setSides("symmetric")} style={{ height: 28 }}>Symmetric</button>
          <button type="button" aria-pressed={sides === "per_side"} onClick={() => setSides("per_side")} style={{ height: 28 }}>Per side</button>
        </span>
      </div>
      {FAMILIES.map(([name, cases]) => (
        <div key={name} style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", fontSize: 12 }}>
          <span className="muted" style={{ width: 200 }}>{name}</span>
          {cases.map((k) => <button key={k} type="button" className="chip-btn" aria-pressed={k === c} onClick={() => setC(k)}>{k}</button>)}
        </div>
      ))}
      {doc && a && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 5fr) minmax(0, 3fr) minmax(0, 3fr)", gap: 16, marginTop: 4 }}>
          <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
            <figcaption className="muted" style={{ fontSize: 12 }}>
              <b style={{ color: "var(--ink)" }}>{doc.case}</b> · <span className="mono" style={{ color: TYPE_COLOR[a.type], fontWeight: 600 }}>{a.type}</span> · b* = ({a.b_left}, {a.b_right}) · AutoVUS {f3(doc.autovus)} · VUS-PR {f3(doc.vus_full)} · {fsec(doc.seconds)}
            </figcaption>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" style={{ display: "block", background: "var(--subtle)", borderRadius: 4 }} role="img" aria-label="The label, the score block and the selected buffer">
              <rect x={x(a.start)} y={0} width={x(a.end + 1) - x(a.start)} height={H} fill="var(--label)" opacity={0.45} />
              {a.b_left > 0 && <rect x={x(a.start - a.b_left)} y={0} width={x(a.start) - x(a.start - a.b_left)} height={H} fill={TYPE_COLOR[a.type]} opacity={0.35} />}
              {a.b_right > 0 && <rect x={x(a.end + 1)} y={0} width={x(a.end + 1 + a.b_right) - x(a.end + 1)} height={H} fill={TYPE_COLOR[a.type]} opacity={0.35} />}
              <polyline points={doc.score.map((v, i) => `${x(i + 0.5)},${H - 12 - v * (H - 30)}`).join(" ")} fill="none" stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
              <text x={6} y={H - 4} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">0</text>
              <text x={W - 6} y={H - 4} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="end">{T}</text>
              <text x={x(a.start) + 4} y={12} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)">label {a.start} to {a.end + 1}</text>
            </svg>
          </figure>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {sideKeys.map((s) => <LineChart key={s} x={doc.buffers} title={`(ii) ${s === "both" ? "both sides" : `${s} side`} · detector C(b) in ink, random G(b) in grey`} lines={[{ y: doc.sides[s].random_curve, color: "var(--missed)" }, { y: doc.sides[s].vus_curve, color: "var(--ink)" }]} mark={doc.sides[s].b} left="b = 0" right="20" />)}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {sideKeys.map((s) => <LineChart key={s} x={doc.buffers} title={`(iii) ${s === "both" ? "both sides" : `${s} side`} · the rule, first argmax is b* = ${doc.sides[s].b}`} lines={[{ y: doc.sides[s].rule_curve, color: "var(--accent)", width: 1.8 }]} mark={doc.sides[s].b} dot={{ line: 0, i: doc.sides[s].idx }} left="b = 0" right="20" />)}
          </div>
        </div>
      )}
    </section>
  );
}
