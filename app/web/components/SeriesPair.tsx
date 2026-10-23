"use client";
// One series with two detectors under it: the value lane with the labelled
// anomalies, one score lane per detector with both metrics, and a link to
// the Series page. Used by Rankings (winner and mover) and Cases (the picked case).

import Link from "next/link";
import { useEffect, useState } from "react";
import { loadScoreDoc, loadSeriesDoc } from "@/lib/api";
import { f2, fmt, href, metricLabel, type MetricKey, type PairRow, type ScoreDoc, type SeriesDoc, type State } from "@/lib/model";
import { Lane, type Band } from "./Lane";

export function SeriesPair({ state, file, a, b, move, am, bm, rows }: { state: State; file: string; a: string; b: string; move: number; am: MetricKey; bm: MetricKey; rows: PairRow[] }) {
  const [doc, setDoc] = useState<SeriesDoc | null>(null);
  const [lanes, setLanes] = useState<Record<string, ScoreDoc>>({});
  useEffect(() => {
    let live = true;
    setDoc(null); setLanes({});
    loadSeriesDoc(state.benchmark, file).then((d) => { if (live) setDoc(d); });
    for (const det of Array.from(new Set([a, b]))) {
      loadScoreDoc(state.benchmark, file, det).then((score) => {
        if (live && score) setLanes((m) => ({ ...m, [det]: score }));
      });
    }
    return () => { live = false; };
  }, [state.benchmark, file, a, b]);
  const labels: Band[] = (doc?.labels ?? []).map(([s, e], i) => ({ from: s, to: e + 1, color: "var(--label)", opacity: 0.45, key: `l${i}` }));
  const rowOf = (det: string) => rows.find((r) => r.file === file && r.detector === det);
  return (
    <section className="panel" aria-label="The swap on the chosen series">
      <div className="lane-row"><div className="lane-name"><span className="mono" style={{ fontWeight: 600, fontSize: 11, wordBreak: "break-all" }}>{file.replace(/\.csv$/, "")}</span>{doc && <span className="muted" style={{ fontSize: 11 }}>{fmt(doc.n)} points · {doc.labels.length} anomalies</span>}</div>
        {doc ? <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} height={100} bands={labels} /> : <div style={{ height: 100, background: "var(--subtle)", borderRadius: 4 }} />}</div>
      {Array.from(new Set([a, b])).map((det) => {
        const l = lanes[det], r = rowOf(det);
        return (
          <div key={det} className="lane-row">
            <div className="lane-name">
              <span style={{ fontWeight: 600, color: det === b && move ? (move < 0 ? "var(--up)" : "var(--down)") : undefined }}>{det}</span>
              {r ? <><span className="mono muted" style={{ fontSize: 11 }}>{metricLabel(am)} {f2(r[am])}</span><span className="mono muted" style={{ fontSize: 11 }}>{metricLabel(bm)} {f2(r[bm])}</span></> : <span className="mono muted" style={{ fontSize: 11 }}>…</span>}
            </div>
            {l ? <Lane lo={l.lo} hi={l.hi} bins={l.bins} n={l.n} ymin={0} ymax={1} bands={labels} /> : <div style={{ height: 56, background: "var(--subtle)", borderRadius: 4 }} />}
          </div>
        );
      })}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="muted" style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ width: 10, height: 8, background: "var(--label)" }} />labelled anomaly</span>
        <Link href={href("/series", state, { s: file, pin: Array.from(new Set([a, b])).join(","), top: "0" })} style={{ fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Open in Series with both pinned</Link>
      </div>
    </section>
  );
}
