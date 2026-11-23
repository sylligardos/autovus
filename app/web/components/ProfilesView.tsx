"use client";
// Profiles: the anomalies of one dataset as a graph (app/export_profiles.py).
// A point is an anomaly, described by its local AutoVUS under every detector;
// each is linked to the 10 anomalies with the closest scores, and the profiles
// are the Louvain communities of that graph. The points sit where the force
// layout of the graph puts them. A switch turns the map into the series: a
// series is a point painted with the profiles of its anomalies, linked to the
// series with the closest mix of profiles. The right column lists the profiles, or shows the picked profile
// (its mean score per detector and its anomalies), or the clicked anomaly (its scores and its
// neighbours). The layout follows Pyxida's graph page.

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadProfiles, loadProfilesIndex, loadSeriesDoc } from "@/lib/api";
import { href, type SeriesDoc, type State } from "@/lib/model";
import { GRADIENT_CSS, chipColors, gradient, seriesName, type ProfilesIndex, type ProfilesTable } from "@/lib/local";
import { useRemembered } from "@/lib/remember";
import { Lane, type Zone } from "./Lane";

const W = 1000, H = 700, PAD = 24, PAGE = 6;
// 20 colours, the most separated ones first (greedy farthest point in OKLab, lightness 0.50 to 0.70 so they read on
// both themes). 20 colours cannot all be told apart by colour alone: the letter of the profile is the second cue.
const COLORS = ["#2f5bc2", "#22bf12", "#fd1a06", "#e436fe", "#656902", "#3baece", "#a6207d", "#c49548", "#b879b2", "#8b04fb",
  "#aa3606", "#f70c97", "#6b83fb", "#129b6b", "#9052c0", "#0683aa", "#ed7473", "#c84c6b", "#442ffd", "#c006c2"];
const f2 = (v: number) => v.toFixed(2);
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmt = (n: number) => n.toLocaleString("en-US");
const letter = (g: number) => (g < 0 ? "·" : String.fromCharCode(65 + g));
const colorOf = (g: number) => (g < 0 ? "var(--missed)" : COLORS[g % COLORS.length]);

type Colour = "group" | "det";
type Mode = "anomalies" | "series";
type Painted = { file: string; anomalies: { i: number; start: number; end: number; g: number }[]; parts: { g: number; n: number }[] };
type Group = { g: number; members: number[]; centre: number[]; top: number[]; series: number };

// ---- a window of the series around one anomaly, fetched once per tab ----
const windowOf = (start: number, end: number): [number, number] => {
  const pad = Math.max(60, 2 * (end - start + 1));
  return [Math.max(0, start - pad), end + 1 + pad];
};
const cache = new Map<string, Promise<SeriesDoc | null>>();
function Snip({ b, file, start, end, height }: { b: string; file: string; start: number; end: number; height: number }) {
  const [lo, hi] = windowOf(start, end);
  const key = `${b}|${file}|${lo}|${hi}`;
  const [got, setGot] = useState<{ key: string; doc: SeriesDoc | null } | null>(null);
  useEffect(() => {
    let live = true;
    let p = cache.get(key);
    if (!p) { p = loadSeriesDoc(b, file, 400, [lo, hi]); cache.set(key, p); }
    p.then((d) => { if (live) setGot({ key, doc: d }); });
    return () => { live = false; };
  }, [key, b, file, lo, hi]);
  const doc = got?.key === key ? got.doc : null;
  if (!doc) return <div style={{ height, background: "var(--subtle)", borderRadius: 4 }} />;
  const s = doc.start ?? 0, e = doc.end ?? doc.n;
  const bands = doc.labels.filter(([x, y]) => y >= s && x < e)
    .map(([x, y]) => ({ from: Math.max(x, s), to: Math.min(y + 1, e), color: "var(--label)", opacity: x === start ? 0.6 : 0.18, key: `l${x}` }));
  return <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} start={s} end={e} view={[s, e]} height={height} bands={bands} />;
}

// One bar per detector, highest first, in two columns. A tick marks a second value (the mean of the profile).
function Bars({ names, values, marks }: { names: string[]; values: number[]; marks?: number[] }) {
  const order = values.map((_, d) => d).sort((a, b) => values[b] - values[a]);
  return (
    <div style={{ columns: 2, columnGap: 16 }}>
      {order.map((d) => (
        <div key={d} style={{ display: "flex", alignItems: "center", gap: 6, width: "100%", height: 17, breakInside: "avoid" }}>
          <span className="mono" style={{ flex: "0 0 96px", fontSize: 10.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{names[d]}</span>
          <span style={{ position: "relative", flex: "1 1 auto", height: 8, background: "var(--subtle)", borderRadius: 2 }}>
            <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${Math.max(0, Math.min(1, values[d])) * 100}%`, background: gradient(values[d]), borderRadius: 2 }} />
            {marks && <span style={{ position: "absolute", top: -2, bottom: -2, left: `${Math.max(0, Math.min(1, marks[d])) * 100}%`, width: 2, background: "var(--ink)" }} />}
          </span>
          <span className="mono" style={{ flex: "0 0 28px", fontSize: 10.5, textAlign: "right" }}>{f2(values[d])}</span>
        </div>
      ))}
    </div>
  );
}

const Badge = ({ g }: { g: number }) => <span className="pf-badge" style={{ background: colorOf(g) }}>{letter(g)}</span>;
const Chip = ({ v, text }: { v: number; text: string }) => {
  const c = chipColors(v);
  return <span className="chip mono" style={{ background: c.bg, color: c.fg }}>{text}</span>;
};
const Close = ({ onClick }: { onClick: () => void }) => <button type="button" className="icon-btn" aria-label="Close" style={{ marginLeft: "auto", flex: "0 0 auto" }} onClick={onClick}>×</button>;
const Label = ({ children }: { children: React.ReactNode }) => <span className="eyebrow">{children}</span>;

const secs = (v: number | null) => (v === null ? "" : v < 0.01 ? "<0.01" : v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : fmt(Math.round(v)));
const Meter = ({ v }: { v: number }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
    <span style={{ position: "relative", width: 90, height: 8, background: "var(--subtle)", borderRadius: 2 }}>
      <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${Math.max(0, Math.min(1, v)) * 100}%`, background: gradient(v), borderRadius: 2 }} />
    </span>
    <span className="mono">{f2(v)}</span>
  </span>
);

// The 3 detectors that cover the dataset (the greedy set by the best score per anomaly), each with its time and
// the most similar detector that is faster. A click on a detector colours the map by its score.
function Cover({ cover, detectors, picked, onPick }: { cover: ProfilesTable["cover"]; detectors: number; picked: string | null; onPick: (detector: string) => void }) {
  const plain = { cursor: "default" } as const;
  const name = (d: string) => (
    <button type="button" className="chip-btn" aria-pressed={picked === d} title={`Colour the map by ${d}`} onClick={() => onPick(d)}>{d}</button>
  );
  return (
    <section className="panel" aria-label="Detectors that cover the dataset">
      <h2>This dataset is covered by</h2>
      <table className="table">
        <thead><tr>
          <th>Detector</th><th>Best per anomaly</th><th className="r">Seconds per series</th>
          <th style={{ paddingLeft: 40 }}>Similar and faster</th><th className="r">Correlation</th><th className="r">Seconds per series</th>
        </tr></thead>
        <tbody>
          {cover.rows.map((x, n) => (
            <tr key={x.detector} style={plain}>
              <td>{n > 0 && <span className="mono muted" style={{ marginRight: 6 }}>+</span>}{name(x.detector)}</td>
              <td><Meter v={x.best} /></td><td className="r mono">{secs(x.seconds)}</td>
              <td style={{ paddingLeft: 40 }}>{x.similar && name(x.similar.detector)}</td>
              <td className="r mono">{x.similar && f2(x.similar.corr)}</td><td className="r mono">{x.similar && secs(x.similar.seconds)}</td>
            </tr>
          ))}
          <tr style={plain}><td className="mono muted">all {detectors} detectors</td><td><Meter v={cover.all} /></td><td /><td /><td /><td /></tr>
        </tbody>
      </table>
    </section>
  );
}

// One series with every anomaly coloured by its profile. Fetched when it scrolls into the page.
function PaintedSeries({ b, file, anomalies, lit, sel, onPick, onName, height = 56 }: {
  b: string; file: string; anomalies: { i: number; start: number; end: number; g: number }[]; lit: number | null; sel: number | null;
  onPick: (i: number) => void; onName?: () => void; height?: number;
}) {
  const key = `${b}|${file}|all`;
  const [got, setGot] = useState<{ key: string; doc: SeriesDoc | null } | null>(null);
  useEffect(() => {
    let live = true;
    let p = cache.get(key);
    if (!p) { p = loadSeriesDoc(b, file, 1200); cache.set(key, p); }
    p.then((d) => { if (live) setGot({ key, doc: d }); });
    return () => { live = false; };
  }, [key, b, file]);
  const doc = got?.key === key ? got.doc : null;
  const profiles = Array.from(new Set(anomalies.map((x) => x.g))).sort((p, q) => (p < 0 ? 1 : q < 0 ? -1 : p - q));
  const zones: Zone[] = anomalies.map((x) => ({ from: x.start, to: x.end + 1, color: colorOf(x.g), key: `a${x.i}`, selected: x.i === sel, onClick: () => onPick(x.i) }));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
        {onName
          ? <button type="button" className="pf-link mono" style={{ fontSize: 11, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} onClick={onName}>{seriesName(file)}</button>
          : <span className="mono" style={{ fontSize: 11, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{seriesName(file)}</span>}
        <span className="mono muted" style={{ fontSize: 11, whiteSpace: "nowrap" }}>{fmt(anomalies.length)} anomalies · {profiles.length} {profiles.length === 1 ? "profile" : "profiles"}</span>
        <span style={{ marginLeft: "auto", display: "inline-flex", gap: 3, flex: "0 0 auto" }}>
          {profiles.map((p) => <span key={p} className="pf-badge" style={{ background: colorOf(p), width: 18, height: 18, fontSize: 10, borderRadius: 4, opacity: lit === null || lit === p ? 1 : 0.25 }}>{letter(p)}</span>)}
        </span>
      </div>
      {doc
        ? <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} height={height} zones={zones}
            bands={anomalies.map((x) => ({ from: x.start, to: x.end + 1, color: colorOf(x.g), opacity: lit === null || lit === x.g ? 0.8 : 0.12, key: `b${x.i}` }))} />
        : <div style={{ height, background: "var(--subtle)", borderRadius: 4 }} />}
    </div>
  );
}

// A series as a point: one slice per profile, sized by the number of its anomalies in the profile.
function Pie({ parts, r, flat }: { parts: { g: number; n: number }[]; r: number; flat?: string }) {
  if (flat || parts.length === 1) return <circle r={r} fill={flat ?? colorOf(parts[0].g)} />;
  const total = parts.reduce((s, p) => s + p.n, 0);
  const ends = parts.map((_, k) => (parts.slice(0, k + 1).reduce((s, p) => s + p.n, 0) / total) * 2 * Math.PI - Math.PI / 2);
  const at = (a: number) => `${(r * Math.cos(a)).toFixed(2)},${(r * Math.sin(a)).toFixed(2)}`;
  return <>{parts.map((p, k) => {
    const a0 = k ? ends[k - 1] : -Math.PI / 2, a1 = ends[k];
    return <path key={p.g} d={`M0,0 L${at(a0)} A${r},${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${at(a1)} Z`} fill={colorOf(p.g)} />;
  })}</>;
}

const LINE = "Which kinds of anomalies does a dataset hold, and which detectors cover them?";
function DatasetSelect({ index, dataset, setDataset }: { index: ProfilesIndex; dataset: string; setDataset: (d: string) => void }) {
  return (
    <label className="field"><span>Dataset</span>
      <select className="select mono" value={dataset} onChange={(e) => setDataset(e.target.value)}>
        {index.map((x) => <option key={x.dataset} value={x.dataset}>{x.dataset} · {fmt(x.anomalies)} anomalies{x.too_many ? "" : ` · ${x.profiles} profiles`}</option>)}
      </select>
    </label>
  );
}

// The page: the datasets that were exported, and the map of the chosen one.
export function ProfilesView({ state }: { state: State }) {
  const b = state.benchmark;
  const [index, setIndex] = useState<ProfilesIndex | null | undefined>(undefined);
  const [dataset, setDataset] = useRemembered<string | null>(`profiles:${b}:dataset`, null);
  useEffect(() => {
    let live = true;
    loadProfilesIndex<ProfilesIndex>(b).then((x) => { if (live) setIndex(x); });
    return () => { live = false; };
  }, [b]);
  if (index === undefined) return <p className="muted">Loading…</p>;
  if (index === null || !index.length) return <p className="muted">Not exported. Run <code className="mono">app/export_profiles.py</code>.</p>;
  const current = index.some((x) => x.dataset === dataset) ? dataset! : index.find((x) => x.dataset === "YAHOO")?.dataset ?? index[0].dataset;
  const entry = index.find((x) => x.dataset === current)!;
  if (entry.too_many) return (
    <main className="content">
      <p className="page-line">{LINE}</p>
      <section className="panel" style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "flex-end", gap: "12px 24px" }} aria-label="Controls">
        <DatasetSelect index={index} dataset={current} setDataset={setDataset} />
      </section>
      <section className="panel" role="alert" style={{ borderColor: "var(--down)" }}>
        <span style={{ fontWeight: 600 }}>Too many anomalies</span>
        <span className="mono muted" style={{ fontSize: 12 }}>{current} holds {fmt(entry.anomalies)} anomalies in {entry.series} series.</span>
      </section>
    </main>
  );
  return <ProfilesMap key={`${b}:${current}`} state={state} index={index} dataset={current} setDataset={setDataset} />;
}

function ProfilesMap({ state, index, dataset, setDataset }: { state: State; index: ProfilesIndex; dataset: string; setDataset: (d: string) => void }) {
  const b = `${state.benchmark}`;
  const [t, setT] = useState<ProfilesTable | null | undefined>(undefined);
  const [colour, setColour] = useRemembered<Colour>(`profiles:${b}:${dataset}:colour`, "group");
  const [detName, setDetName] = useRemembered<string | null>(`profiles:${b}:${dataset}:det`, null);
  const [sel, setSel] = useRemembered<number | null>(`profiles:${b}:${dataset}:sel`, null);
  const [picked, setPicked] = useRemembered<number | null>(`profiles:${b}:${dataset}:picked`, null);
  const [view, setView] = useRemembered<"nb" | "group">(`profiles:${b}:${dataset}:view`, "nb");
  const [mode, setMode] = useRemembered<Mode>(`profiles:${b}:${dataset}:mode`, "anomalies");
  const [selS, setSelS] = useRemembered<number | null>(`profiles:${b}:${dataset}:selS`, null);
  const [page, setPage] = useState(0);
  const [hoverG, setHoverG] = useState<number | null>(null);
  const [tf, setTf] = useState({ s: 1, tx: 0, ty: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null);
  const dragged = useRef(false);

  useEffect(() => {
    let live = true;
    loadProfiles<ProfilesTable>(b, dataset).then((x) => { if (live) setT(x); });
    return () => { live = false; };
  }, [b, dataset]);

  const zoomBy = useCallback((f: number, mx = W / 2, my = H / 2) => setTf((v) => {
    const s = Math.min(40, Math.max(0.5, v.s * f)), k = s / v.s;
    return { s, tx: mx - (mx - v.tx) * k, ty: my - (my - v.ty) * k };
  }), []);
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const k = Math.min(r.width / W, r.height / H), ox = (r.width - W * k) / 2, oy = (r.height - H * k) / 2;
      zoomBy(e.deltaY > 0 ? 1 / 1.15 : 1.15, (e.clientX - r.left - ox) / k, (e.clientY - r.top - oy) / k);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [t, zoomBy, mode]);

  const g = t?.graph ?? null;
  const X = useMemo(() => (t ? t.X.map((r) => r.map((v) => v / 100)) : []), [t]);
  const best = useMemo(() => X.map((r) => Math.max(...r)), [X]);
  const label = useMemo(() => g?.group ?? [], [g]);
  const det = t && detName ? t.detectors.indexOf(detName) : -1;          // the detector the map is coloured by
  const tint: Colour = colour === "det" && det < 0 ? "group" : colour;

  const groups: Group[] = useMemo(() => {
    if (!t) return [];
    const m = new Map<number, number[]>();
    label.forEach((v, i) => { if (!m.has(v)) m.set(v, []); m.get(v)!.push(i); });
    return Array.from(m.entries()).sort((p, q) => (p[0] < 0 ? 1 : q[0] < 0 ? -1 : p[0] - q[0])).map(([gid, members]) => {
      const centre = t.detectors.map((_, d) => members.reduce((s, i) => s + X[i][d], 0) / members.length);
      return {
        g: gid, members, centre, top: centre.map((_, d) => d).sort((p, q) => centre[q] - centre[p]).slice(0, 3),
        series: new Set(members.map((i) => t.s[i])).size,
      };
    });
  }, [t, label, X]);
  const groupOf = useMemo(() => new Map(groups.map((x) => [x.g, x])), [groups]);
  // The series: their anomalies with the profiles, and the number of anomalies per profile.
  const painted: Painted[] = useMemo(() => {
    if (!t) return [];
    const rows: Painted[] = t.series.map((file) => ({ file, anomalies: [], parts: [] }));
    t.s.forEach((si, i) => rows[si].anomalies.push({ i, start: t.start[i], end: t.end[i], g: label[i] }));
    for (const r of rows) {
      const m = new Map<number, number>();
      for (const x of r.anomalies) m.set(x.g, (m.get(x.g) ?? 0) + 1);
      r.parts = Array.from(m.entries()).sort((p, q) => (p[0] < 0 ? 1 : q[0] < 0 ? -1 : p[0] - q[0])).map(([gid, n]) => ({ g: gid, n }));
    }
    return rows;
  }, [t, label]);
  const sg = t?.series_graph ?? null;
  const posS = useMemo(() => (sg ? sg.pos.map(([x, y]) => [2 * PAD + x * (W - 4 * PAD), 2 * PAD + (1 - y) * (H - 4 * PAD)] as const) : []), [sg]);
  const adjS = useMemo(() => {
    const a: { j: number; w: number }[][] = posS.map(() => []);
    for (const [i, j, w] of sg?.edges ?? []) { a[i].push({ j, w }); a[j].push({ j: i, w }); }
    for (const l of a) l.sort((p, q) => q.w - p.w);
    return a;
  }, [sg, posS]);
  const maxK = useMemo(() => Math.max(1, ...painted.map((x) => x.anomalies.length)), [painted]);

  const pos = useMemo(() => (g ? g.pos.map(([x, y]) => [PAD + x * (W - 2 * PAD), PAD + (1 - y) * (H - 2 * PAD)] as const) : []), [g]);
  const adj = useMemo(() => {
    const a: { j: number; w: number }[][] = pos.map(() => []);
    for (const [i, j, w] of g?.edges ?? []) { a[i].push({ j, w }); a[j].push({ j: i, w }); }
    for (const l of a) l.sort((p, q) => q.w - p.w);
    return a;
  }, [g, pos]);
  const maxW = useMemo(() => Math.max(1e-9, ...(g?.edges.map((e) => e[2]) ?? [])), [g]);

  const name = useCallback((i: number) => (t ? `${seriesName(t.series[t.s[i]])} #${t.a[i] + 1}` : ""), [t]);
  const fill = useCallback((i: number) => (tint === "group" ? colorOf(label[i]) : gradient(X[i][det])), [tint, label, X, det]);
  const lit = sel === null ? picked ?? hoverG : null;
  const near = useMemo(() => new Set(sel !== null ? adj[sel]?.map((e) => e.j) : []), [adj, sel]);

  const edgeLayer = useMemo(() => g && (
    <g>
      {g.edges.map(([a, c, w], n) => {
        const on = sel !== null && (a === sel || c === sel);
        let o = 0.06 + 0.25 * (w / maxW);
        if (lit !== null) o = label[a] === lit && label[c] === lit ? o : 0.02;
        if (sel !== null) o = on ? 0.9 : 0.03;
        return <line key={n} className={on ? "pf-edge lit" : "pf-edge"} x1={pos[a][0]} y1={pos[a][1]} x2={pos[c][0]} y2={pos[c][1]} strokeOpacity={o.toFixed(2)} />;
      })}
    </g>
  ), [g, pos, sel, lit, label, maxW]);

  const pick = useCallback((i: number | null) => { setSel(i); setPage(0); if (i !== null) setMode("anomalies"); }, [setSel, setMode]);
  const nodeLayer = useMemo(() => t && (
    <g>
      {pos.map(([x, y], i) => {
        const isSel = i === sel, isNear = near.has(i);
        let o = 1;
        if (lit !== null) o = label[i] === lit ? 1 : 0.1;
        if (sel !== null) o = isSel || isNear ? 1 : 0.22;
        return (
          <circle key={i} className={`pf-node${isSel ? " sel" : isNear ? " near" : ""}`} cx={x} cy={y} fill={fill(i)} fillOpacity={o}
            onClick={() => { if (!dragged.current) pick(isSel ? null : i); }}>
            <title>{`${name(i)} · profile ${letter(label[i])} · best ${f2(best[i])}`}</title>
          </circle>
        );
      })}
    </g>
  ), [t, pos, sel, near, lit, label, fill, name, best, pick]);

  const litS = selS === null ? picked ?? hoverG : null;
  const nearS = useMemo(() => new Set(selS !== null ? adjS[selS]?.map((e) => e.j) : []), [adjS, selS]);
  const pickSeries = useCallback((i: number | null) => { setSelS(i); setPage(0); }, [setSelS]);
  const seriesLayer = useMemo(() => sg && mode === "series" && (
    <g>
      {sg.edges.map(([a, c], n) => {
        const on = selS !== null && (a === selS || c === selS);
        return <line key={n} className={on ? "pf-edge lit" : "pf-edge"} x1={posS[a][0]} y1={posS[a][1]} x2={posS[c][0]} y2={posS[c][1]} strokeOpacity={selS !== null ? (on ? 0.9 : 0.04) : 0.14} />;
      })}
      {painted.map((x, i) => {
        const isSel = i === selS, r = 5 + 9 * Math.sqrt(x.anomalies.length / maxK);
        let o = 1;
        if (litS !== null) o = x.parts.some((p) => p.g === litS) ? 1 : 0.12;
        if (selS !== null) o = isSel || nearS.has(i) ? 1 : 0.25;
        return (
          <g key={x.file} className={`pf-pie${isSel ? " sel" : ""}`} opacity={o} style={{ transform: `translate(${posS[i][0]}px, ${posS[i][1]}px) scale(var(--z, 1))` }}
            onClick={() => { if (!dragged.current) pickSeries(isSel ? null : i); }}>
            <Pie parts={x.parts} r={r} flat={tint === "group" ? undefined : gradient(x.anomalies.reduce((sum, an) => sum + X[an.i][det], 0) / Math.max(1, x.anomalies.length))} />
            <circle className="ring" r={r} />
            <title>{`${seriesName(x.file)} · ${x.anomalies.length} anomalies · profiles ${x.parts.map((p) => letter(p.g)).join(" ")}`}</title>
          </g>
        );
      })}
    </g>
  ), [sg, mode, posS, painted, maxK, selS, nearS, litS, tint, X, det, pickSeries]);

  if (t === undefined) return <p className="muted">Loading…</p>;
  if (t === null || !g) return <p className="muted">Not exported. Run <code className="mono">app/export_profiles.py</code>.</p>;

  const clear = () => { setSel(null); setSelS(null); setPicked(null); setHoverG(null); setPage(0); };
  const file = (i: number) => t.series[t.s[i]];
  const bestDet = (i: number) => X[i].indexOf(best[i]);
  const tops = (x: Group) => x.top.map((d) => `${t.detectors[d]} ${f2(x.centre[d])}`).join(" · ");
  const summary = (x: Group) => `${fmt(x.members.length)} anomalies · ${pct(x.members.length / X.length)} · ${x.series} series`;
  const pageOf = (list: number[]) => {
    const pages = Math.max(1, Math.ceil(list.length / PAGE)), p = Math.min(page, pages - 1);
    return { rows: list.slice(p * PAGE, p * PAGE + PAGE), text: `${p * PAGE + 1} to ${Math.min(list.length, p * PAGE + PAGE)} of ${fmt(list.length)}`, p, pages };
  };
  const pager = (x: ReturnType<typeof pageOf>) => x.pages > 1 && (
    <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 6 }}>
      <button type="button" className="icon-btn" aria-label="Previous" disabled={x.p === 0} onClick={() => setPage(x.p - 1)}>‹</button>
      <span className="mono muted" style={{ fontSize: 11 }}>{x.text}</span>
      <button type="button" className="icon-btn" aria-label="Next" disabled={x.p >= x.pages - 1} onClick={() => setPage(x.p + 1)}>›</button>
    </span>
  );
  const lane = (i: number, tag: React.ReactNode) => (
    <div key={i} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
        <button type="button" className="pf-link mono" style={{ fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} onClick={() => pick(i)}>{name(i)}</button>
        <span style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 6, flex: "0 0 auto" }}>{tag}</span>
      </div>
      <Snip b={b} file={file(i)} start={t.start[i]} end={t.end[i]} height={44} />
    </div>
  );
  const bestTag = (i: number) => <><span className="mono muted" style={{ fontSize: 11 }}>{t.detectors[bestDet(i)]}</span><Chip v={best[i]} text={f2(best[i])} /></>;

  let side: React.ReactNode;
  const selGroup = sel !== null ? groupOf.get(label[sel]) : undefined;
  const pickedGroup = picked !== null ? groupOf.get(picked) : undefined;
  if (mode === "series" && selS !== null) {
    const x = painted[selS];
    const p = pageOf(adjS[selS].map((e) => e.j));
    side = (
      <>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Label>Series</Label>
          <Link href={href("/local", state, { s: x.file })} style={{ marginLeft: "auto", fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Open in Explore</Link>
          <Close onClick={clear} />
        </div>
        <PaintedSeries b={b} file={x.file} anomalies={x.anomalies} lit={null} sel={null} onPick={pick} height={110} />
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {x.parts.map((part) => {
            const grp = groupOf.get(part.g);
            return (
              <div key={part.g} style={{ display: "flex", alignItems: "center", gap: 10, padding: "4px 0" }}>
                <Badge g={part.g} />
                <span className="mono" style={{ fontSize: 12, flex: "0 0 96px" }}>{fmt(part.n)} · {pct(part.n / x.anomalies.length)}</span>
                <span className="mono muted" style={{ fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{grp ? tops(grp) : ""}</span>
              </div>
            );
          })}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}><Label>Linked series · {adjS[selS].length}</Label>{pager(p)}</div>
        {p.rows.map((j) => <PaintedSeries key={painted[j].file} b={b} file={painted[j].file} anomalies={painted[j].anomalies} lit={null} sel={null} onPick={pick} onName={() => pickSeries(j)} height={44} />)}
      </>
    );
  } else if (mode === "anomalies" && sel !== null && selGroup) {
    const [lo, hi] = windowOf(t.start[sel], t.end[sel]);
    const top3 = X[sel].map((_, d) => d).sort((p, q) => X[sel][q] - X[sel][p]).slice(0, 3).map((d) => t.detectors[d]);
    const inGroup = view === "group";
    const p = pageOf(inGroup ? selGroup.members.filter((i) => i !== sel) : adj[sel].map((e) => e.j));
    side = (
      <>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Badge g={selGroup.g} />
          <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
            <span className="mono" style={{ fontSize: 12, fontWeight: 600, wordBreak: "break-all" }}>{name(sel)}</span>
            <span className="mono muted" style={{ fontSize: 11 }}>points {fmt(t.start[sel])} to {fmt(t.end[sel])} · best {t.detectors[bestDet(sel)]} {f2(best[sel])}</span>
          </span>
          <Close onClick={clear} />
        </div>
        <Snip b={b} file={file(sel)} start={t.start[sel]} end={t.end[sel]} height={110} />
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Label>Local AutoVUS · tick = mean of profile {letter(selGroup.g)}</Label>
          <Link href={href("/local", state, { s: file(sel), det: top3.join(","), v: `${lo},${hi}`, a: String(t.a[sel]) })} style={{ marginLeft: "auto", fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Open in Explore</Link>
        </div>
        <Bars names={t.detectors} values={X[sel]} marks={selGroup.centre} />
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div className="seg" role="group" aria-label="Anomalies listed below" style={{ flex: "0 0 220px" }}>
            <button type="button" aria-pressed={!inGroup} onClick={() => { setView("nb"); setPage(0); }}>Neighbours</button>
            <button type="button" aria-pressed={inGroup} onClick={() => { setView("group"); setPage(0); }}>Profile {letter(selGroup.g)}</button>
          </div>
          {pager(p)}
        </div>
        {p.rows.map((i) => lane(i, bestTag(i)))}
      </>
    );
  } else if (pickedGroup) {
    const p = pageOf(pickedGroup.members);
    side = (
      <>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Badge g={pickedGroup.g} />
          <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>{summary(pickedGroup)}</span>
          </span>
          <Close onClick={clear} />
        </div>
        <Label>Mean local AutoVUS per detector</Label>
        <Bars names={t.detectors} values={pickedGroup.centre} />
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}><Label>Anomalies</Label>{pager(p)}</div>
        {p.rows.map((i) => lane(i, bestTag(i)))}
      </>
    );
  } else {
    side = (
      <>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <Label>Profiles · {groups.filter((x) => x.g >= 0).length}</Label>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {groups.map((x) => (
            <button key={x.g} type="button" className="pf-row" onClick={() => { setPicked(x.g); setHoverG(null); setPage(0); }}
              onMouseEnter={() => setHoverG(x.g)} onMouseLeave={() => setHoverG((v) => (v === x.g ? null : v))}>
              <Badge g={x.g} />
              <span style={{ display: "flex", flexDirection: "column", minWidth: 0, flex: "1 1 auto" }}>
                <span style={{ fontSize: 13 }}>{summary(x)}</span>
                <span className="mono muted" style={{ fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{tops(x)}</span>
              </span>
            </button>
          ))}
        </div>
      </>
    );
  }

  return (
    <main className="content">
      <p className="page-line">{LINE} <span>{mode === "series"
        ? "Each point is a series, painted with the profiles of its anomalies."
        : "Each point is an anomaly. Points close together are found by the same detectors."}</span></p>
      <section className="panel" style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "flex-end", gap: "12px 24px" }} aria-label="Controls">
        <DatasetSelect index={index} dataset={dataset} setDataset={setDataset} />
        <div className="field"><span>Points</span>
          <div className="seg" role="group" aria-label="Points" style={{ width: 200 }}>
            <button type="button" aria-pressed={mode === "anomalies"} onClick={() => { setMode("anomalies"); setPage(0); setTf({ s: 1, tx: 0, ty: 0 }); }}>Anomalies</button>
            <button type="button" aria-pressed={mode === "series"} onClick={() => { setMode("series"); setPage(0); setTf({ s: 1, tx: 0, ty: 0 }); }}>Series</button>
          </div>
        </div>
      </section>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "stretch" }}>
        <figure className="panel" style={{ flex: "3 1 560px", minWidth: 0, margin: 0, padding: 0, position: "relative", height: H, overflow: "hidden" }}>
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label={mode === "series" ? "Graph of the series" : "Graph of the anomalies"}
            style={{ display: "block", cursor: "grab", touchAction: "none" }}
            onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, tx: tf.tx, ty: tf.ty, moved: false }; }}
            onPointerMove={(e) => {
              const d = drag.current; if (!d) return;
              const r = e.currentTarget.getBoundingClientRect(), s = Math.min(r.width / W, r.height / H);
              if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 3) d.moved = true;
              if (d.moved) setTf((v) => ({ ...v, tx: d.tx + (e.clientX - d.x) / s, ty: d.ty + (e.clientY - d.y) / s }));
            }}
            onPointerUp={() => { dragged.current = drag.current?.moved ?? false; drag.current = null; }}
            onPointerLeave={() => { drag.current = null; }}
            onDoubleClick={() => setTf({ s: 1, tx: 0, ty: 0 })}>
            <g transform={`translate(${tf.tx} ${tf.ty}) scale(${tf.s})`} style={{ "--z": 1 / tf.s } as React.CSSProperties}>
              {mode === "series" ? seriesLayer : <>{edgeLayer}{nodeLayer}</>}
            </g>
          </svg>
          <div className="mono muted" style={{ position: "absolute", left: 14, top: 10, fontSize: 11, pointerEvents: "none" }}>
            {fmt(X.length)} anomalies · {t.series.length} series
          </div>
          {tint !== "group" && (
            <div className="mono muted" style={{ position: "absolute", left: 14, bottom: 12, fontSize: 11, display: "inline-flex", alignItems: "center", gap: 6, pointerEvents: "none" }}>
              {mode === "series" ? `${detName}, mean over the series` : detName} 0<span aria-hidden="true" style={{ width: 90, height: 8, borderRadius: 4, background: GRADIENT_CSS }} />1
            </div>
          )}
          <div style={{ position: "absolute", right: 12, bottom: 12, display: "flex", flexDirection: "column", gap: 6 }}>
            <button type="button" className="icon-btn" aria-label="Zoom in" onClick={() => zoomBy(1.25)}>+</button>
            <button type="button" className="icon-btn" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.25)}>−</button>
          </div>
        </figure>
        <aside className="panel" style={{ flex: "2 1 400px", minWidth: 0, height: H, overflowY: "auto" }} aria-label="Profiles">{side}</aside>
      </div>
      <Cover cover={t.cover} detectors={t.detectors.length} picked={tint === "det" ? detName : null}
        onPick={(d) => { if (tint === "det" && detName === d) setColour("group"); else { setDetName(d); setColour("det"); } }} />
    </main>
  );
}
