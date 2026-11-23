import Link from "next/link";
import { TopBar } from "@/components/TopBar";
import { Toy } from "@/components/Toy";
import { SpeedBox, type SpeedTable } from "@/components/SpeedBox";
import { loadLocalTable, loadSummary } from "@/lib/api";
import { fmt, href, stateFrom } from "@/lib/model";

// The authors and affiliations of the AutoVUS paper.
const AUTHORS: [string, string][] = [["Emmanouil Sylligardos", "1"], ["Mingyi Huang", "2"], ["John Paparrizos", "2,3"], ["Pierre Senellart", "1"], ["Paul Boniol", "1"]];
const AFFILIATIONS: [string, string][] = [["1", "DI ENS, ENS, PSL Univ., CNRS, Inria, Paris, France"], ["2", "The Ohio State Univ., Columbus, OH, USA"], ["3", "Aristotle Univ. Thessaloniki, Thessaloniki, Greece"]];

export default async function Intro({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  const summary = await loadSummary(state.benchmark);
  const speed = await loadLocalTable<SpeedTable>(state.benchmark, "speed");
  // The four questions of the demo, in the order of the top bar.
  const cards = [
    { n: "01", q: "Why should the buffer follow the detection?", title: "Play", text: "One anomaly, one detection. Set the buffer it deserves on each side, then see the one AutoVUS picks.", cta: "Play 6 rounds", to: href("/play", state) },
    { n: "02", q: "How does AutoVUS find the lag?", title: "Explore", text: "One series, every detector as a lane, a score on every anomaly. Click one to see why its buffer was chosen.", cta: "Open a series", to: href("/local", state) },
    { n: "03", q: "What changes against current metrics?", title: "Rankings", text: "Two metrics side by side on the whole benchmark, every rank move drawn between them.", cta: "AUC-PR to AutoVUS", to: href("/rankings", state) },
    { n: "04", q: "Where can this go?", title: "Coverage", text: "AutoVUS per anomaly: the best detector changes from one anomaly to the next. What a choice per anomaly would gain.", cta: "Best detector per anomaly", to: href("/coverage", state) },
  ];
  return (
    <>
      <TopBar state={state} current="/" />
      <main className="page" style={{ flexDirection: "column", flexWrap: "nowrap", gap: 28, paddingTop: 36 }}>
        <section style={{ display: "flex", flexWrap: "wrap", gap: 32, alignItems: "flex-end", justifyContent: "space-between" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: "46em" }}>
            <span className="eyebrow">ICDE 2027 demo</span>
            <h1 style={{ fontSize: 34, fontWeight: 700, letterSpacing: "-0.015em", lineHeight: 1.15 }}>A buffer for every anomaly, chosen by the detector&apos;s own lag.</h1>
            <p className="muted" style={{ margin: 0, fontSize: 15 }}>VUS scores a detector with one buffer around every labelled anomaly. AutoVUS picks the buffer per anomaly, as wide as the detection needs, and charges the detector for the width. FastVUS makes that affordable on a whole benchmark.</p>
            <p style={{ margin: "4px 0 0", fontSize: 13 }}>
              {AUTHORS.map(([name, marks], k) => <span key={name}>{k > 0 && ", "}{name}<sup className="muted">{marks}</sup></span>)}
            </p>
            <p className="muted" style={{ margin: 0, fontSize: 12 }}>
              {AFFILIATIONS.map(([mark, place], k) => <span key={mark}>{k > 0 && " · "}<sup>{mark}</sup>{place}</span>)}
            </p>
          </div>
          {summary ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 130px)", gap: 10 }}>
              <div className="tile"><span className="n">{fmt(summary.n_series)}</span><span className="l">series</span></div>
              <div className="tile"><span className="n">{fmt(summary.n_datasets)}</span><span className="l">datasets</span></div>
              <div className="tile"><span className="n">{summary.n_anomalies === null ? "…" : fmt(summary.n_anomalies)}</span><span className="l">anomalies</span></div>
              <div className="tile"><span className="n">{fmt(summary.n_detectors)}</span><span className="l">detectors</span></div>
            </div>
          ) : (
            <div className="panel"><span className="muted">The Python side is not running. See app/README.md.</span></div>
          )}
        </section>
        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }} aria-label="Pages">
          {cards.map((c) => (
            <Link key={c.n} href={c.to} className="card">
              <span className="mono muted" style={{ fontSize: 11 }}>{c.n} · {c.title}</span>
              <span style={{ fontSize: 16, fontWeight: 700, lineHeight: 1.25 }}>{c.q}</span>
              <span className="muted" style={{ fontSize: 13 }}>{c.text}</span>
              <span style={{ fontSize: 13, fontWeight: 600, color: "var(--accent)", marginTop: "auto" }}>{c.cta}</span>
            </Link>
          ))}
        </section>
        {speed && <SpeedBox state={state} speed={speed} />}
        {summary && <Toy />}
        <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }} aria-label="Vocabulary">
          <div className="panel" style={{ gap: 6 }}><span className="eyebrow">VUS</span><span className="muted" style={{ fontSize: 13 }}>The volume under the surface of the PR or ROC curve over every buffer from 0 to L. One buffer grid for every anomaly of the series.</span></div>
          <div className="panel" style={{ gap: 6 }}><span className="eyebrow">FastVUS, the engine</span><span className="muted" style={{ fontSize: 13 }}>The same volume, computed once for all buffers at once. Exact on the CPU. On the GPU every term is exact except the existence term, which is approximated.</span></div>
          <div className="panel panel-accent" style={{ gap: 6 }}><span className="eyebrow" style={{ color: "var(--accent)" }}>AutoVUS, the metric</span><span className="muted" style={{ fontSize: 13 }}>For every anomaly, one buffer for both sides, where the detector&apos;s gain over a random score peaks. The score is chance-adjusted, so a wide buffer costs.</span></div>
        </section>
      </main>
      <footer className="muted" style={{ padding: "16px 24px", borderTop: "1px solid var(--border)", fontSize: 12, display: "flex", gap: 16 }}><span>[Code]</span><span>[Data]</span><span>[Paper]</span></footer>
    </>
  );
}
