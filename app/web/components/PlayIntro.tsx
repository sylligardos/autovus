"use client";
// The Play landing: the two games as moving sketches, each one a link into its
// game, and who has played so far (app/data/games.json through GET /games).
// The sketches loop in CSS (globals.css, "play landing").

import Link from "next/link";
import { useEffect, useState } from "react";
import { loadGames, type Played } from "@/lib/api";
import { TYPE_COLOR, href, type DetectionType, type State } from "@/lib/model";
import { getPlayer, setPlayer } from "@/lib/player";
import { AFTER, BEFORE } from "./Drawer";

const SHOW = 14;                                   // pills in view; the rest is a count
const VERDICTS: DetectionType[] = ["early", "late", "exact", "missed"];

// "4 min", "2 h", "3 d" since an ISO time.
function ago(when: string) {
  const s = Math.max(0, (Date.now() - new Date(when).getTime()) / 1000);
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min`;
  if (s < 86400) return `${Math.round(s / 3600)} h`;
  return `${Math.round(s / 86400)} d`;
}
// One drawing in a few words: "3 labels · 2 early, 1 late".
function drawn(x: Played) {
  const count = new Map<string, number>();
  for (const t of x.types ?? []) count.set(t, (count.get(t) ?? 0) + 1);
  const parts = [...count.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([t, n]) => `${n} ${t}`);
  return `${x.labels ?? 0} label${x.labels === 1 ? "" : "s"}${parts.length ? " · " + parts.join(", ") : ""}`;
}

export function PlayIntro({ state }: { state: State }) {
  const [games, setGames] = useState<Played[] | null>(null);
  const [name, setName] = useState("");
  useEffect(() => {
    let live = true;
    setName(getPlayer());                          // the browser remembers the name between visits
    loadGames().then((d) => { if (live) setGames(d?.games ?? []); });
    return () => { live = false; };
  }, []);
  const players = games ? new Set(games.map((x) => x.name.trim().toLowerCase())).size : 0;

  return (
    <main className="content" style={{ maxWidth: 1160, margin: "0 auto", gap: 40, padding: "16px 0 8px" }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between", gap: "12px 32px" }}>
        <h1 style={{ fontSize: 36, fontWeight: 800, letterSpacing: "-0.025em", lineHeight: 1.08 }}>One detection, one buffer. Go head to head with AutoVUS.</h1>
        <div className="mono" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 18, fontSize: 13, paddingBottom: 6 }} aria-label="The four verdicts">
          {VERDICTS.map((t) => (
            <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 7, color: t === "missed" ? "var(--muted)" : TYPE_COLOR[t] }}><span style={{ width: 10, height: 10, borderRadius: "50%", background: TYPE_COLOR[t] }} />{t}</span>
          ))}
          <span className="muted" style={{ paddingLeft: 10, borderLeft: "1px solid var(--border)" }}>L {state.L} per side</span>
        </div>
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 14, fontWeight: 600, marginTop: -16 }}>
        Your name
        <input value={name} onChange={(e) => { setName(e.target.value); setPlayer(e.target.value); }} maxLength={24} placeholder="for Played so far" autoComplete="nickname"
          style={{ height: 38, width: 220, padding: "0 12px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--panel)", fontSize: 14, fontWeight: 400 }} />
      </label>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))", gap: "40px 56px" }}>
        <Link href={href("/play", state, { mode: "game" })} className="door" aria-label="Find the buffer, six rounds">
          <svg viewBox="0 0 520 180" width="100%" style={{ display: "block", overflow: "visible" }} role="img" aria-label="A labelled anomaly, a detector that fires late, and a buffer edge sliding after the label">
            <rect x={0} y={0} width={520} height={150} rx={10} fill="var(--panel)" stroke="var(--border)" />
            <rect x={196} y={0} width={64} height={150} fill="var(--label)" opacity={0.6} />
            <rect className="play-zone" x={260} y={0} width={74} height={150} fill={AFTER} opacity={0.22} />
            <polyline points="0,52 176,52 188,46 196,30 226,26 260,34 274,50 520,52" fill="none" stroke="var(--ink)" strokeWidth={2} />
            <polyline points="0,124 232,124 240,78 326,74 334,124 520,124" fill="none" stroke="var(--accent)" strokeWidth={2.5} />
            <rect x={195} y={0} width={2} height={150} fill={BEFORE} opacity={0.8} />
            <g className="play-edge">
              <rect x={263} y={0} width={3} height={150} fill={AFTER} />
              <rect x={256} y={152} width={18} height={24} rx={6} fill="var(--panel)" stroke={AFTER} strokeWidth={2} />
              <path d="M261,160 L261,168 M265,160 L265,168 M269,160 L269,168" stroke={AFTER} strokeWidth={1.5} opacity={0.8} />
              <text className="play-gain" x={264} y={-8} textAnchor="middle" fontFamily="var(--mono)" fontSize={13} fontWeight={500} fill={AFTER}>gain</text>
            </g>
          </svg>
          <span className="press" style={{ background: "var(--accent)", boxShadow: "0 4px 0 color-mix(in srgb, var(--accent) 70%, #000), 0 8px 16px rgba(0, 0, 0, 0.16)" }}>
            <span style={{ flex: 1, display: "inline-flex", alignItems: "center", gap: 12, padding: "0 18px" }}>
              <span style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.01em" }}>Find the buffer</span>
              <span className="mono" style={{ display: "inline-flex", alignItems: "center", height: 22, padding: "0 9px", borderRadius: 999, background: "rgba(255, 255, 255, 0.18)", fontSize: 12 }}>6 rounds</span>
            </span>
            <span className="ico" style={{ background: "color-mix(in srgb, var(--accent) 70%, #000)" }}>
              <svg width={22} height={22} viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M7 4v16l13-8z" fill="#FFFFFF" /></svg>
            </span>
          </span>
          <p className="muted" style={{ margin: 0, padding: "0 4px", fontSize: 15 }}>Drag the buffer a detection deserves, then see the one AutoVUS picks and why.</p>
        </Link>

        <Link href={href("/play", state, { mode: "draw" })} className="door" aria-label="Draw your detection, free play">
          <svg viewBox="0 0 520 180" width="100%" style={{ display: "block", overflow: "visible" }} role="img" aria-label="Two labelled anomalies, a score being drawn by hand, and the buffers AutoVUS pops around each label">
            <rect x={0} y={0} width={520} height={150} rx={10} fill="var(--panel)" stroke="var(--border)" />
            <rect x={112} y={0} width={40} height={150} fill="var(--label)" opacity={0.6} />
            <rect x={342} y={0} width={60} height={150} fill="var(--label)" opacity={0.6} />
            <rect className="play-buf" x={84} y={0} width={28} height={150} fill={BEFORE} opacity={0.2} />
            <rect className="play-buf" x={152} y={0} width={14} height={150} fill={AFTER} opacity={0.2} />
            <rect className="play-buf" x={402} y={0} width={46} height={150} fill={AFTER} opacity={0.2} />
            <polyline points="0,52 106,52 114,40 130,26 146,42 156,52 330,52 344,30 374,24 402,36 414,52 520,52" fill="none" stroke="var(--ink)" strokeWidth={2} />
            <path className="play-ink" d="M0,112 C40,114 60,104 86,100 C104,96 110,70 126,68 C144,66 150,96 172,100 C206,106 240,108 274,108 C298,108 312,106 334,104 C358,100 370,66 394,62 C418,58 432,84 452,98 C472,110 500,112 520,112" fill="none" stroke="var(--accent)" strokeWidth={3} strokeLinecap="round" />
            <circle className="play-pen" r={7} fill="var(--panel)" stroke="var(--accent)" strokeWidth={3} />
            <g className="play-buf" fontFamily="var(--mono)" fontSize={13} fontWeight={500}>
              <text x={98} y={170} textAnchor="middle" fill={BEFORE}>early</text>
              <text x={425} y={170} textAnchor="middle" fill={AFTER}>late</text>
            </g>
          </svg>
          <span className="press" style={{ background: "var(--topbar)", color: "var(--topbar-ink)", boxShadow: "0 4px 0 #000, 0 8px 16px rgba(0, 0, 0, 0.16)" }}>
            <span style={{ flex: 1, display: "inline-flex", alignItems: "center", gap: 12, padding: "0 18px" }}>
              <span style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.01em" }}>Draw your detection</span>
              <span className="mono" style={{ display: "inline-flex", alignItems: "center", height: 22, padding: "0 9px", borderRadius: 999, background: "rgba(255, 255, 255, 0.14)", fontSize: 12 }}>free play</span>
            </span>
            <span className="ico ico-pen" style={{ background: "#000" }}>
              <svg width={22} height={22} viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 20c4-1 6-2 8-4l7-7a2.5 2.5 0 0 0-3.5-3.5l-7 7c-2 2-3 4-4 7.5z" stroke="#FFFFFF" strokeWidth={2} strokeLinejoin="round" /><path d="M13 7.5l3.5 3.5" stroke="#FFFFFF" strokeWidth={2} /></svg>
            </span>
          </span>
          <p className="muted" style={{ margin: 0, padding: "0 4px", fontSize: 15 }}>Draw a detection over real anomalies and watch AutoVUS buffer it as you go.</p>
        </Link>
      </div>

      {games && games.length > 0 && (
        <section aria-label="Who played" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
            <span className="eyebrow">Played so far</span>
            <span className="mono muted" style={{ fontSize: 13 }}>{games.length} game{games.length === 1 ? "" : "s"} · {players} player{players === 1 ? "" : "s"} · latest {ago(games[0].when) === "just now" ? "just now" : `${ago(games[0].when)} ago`}</span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {games.slice(0, SHOW).map((x, k) => (
              <span key={k} className="pill" style={x.mode === "draw"
                ? { background: "var(--topbar)", border: "1px solid var(--topbar)", color: "var(--topbar-ink)" }
                : { background: "color-mix(in srgb, var(--accent) 12%, var(--panel))", border: "1px solid color-mix(in srgb, var(--accent) 40%, var(--panel))", color: "var(--ink)" }}>
                <span style={{ fontWeight: 600 }}>{x.name}</span>
                <span className="mono" style={{ fontSize: 12, color: x.mode === "draw" ? "var(--topbar-muted)" : "var(--muted)" }}>
                  {x.mode === "draw" ? drawn(x) : `${x.matched ?? 0} of ${x.rounds ?? 0}`} · {ago(x.when)}
                </span>
              </span>
            ))}
            {games.length > SHOW && <span className="pill mono muted" style={{ border: "1px dashed var(--border)", fontSize: 12 }}>+ {games.length - SHOW} more</span>}
          </div>
        </section>
      )}
    </main>
  );
}
