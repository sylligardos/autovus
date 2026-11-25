// Demo paper, Figure 2: screenshots of the demo views, taken with headless Chrome through the
// DevTools protocol (no extra package). The demo must run (API on :8765, web on :3000).
//
//   node src/experiments/demo_screens_capture.mjs [outdir]          (default experiments/figures/demo_screens)
//
// Light theme, 1600 x 1000 CSS pixels at device scale 2. Each panel is one PNG; the compose step
// (src/experiments/demo_screens.py) crops and arranges them.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const OUT = resolve(process.argv[2] ?? "experiments/figures/demo_screens");
const WEB = "http://localhost:3000";
const PORT = 9333, W = 1600, H = 1000, SCALE = 2;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The panels: url, then JS run in the page (may await), then the element to capture (CSS selector,
// first match) or the whole page. 556_YAHOO_id_6 is the series of Figure 1.
const S556 = "556_YAHOO_id_6_WebService_tr_500_1st_359.csv";
const PANELS = [
  // The rounds are drawn at random: lock in with no buffer and keep the round only when AutoVUS calls the
  // detection early or late, so that the two buffers differ (the run reloads the page otherwise).
  { name: "play", url: `/play?b=tsb-ad&L=20&mode=game`, retry: 15, run: `
      const t = (s) => [...document.querySelectorAll("button")].find((b) => b.textContent.trim().startsWith(s));
      for (let i = 0; i < 40 && !t("Lock in"); i++) await new Promise((r) => setTimeout(r, 250));
      t("Lock in").click();
      await new Promise((r) => setTimeout(r, 3000));
      const row = [...document.querySelectorAll("tr")].find((r) => r.textContent.trim().startsWith("AutoVUS"));
      return /(Before|After) the label/.test(row?.textContent ?? "");` },
  { name: "explore", url: `/local?b=tsb-ad&L=20&s=${S556}&det=Chronos,AutoEncoder&a=0` },
  { name: "cases", url: `/cases?b=tsb-ad&L=20&tab=mean` },
  { name: "rankings", url: `/rankings?b=tsb-ad&L=20&a=autovus&bm=local_anomaly` },
  { name: "profiles", url: `/profiles?b=tsb-ad&L=20` },
];

const dir = mkdtempSync(join(tmpdir(), "demo-shots-"));
const chrome = spawn("google-chrome", ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${dir}`,
  "--hide-scrollbars", "--no-first-run", "--no-default-browser-check", `--window-size=${W},${H}`, "about:blank"], { stdio: "ignore" });

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error("Chrome did not start");
}

const ws = new WebSocket(await target());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let next = 0;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  // The game records a played round in "Played so far" (POST /games): never from a screenshot run.
  if (m.method === "Fetch.requestPaused") {
    const { requestId, request } = m.params;
    ws.send(JSON.stringify({ id: ++next, method: request.method === "POST" ? "Fetch.failRequest" : "Fetch.continueRequest",
      params: request.method === "POST" ? { requestId, errorReason: "BlockedByClient" } : { requestId } }));
    return;
  }
  if (m.id && pending.has(m.id)) { const { ok, ko } = pending.get(m.id); pending.delete(m.id); m.error ? ko(new Error(m.error.message)) : ok(m.result); }
});
const send = (method, params = {}) => new Promise((ok, ko) => { const id = ++next; pending.set(id, { ok, ko }); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  return r.result.value;
};

await send("Page.enable");
await send("Fetch.enable", { patterns: [{ urlPattern: "*games*", requestStage: "Request" }] });
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: SCALE, mobile: false });
// Light theme, no player name, and no Next.js development badge.
await send("Page.addScriptToEvaluateOnNewDocument", { source: `
  try { localStorage.setItem("autovus.theme", "light"); localStorage.removeItem("autovus:player"); } catch {}
  document.addEventListener("DOMContentLoaded", () => {
    const s = document.createElement("style"); s.textContent = "nextjs-portal { display: none !important; }"; document.head.appendChild(s);
  });` });

mkdirSync(OUT, { recursive: true });
for (const p of PANELS) {
  for (let tries = p.retry ?? 1; tries > 0; tries--) {
    await send("Page.navigate", { url: WEB + p.url });
    await sleep(4000);                                     // data and charts
    if (!p.run) break;
    const ok = await evaluate(p.run);
    if (ok !== false) break;
    if (tries === 1) console.log(`${p.name}: no early or late round in ${p.retry} draws, keeping the last one`);
  }
  await sleep(800);
  const { cssContentSize } = await send("Page.getLayoutMetrics");
  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: W, height: Math.min(cssContentSize.height, 2400), scale: 1 } });
  writeFileSync(join(OUT, `${p.name}.png`), Buffer.from(shot.data, "base64"));
  console.log("saved", join(OUT, `${p.name}.png`));
}
ws.close();
chrome.kill();
