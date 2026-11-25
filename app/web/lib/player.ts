// The visitor's name for Played so far: asked once on the Play landing, kept in
// the browser (localStorage), so every game they finish is recorded under it.

const KEY = "autovus:player";
export const ANON = "Someone";

export function getPlayer(): string {
  try { return (localStorage.getItem(KEY) ?? "").trim(); } catch { return ""; }
}
export function setPlayer(name: string) {
  try { localStorage.setItem(KEY, name.trim().slice(0, 24)); } catch { /* private window or storage off: the name lives for this page only */ }
}
