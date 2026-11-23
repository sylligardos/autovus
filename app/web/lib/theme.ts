"use client";
// The colour theme. "light" is the default; "dark" is set as data-theme on
// <html> and remembered in localStorage. A script in layout.tsx applies it
// before paint.
import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "autovus.theme";

export function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function setTheme(t: Theme) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem(KEY, t); } catch { /* private window */ }
}

export function useTheme(): Theme {
  const [t, setT] = useState<Theme>("light");
  useEffect(() => {
    setT(readTheme());
    const o = new MutationObserver(() => setT(readTheme()));
    o.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => o.disconnect();
  }, []);
  return t;
}

// Inline, runs before paint: localStorage first, then ?theme=, then the OS preference, then light.
export const THEME_BOOT = `(function(){try{var t=localStorage.getItem(${JSON.stringify(KEY)});if(t!=="light"&&t!=="dark"){var q=new URLSearchParams(location.search).get("theme");t=q==="light"||q==="dark"?q:(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");}document.documentElement.dataset.theme=t;}catch(e){}})();`;
