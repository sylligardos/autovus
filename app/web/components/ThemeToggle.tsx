"use client";
import { setTheme, useTheme } from "@/lib/theme";

export function ThemeToggle() {
  const theme = useTheme();
  const next = theme === "light" ? "dark" : "light";
  return (
    <button type="button" className="icon-btn" onClick={() => setTheme(next)} title={`Switch to ${next} mode`} aria-label={`Switch to ${next} mode`}>
      {theme === "light" ? (
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5z" /></svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><circle cx="8" cy="8" r="3.2" /><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6L13 13M3 13l1.4-1.4M11.6 4.4L13 3" /></svg>
      )}
    </button>
  );
}
