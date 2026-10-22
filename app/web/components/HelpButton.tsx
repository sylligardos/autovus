"use client";
// The "?" in the top bar: one popup per page with how the page works.
// The words live in lib/help.ts.
import { useEffect, useState } from "react";
import { HELP } from "@/lib/help";

export function HelpButton({ page }: { page: string }) {
  const [open, setOpen] = useState(false);
  const sections = HELP[page] ?? [];

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (!sections.length) return null;
  return (
    <>
      <button type="button" className="icon-btn" aria-label="How this page works" aria-expanded={open} title="How this page works" onClick={() => setOpen(!open)}>?</button>
      {open && (
        <div className="overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) setOpen(false); }} role="presentation">
          <div className="help-card" role="dialog" aria-modal="true" aria-label={sections[0].title}>
            <div className="help-head"><h2>{sections[0].title}</h2><button type="button" className="icon-btn" aria-label="Close" onClick={() => setOpen(false)}>×</button></div>
            {sections.map((sec) => (
              <dl key={sec.title} className="help-list">
                {sec.entries.map((e) => (<div key={e.term}><dt>{e.term}</dt><dd>{e.text}</dd></div>))}
              </dl>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
