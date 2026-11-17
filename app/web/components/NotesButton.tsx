"use client";
// Notes on one series: a small button that opens a popup with a text box.
// The notes live on the Python side (/notes), one per series.
import { useEffect, useState } from "react";
import { loadNotes, saveNotes, type Notes } from "@/lib/api";

export function NotesButton({ file, name }: { file: string; name: string }) {
  const [open, setOpen] = useState(false);
  const [notes, setNotes] = useState<Notes>({});
  const [draft, setDraft] = useState("");

  useEffect(() => { loadNotes().then((n) => setNotes(n ?? {})); }, []);
  useEffect(() => { setDraft(notes[file]?.note ?? ""); }, [file, notes, open]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const save = () => {
    const next = { ...notes, [file]: { note: draft, updated: new Date().toISOString() } };
    if (!draft.trim()) delete next[file];
    setNotes(next); saveNotes(next); setOpen(false);
  };
  const has = !!notes[file];
  return (
    <>
      <button type="button" className="btn" aria-expanded={open} onClick={() => setOpen(true)} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        Notes{has && <span aria-label="has a note" style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--accent)" }} />}
      </button>
      {open && (
        <div className="overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) setOpen(false); }} role="presentation">
          <div className="help-card" role="dialog" aria-modal="true" aria-label="Notes on this series">
            <div className="help-head"><h2>Notes <span className="mono muted" style={{ fontSize: 12, fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>{name}</span></h2><button type="button" className="icon-btn" aria-label="Close" onClick={() => setOpen(false)}>×</button></div>
            <textarea rows={6} autoFocus className="input" aria-label="Note" style={{ width: "100%", height: "auto", padding: "10px 12px", resize: "vertical", background: "var(--subtle)", boxSizing: "border-box" }} value={draft} onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save(); } }} />
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
              <button type="button" className="btn btn-primary" onClick={save} title="Ctrl + Enter">Save</button>
              {has && <span className="muted" style={{ fontSize: 12 }}>saved {notes[file].updated.slice(0, 10)}</span>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
