"use client";
// Notes on one series: a small button that opens a popup with a text box.
// A second button lists every note with its series; a click opens the series.
// The notes live on the Python side (/notes), one per series.
import { useEffect, useState } from "react";
import { loadNotes, saveNotes, type Notes } from "@/lib/api";

export function NotesButton({ file, name, onOpen }: { file: string; name: string; onOpen?: (file: string) => void }) {
  const [open, setOpen] = useState(false);
  const [all, setAll] = useState(false);
  const [notes, setNotes] = useState<Notes>({});
  const [draft, setDraft] = useState("");

  useEffect(() => { loadNotes().then((n) => setNotes(n ?? {})); }, []);
  useEffect(() => { setDraft(notes[file]?.note ?? ""); }, [file, notes, open]);
  useEffect(() => {
    if (!open && !all) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpen(false); setAll(false); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, all]);
  // Every note, the latest first.
  const list = Object.entries(notes).sort((a, b) => b[1].updated.localeCompare(a[1].updated));

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
      <button type="button" className="btn" aria-expanded={all} onClick={() => { loadNotes().then((n) => setNotes(n ?? {})); setAll(true); }}>All notes <span className="mono muted">{list.length}</span></button>
      {all && (
        <div className="overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) setAll(false); }} role="presentation">
          <div className="help-card" role="dialog" aria-modal="true" aria-label="All notes" style={{ width: "min(720px, 100%)" }}>
            <div className="help-head"><h2>All notes</h2><button type="button" className="icon-btn" aria-label="Close" onClick={() => setAll(false)}>×</button></div>
            {list.length === 0 && <span className="muted" style={{ fontSize: 13 }}>No notes yet.</span>}
            <div style={{ display: "flex", flexDirection: "column" }}>
              {list.map(([f, n]) => (
                <button key={f} type="button" onClick={() => { onOpen?.(f); setAll(false); }} disabled={!onOpen}
                  style={{ display: "flex", flexDirection: "column", gap: 4, padding: "10px 4px", border: 0, borderTop: "1px solid var(--border-soft)", background: "transparent", color: "var(--ink)", font: "inherit", textAlign: "left", cursor: onOpen ? "pointer" : "default" }}>
                  <span style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: f === file ? "var(--accent)" : undefined, wordBreak: "break-all" }}>{f.replace(/\.csv$/, "")}</span>
                    <span className="mono muted" style={{ fontSize: 11, flex: "none" }}>{n.updated.slice(0, 10)}</span>
                  </span>
                  <span className="muted" style={{ fontSize: 13, whiteSpace: "pre-wrap" }}>{n.note}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
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
