import type { ReactNode } from "react";
import type { State } from "@/lib/model";
import { TopBar } from "./TopBar";

// A page: the top bar, then a controls rail (optional) beside the content.
export function Shell({ state, current, rail, children }: { state: State; current: string; rail?: ReactNode; children: ReactNode }) {
  return (
    <>
      <TopBar state={state} current={current} />
      <div className="page">
        {rail && <aside className="rail" aria-label="Controls">{rail}</aside>}
        <main className="content">{children}</main>
      </div>
    </>
  );
}
