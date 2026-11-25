"use client";
// The gear of the whole demo: the maximum buffer L, kept in the URL (?L=) like the benchmark.

import { usePathname, useRouter } from "next/navigation";
import { GEARS, type Gear } from "@/lib/model";

export function GearSwitch({ L }: { L: Gear }) {
  const router = useRouter(), pathname = usePathname();
  const go = (g: Gear) => {
    // The live URL, not useSearchParams: pages write their own state (series, zoom) with history.replaceState.
    const q = new URLSearchParams(window.location.search);
    q.set("L", String(g));
    router.push(`${pathname}?${q}`);
  };
  return (
    <div className="gear" role="group" aria-label="Maximum buffer L">
      <span>L</span>
      {GEARS.map((g) => (
        <button key={g} type="button" aria-pressed={g === L} onClick={() => go(g)}>{g}</button>
      ))}
    </div>
  );
}
