"use client";
import { useRouter } from "next/navigation";

export function BenchmarkSelect({ value, options, current }: { value: string; options: { id: string; label: string }[]; current: string }) {
  const router = useRouter();
  return (
    <select
      className="select"
      aria-label="Benchmark"
      value={value}
      onChange={(e) => {
        const q = new URLSearchParams(window.location.search);
        q.set("b", e.target.value);
        router.push(`${current}?${q}`);
      }}
    >
      {options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
    </select>
  );
}
