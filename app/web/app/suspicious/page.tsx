import { TopBar } from "@/components/TopBar";
import { SuspiciousView } from "@/components/SuspiciousView";
import { stateFrom } from "@/lib/model";

export default async function Suspicious({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/suspicious" />
      <div className="page"><SuspiciousView state={state} /></div>
    </>
  );
}
