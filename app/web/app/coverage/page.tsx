import { TopBar } from "@/components/TopBar";
import { CoverageView } from "@/components/CoverageView";
import { stateFrom } from "@/lib/model";

export default async function Coverage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/coverage" />
      <div className="page"><CoverageView state={state} /></div>
    </>
  );
}
