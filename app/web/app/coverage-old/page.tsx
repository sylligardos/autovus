import { TopBar } from "@/components/TopBar";
import { CoverageView } from "@/components/CoverageView";
import { stateFrom } from "@/lib/model";

// The first Coverage page (best coverage, minimum set), kept while the new one is built.
export default async function CoverageOld({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/coverage-old" />
      <div className="page"><CoverageView state={state} /></div>
    </>
  );
}
