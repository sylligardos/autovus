import { TopBar } from "@/components/TopBar";
import { FinderView } from "@/components/FinderView";
import { stateFrom } from "@/lib/model";

export default async function Finder({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/finder" />
      <div className="page"><FinderView state={state} /></div>
    </>
  );
}
