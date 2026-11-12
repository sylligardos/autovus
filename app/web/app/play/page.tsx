import { TopBar } from "@/components/TopBar";
import { GameView } from "@/components/GameView";
import { stateFrom } from "@/lib/model";

export default async function Play({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/play" />
      <div className="page"><GameView state={state} /></div>
    </>
  );
}
