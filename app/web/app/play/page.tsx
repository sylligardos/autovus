import { TopBar } from "@/components/TopBar";
import { GameView } from "@/components/GameView";
import { param, stateFrom } from "@/lib/model";

export default async function Play({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  // The candidate rounds: /play?set=candidates (also accepted: /play?b=candidates).
  const candidates = param(params, "set") === "candidates" || param(params, "b") === "candidates";
  const state = stateFrom(param(params, "b") === "candidates" ? { ...params, b: undefined } : params);
  return (
    <>
      <TopBar state={state} current="/play" />
      <div className="page"><GameView state={state} candidates={candidates} /></div>
    </>
  );
}
