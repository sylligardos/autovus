import { TopBar } from "@/components/TopBar";
import { GameView } from "@/components/GameView";
import { DrawView } from "@/components/DrawView";
import { PlayIntro } from "@/components/PlayIntro";
import { param, stateFrom } from "@/lib/model";

// /play: the landing with the two games. /play?mode=game: Find the buffer. /play?mode=draw: Draw your detection.
// /play?set=candidates (also /play?b=candidates): every candidate round in order, to choose what joins the pool.
export default async function Play({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const candidates = param(params, "set") === "candidates" || param(params, "b") === "candidates";
  const mode = param(params, "mode");
  const state = stateFrom(param(params, "b") === "candidates" ? { ...params, b: undefined } : params);
  return (
    <>
      <TopBar state={state} current="/play" />
      <div className="page">
        {candidates ? <GameView state={state} candidates /> : mode === "game" ? <GameView state={state} /> : mode === "draw" ? <DrawView state={state} /> : <PlayIntro state={state} />}
      </div>
    </>
  );
}
