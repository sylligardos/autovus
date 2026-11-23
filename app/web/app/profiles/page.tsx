import { TopBar } from "@/components/TopBar";
import { ProfilesView } from "@/components/ProfilesView";
import { stateFrom } from "@/lib/model";

export default async function Profiles({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/profiles" />
      <div className="page"><ProfilesView state={state} /></div>
    </>
  );
}
