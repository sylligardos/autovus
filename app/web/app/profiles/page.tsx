import { TopBar } from "@/components/TopBar";
import { Tabs } from "@/components/Tabs";
import { FinderView } from "@/components/FinderView";
import { ProfilesView } from "@/components/ProfilesView";
import { param, stateFrom } from "@/lib/model";

// Profiles: the next step, a detector per anomaly. First the anomaly profiles of one dataset (the main story),
// then the stitched score against the best single detector on the whole benchmark.
const TABS = [
  { id: "profiles", label: "Anomaly profiles" },
  { id: "stitched", label: "Stitched vs best detector" },
];
const STITCHED: [string, string] = ["Where does combining detectors per anomaly beat the best single detector?", "Each point is a series, placed by its best detector's AutoVUS and the AutoVUS of the stitched score."];

export default async function Profiles({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
  const tab = param(params, "tab") === "stitched" ? "stitched" : "profiles";
  return (
    <>
      <TopBar state={state} current="/profiles" />
      <Tabs state={state} path="/profiles" label="Profiles" tabs={TABS} tab={tab} line={tab === "stitched" ? STITCHED : undefined} />
      <div className="page">{tab === "stitched" ? <FinderView state={state} mode="stitched" /> : <ProfilesView state={state} />}</div>
    </>
  );
}
