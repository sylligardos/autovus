import { redirect } from "next/navigation";
import { TopBar } from "@/components/TopBar";
import { Tabs } from "@/components/Tabs";
import { FinderView, type Mode } from "@/components/FinderView";
import { href, param, stateFrom } from "@/lib/model";

// Cases: where one score per series hides what happens on its anomalies. Two comparisons, one per tab;
// every point and row opens the series in Explore. (Stitched vs best detector is a tab of Profiles.)
const TABS: { id: Mode; label: string }[] = [
  { id: "mean", label: "Global vs mean local" },
  { id: "worst", label: "Global vs lowest local" },
];
// The one sentence on top of each tab: what it answers, then how to read it.
const LINES: Record<string, [string, string]> = {
  mean: ["Where does one score per series differ from the average score on its anomalies?", "Each point is a detector on a series, placed by its AutoVUS on the whole series and the mean over its anomalies."],
  worst: ["Which detectors score well on a series but miss one of its anomalies?", "Each point is a detector on a series, placed by its AutoVUS on the whole series and its lowest over its anomalies."],
};

export default async function Cases({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
  const p = param(params, "tab");
  if (p === "stitched") redirect(href("/profiles", state, { tab: "stitched" }));   // links from before the tab moved
  const tab: Mode = p === "worst" ? p : "mean";
  return (
    <>
      <TopBar state={state} current="/cases" />
      <Tabs state={state} path="/cases" label="Cases" tabs={TABS} tab={tab} line={LINES[tab]} />
      <div className="page"><FinderView state={state} mode={tab} /></div>
    </>
  );
}
