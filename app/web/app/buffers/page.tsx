import { TopBar } from "@/components/TopBar";
import { BuffersView } from "@/components/BuffersView";
import { stateFrom } from "@/lib/model";

export default async function Buffers({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  return (
    <>
      <TopBar state={state} current="/buffers" />
      <div className="page" style={{ maxWidth: 1640 }}><BuffersView state={state} /></div>
    </>
  );
}
