// app/page.tsx — UCSD Climate Shift visualization (single screen).

"use client";

import dynamic from "next/dynamic";
import { useEffect } from "react";

import ControlPanel from "@/components/ControlPanel";
import { useStore } from "@/lib/store";

// Leaflet uses window — load the map only on the client.
const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="grid h-full w-full place-items-center bg-black text-xs text-zinc-500">
      Loading map…
    </div>
  ),
});

export default function Page() {
  const init = useStore((s) => s.init);
  const error = useStore((s) => s.error);
  const climatology = useStore((s) => s.climatology);
  const speciesList = useStore((s) => s.speciesList);

  useEffect(() => {
    void init();
  }, [init]);

  return (
    <div className="grid h-screen w-screen grid-rows-[auto_1fr] overflow-hidden">
      <header className="flex items-center justify-between border-b border-zinc-900 bg-zinc-950/95 px-5 py-3 backdrop-blur">
        <div>
          <h1 className="text-sm font-semibold tracking-tight">
            UCSD Climate Shift
          </h1>
          <p className="text-[11px] text-zinc-500">
            Counterfactual SDM · iNaturalist + 13-session microclimate sensor
            walk · DataHacks 2026
          </p>
        </div>
        <div className="font-mono text-[11px] text-zinc-500">
          {climatology
            ? `${climatology.count} clim cells · ${climatology.temp_min.toFixed(1)}–${climatology.temp_max.toFixed(1)}°C`
            : "loading…"}{" "}
          · {speciesList.length} high-confidence species
        </div>
      </header>

      <main className="grid grid-cols-[1fr_360px] overflow-hidden">
        <div className="relative">
          <Map />
          {error && (
            <div className="absolute left-4 top-4 z-[1000] rounded border border-red-700 bg-red-950/90 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}
        </div>
        <ControlPanel />
      </main>
    </div>
  );
}
