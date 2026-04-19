// frontend/components/ControlPanel.tsx — species, day, warming, badges, readout.

"use client";

import { DAY_PRESETS, useStore } from "@/lib/store";

function ConfidenceBadge({
  auc,
  high,
}: {
  auc: number | null;
  high: boolean;
}) {
  if (auc == null)
    return (
      <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
        no CV AUC
      </span>
    );
  const color =
    auc >= 0.75
      ? "bg-emerald-700 text-emerald-100"
      : auc >= 0.65
        ? "bg-amber-700 text-amber-100"
        : "bg-red-800 text-red-100";
  const label = high ? "high confidence" : "low confidence";
  return (
    <span
      className={`rounded px-2 py-0.5 text-xs font-mono ${color}`}
      title="Spatial cross-validation AUC"
    >
      AUC {auc.toFixed(2)} · {label}
    </span>
  );
}

export default function ControlPanel() {
  const speciesList = useStore((s) => s.speciesList);
  const selectedSpecies = useStore((s) => s.selectedSpecies);
  const dayOfYear = useStore((s) => s.dayOfYear);
  const warming = useStore((s) => s.warming);
  const baseline = useStore((s) => s.baselineGrid);
  const current = useStore((s) => s.currentGrid);
  const loading = useStore((s) => s.loading);
  const error = useStore((s) => s.error);
  const setSpecies = useStore((s) => s.setSpecies);
  const setDay = useStore((s) => s.setDay);
  const setWarming = useStore((s) => s.setWarming);

  const sp = speciesList.find((s) => s.taxon_name === selectedSpecies);
  // Mean probability across all grid cells — always non-zero and always
  // responds to warming, even for species with low overall habitat coverage.
  const meanProb = (g: typeof current) => (g ? (g.mean_prob ?? 0) : 0);

  const baseFrac = meanProb(baseline);
  const curFrac = meanProb(current);
  const delta = curFrac - baseFrac;

  return (
    <aside className="flex h-full w-full flex-col gap-5 overflow-y-auto bg-zinc-950 p-5 text-zinc-100">
      <header>
        <h2 className="text-base font-semibold tracking-tight">
          Climate counterfactual
        </h2>
        <p className="mt-1 text-xs text-zinc-400">
          Drag the slider to warm UCSD by N degrees and watch this species&rsquo;
          habitat respond.
        </p>
      </header>

      {/* Species */}
      <section className="space-y-2">
        <label className="text-xs uppercase tracking-wider text-zinc-400">
          Species
        </label>
        <select
          value={selectedSpecies}
          onChange={(e) => void setSpecies(e.target.value)}
          className="w-full rounded border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm focus:border-emerald-500 focus:outline-none"
          disabled={loading || speciesList.length === 0}
        >
          {speciesList.map((s) => (
            <option key={s.taxon_name} value={s.taxon_name}>
              {s.common_name} — {s.taxon_name}
            </option>
          ))}
        </select>
        <div className="flex items-center gap-2">
          {sp && (
            <ConfidenceBadge auc={sp.spatial_cv_auc} high={sp.high_confidence} />
          )}
          {sp && (
            <span className="text-xs text-zinc-500">{sp.iconic_taxon}</span>
          )}
        </div>
      </section>

      {/* Day of year */}
      <section className="space-y-2">
        <label className="text-xs uppercase tracking-wider text-zinc-400">
          Season
        </label>
        <div className="grid grid-cols-4 gap-1">
          {Object.entries(DAY_PRESETS).map(([label, doy]) => (
            <button
              key={label}
              onClick={() => void setDay(doy)}
              className={`rounded border px-2 py-1.5 text-xs transition ${
                doy === dayOfYear
                  ? "border-emerald-500 bg-emerald-500/10 text-emerald-300"
                  : "border-zinc-800 bg-zinc-900 text-zinc-300 hover:border-zinc-600"
              }`}
              disabled={loading}
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      {/* Warming slider */}
      <section className="space-y-2">
        <label className="flex items-center justify-between text-xs uppercase tracking-wider text-zinc-400">
          <span>Warming offset</span>
          <span className="font-mono text-emerald-400">
            +{warming.toFixed(1)} °C
          </span>
        </label>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={warming}
          onChange={(e) => setWarming(parseFloat(e.target.value))}
          className="w-full accent-emerald-500"
        />
        <div className="flex justify-between text-[10px] text-zinc-500">
          <span>0°</span>
          <span>+1</span>
          <span>+2</span>
          <span>+3</span>
          <span>+4</span>
          <span>+5°</span>
        </div>
      </section>

      {/* Range readout */}
      <section className="rounded border border-zinc-800 bg-zinc-900/40 p-3">
        <div className="text-xs uppercase tracking-wider text-zinc-400">
          Mean habitat suitability
        </div>
        <div className="mt-2 flex items-end gap-3 font-mono">
          <div>
            <div className="text-[10px] text-zinc-500">baseline</div>
            <div className="text-lg text-zinc-200">
              {(baseFrac * 100).toFixed(1)}%
            </div>
          </div>
          <div className="text-zinc-600">→</div>
          <div>
            <div className="text-[10px] text-zinc-500">
              +{warming.toFixed(1)}°C
            </div>
            <div className="text-lg text-zinc-200">
              {(curFrac * 100).toFixed(1)}%
            </div>
          </div>
          <div className="ml-auto text-right">
            <div className="text-[10px] text-zinc-500">Δ</div>
            <div
              className={`text-lg ${
                delta > 0.001
                  ? "text-emerald-400"
                  : delta < -0.001
                    ? "text-red-400"
                    : "text-zinc-400"
              }`}
            >
              {delta > 0 ? "+" : ""}
              {(delta * 100).toFixed(1)} pp
            </div>
          </div>
        </div>
      </section>

      {error && (
        <div className="rounded border border-red-700 bg-red-900/30 p-2 text-xs text-red-200">
          {error}
        </div>
      )}

      <footer className="mt-auto pt-4 text-[10px] leading-relaxed text-zinc-600">
        Multi-label MLP SDM trained on iNaturalist research-grade observations
        (UCSD campus) with target-group background sampling, spatial 5×5 block
        CV. Per-cell climate from 13-session mobile temperature/humidity sensor
        walk. {speciesList.length} high-confidence species (AUC ≥ 0.65).
      </footer>
    </aside>
  );
}
