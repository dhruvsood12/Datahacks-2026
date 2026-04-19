// frontend/lib/store.ts — Zustand store + actions for the visualization.

"use client";

import { create } from "zustand";
import {
  ClimatologyResponse,
  PredictGridResponse,
  Species,
  getClimatology,
  getSpecies,
  postPredictGrid,
} from "./api";

// Default species: Cotinis mutabilis (Figeater Beetle).
// AUC 0.846, recognizable, responds clearly to warming.
const DEFAULT_SPECIES = "Cotinis mutabilis";

// Day-of-year presets — Jan 15, Apr 15, Jul 15, Oct 15
export const DAY_PRESETS: Record<string, number> = {
  "Jan 15": 15,
  "Apr 15": 105,
  "Jul 15": 196,
  "Oct 15": 288,
};

type State = {
  speciesList: Species[];
  selectedSpecies: string;
  dayOfYear: number;
  warming: number; // °C offset
  climatology: ClimatologyResponse | null;
  baselineGrid: PredictGridResponse | null; // offset = 0, fixed for the selected species/day
  currentGrid: PredictGridResponse | null;
  loading: boolean;
  error: string | null;

  setSpecies: (taxon: string) => Promise<void>;
  setDay: (doy: number) => Promise<void>;
  setWarming: (w: number) => void;
  init: () => Promise<void>;
  refreshGrid: () => Promise<void>;
};

let warmingDebounce: ReturnType<typeof setTimeout> | null = null;

export const useStore = create<State>((set, get) => ({
  speciesList: [],
  selectedSpecies: DEFAULT_SPECIES,
  dayOfYear: 196,
  warming: 0,
  climatology: null,
  baselineGrid: null,
  currentGrid: null,
  loading: false,
  error: null,

  init: async () => {
    set({ loading: true, error: null });
    try {
      const [clim, sp] = await Promise.all([getClimatology(), getSpecies(true)]);
      set({
        climatology: clim,
        speciesList: sp.species,
      });
      // First baseline + current grid load
      await get().refreshGrid();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ error: `Failed to init: ${msg}` });
    } finally {
      set({ loading: false });
    }
  },

  setSpecies: async (taxon) => {
    set({ selectedSpecies: taxon, baselineGrid: null });
    await get().refreshGrid();
  },

  setDay: async (doy) => {
    set({ dayOfYear: doy, baselineGrid: null });
    await get().refreshGrid();
  },

  setWarming: (w) => {
    set({ warming: w });
    if (warmingDebounce) clearTimeout(warmingDebounce);
    warmingDebounce = setTimeout(() => {
      void get().refreshGrid();
    }, 200);
  },

  refreshGrid: async () => {
    const { selectedSpecies, dayOfYear, warming, baselineGrid } = get();
    set({ loading: true, error: null });
    try {
      // Always fetch the current view; fetch baseline once per (species, day)
      const tasks: Promise<PredictGridResponse>[] = [
        postPredictGrid({
          species: selectedSpecies,
          day_of_year: dayOfYear,
          temperature_offset: warming,
        }),
      ];
      if (!baselineGrid) {
        tasks.push(
          postPredictGrid({
            species: selectedSpecies,
            day_of_year: dayOfYear,
            temperature_offset: 0,
          }),
        );
      }
      const results = await Promise.all(tasks);
      const current = results[0];
      const baseline = results[1] ?? baselineGrid;
      set({ currentGrid: current, baselineGrid: baseline ?? current });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ error: `Predict failed: ${msg}` });
    } finally {
      set({ loading: false });
    }
  },
}));
