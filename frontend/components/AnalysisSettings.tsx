"use client";

import { useEffect, useMemo, useState } from "react";

/**
 * Sprint 3 / A9 — Analysis settings panel.
 *
 * Toggles the same fields that PipelineConfig accepts (use_ade /
 * highlight_fence_text / enable_unified_measurement /
 * enable_nonlayer_suggestions / display_image_dpi / fence_keywords).
 * The dashboard passes JSON.stringify of these values to UploadButton,
 * which forwards them as the `config` form field on POST /api/jobs.
 *
 * Persisted to localStorage so users don't re-toggle every session.
 */

const STORAGE_KEY = "leo:analysis-settings:v1";

// Contracting trades (must mirror config.TRADE_PROFILES on the backend). Each
// trade brings its own default keyword set; `supportsMeasurement` gates the
// fence-only linear-measurement options.
export const TRADES = {
  fence: {
    label: "Fence",
    supportsMeasurement: true,
    keywords: [
      "fence", "fencing", "gate", "barrier", "guardrail", "post", "mesh",
      "panel", "chain link", "masonry", "fence details", "canopy shading",
      "adot specifications", "mag specifications", "rail", "railing",
      "bollards", "handrails", "wall", "cmu",
      "operator", "davis", "bacon", "davis-bacon", "davis – bacon",
      "buy america", "american", "dug out",
    ],
  },
  electrical: {
    label: "Electrical",
    supportsMeasurement: false,
    keywords: [
      "one line", "one-line", "single line", "single-line", "one line diagram",
      "available power", "loads", "load schedule",
      "cable schedule", "wire schedule", "conduit schedule",
      "outlets", "receptacles", "lights", "lighting", "luminaire",
      "panel schedule", "panelboard schedule", "panelboard",
      "feeder", "circuit", "breaker", "switchgear", "transformer", "grounding",
    ],
  },
} as const;

export type TradeId = keyof typeof TRADES;

export type AnalysisSettings = {
  trade: TradeId;
  use_ade: boolean;
  highlight_fence_text: boolean;
  enable_unified_measurement: boolean;
  enable_nonlayer_suggestions: boolean;
  display_image_dpi: number;
  fence_keywords: string[];
};

const DEFAULTS: AnalysisSettings = {
  trade: "fence",
  use_ade: true,
  highlight_fence_text: true,
  enable_unified_measurement: true,
  enable_nonlayer_suggestions: false,
  display_image_dpi: 150,
  fence_keywords: [...TRADES.fence.keywords],
};

function loadFromStorage(): AnalysisSettings {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw);
    return { ...DEFAULTS, ...parsed };
  } catch {
    return DEFAULTS;
  }
}

export function useAnalysisSettings() {
  // Default value only matters on first paint pre-hydration; the effect
  // below pulls the persisted state. SSR-safe.
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULTS);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setSettings(loadFromStorage());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // localStorage unavailable (Safari private mode etc.) — silently skip.
    }
  }, [settings, hydrated]);

  const configJson = useMemo(() => JSON.stringify(settings), [settings]);

  return { settings, setSettings, configJson, hydrated };
}

/**
 * Fence / Electrical segmented control. Shared between the prominent
 * dashboard placement and the Analysis settings panel so the active mode is
 * always visible and in sync. Switching swaps in that trade's default
 * keywords (the panel's textarea resyncs via its fence_keywords effect) and
 * disables linear measurement for trades that don't support it.
 */
export function TradeModeSelector({
  settings,
  setSettings,
  size = "md",
}: {
  settings: AnalysisSettings;
  setSettings: (s: AnalysisSettings) => void;
  size?: "sm" | "md";
}) {
  const trade: TradeId = settings.trade ?? "fence";

  function switchTrade(next: TradeId) {
    if (next === trade) return;
    const meta = TRADES[next];
    setSettings({
      ...settings,
      trade: next,
      fence_keywords: [...meta.keywords],
      enable_unified_measurement: meta.supportsMeasurement
        ? settings.enable_unified_measurement
        : false,
    });
  }

  const pad = size === "sm" ? "px-2.5 py-0.5 text-xs" : "px-3 py-1 text-sm";
  return (
    <div className="inline-flex rounded-lg border bg-gray-50 p-0.5">
      {(Object.keys(TRADES) as TradeId[]).map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => switchTrade(t)}
          aria-pressed={trade === t}
          className={`${pad} rounded-md transition ${
            trade === t
              ? "bg-white shadow text-gray-900 font-medium"
              : "text-gray-600 hover:text-gray-900"
          }`}
        >
          {TRADES[t].label}
        </button>
      ))}
    </div>
  );
}

export function AnalysisSettingsPanel({
  settings,
  setSettings,
}: {
  settings: AnalysisSettings;
  setSettings: (s: AnalysisSettings) => void;
}) {
  const [open, setOpen] = useState(false);
  const [keywordsText, setKeywordsText] = useState(settings.fence_keywords.join(", "));

  const trade: TradeId = settings.trade ?? "fence";
  const tradeMeta = TRADES[trade] ?? TRADES.fence;
  const tradeLabel = tradeMeta.label;

  // Resync keyword text if the parent settings change (e.g. on reset or a
  // mode switch, which swaps the keyword set).
  useEffect(() => {
    setKeywordsText(settings.fence_keywords.join(", "));
  }, [settings.fence_keywords]);

  const nonDefaultCount =
    (settings.use_ade !== DEFAULTS.use_ade ? 1 : 0) +
    (settings.highlight_fence_text !== DEFAULTS.highlight_fence_text ? 1 : 0) +
    (settings.enable_unified_measurement !== DEFAULTS.enable_unified_measurement ? 1 : 0) +
    (settings.enable_nonlayer_suggestions !== DEFAULTS.enable_nonlayer_suggestions ? 1 : 0) +
    (settings.display_image_dpi !== DEFAULTS.display_image_dpi ? 1 : 0) +
    (settings.fence_keywords.length !== DEFAULTS.fence_keywords.length ? 1 : 0);

  function setField<K extends keyof AnalysisSettings>(
    key: K,
    value: AnalysisSettings[K],
  ) {
    setSettings({ ...settings, [key]: value });
  }

  function applyKeywords() {
    const parsed = keywordsText
      .split(/[,\n]/)
      .map((k) => k.trim().toLowerCase())
      .filter((k) => k.length > 0);
    setField("fence_keywords", parsed);
  }

  function resetAll() {
    setSettings(DEFAULTS);
    setKeywordsText(DEFAULTS.fence_keywords.join(", "));
  }

  return (
    <details
      className="border rounded-lg bg-white"
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary className="cursor-pointer px-4 py-3 flex items-center justify-between gap-3 hover:bg-gray-50 rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-800">
            Analysis settings
          </span>
          {nonDefaultCount > 0 && (
            <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700">
              {nonDefaultCount} customised
            </span>
          )}
        </div>
        <span className="text-xs text-gray-500">
          {open ? "hide" : "show"}
        </span>
      </summary>

      <div className="px-4 pb-4 pt-2 space-y-4 border-t">
        {/* Analysis mode / contracting trade */}
        <div>
          <label className="text-sm text-gray-700 block mb-1.5">Analysis mode</label>
          <TradeModeSelector settings={settings} setSettings={setSettings} />
          <div className="text-xs text-gray-500 mt-1">
            Which trade to detect. Switching mode loads that trade&apos;s default
            keywords.
          </div>
        </div>

        <ToggleRow
          label="Use ADE detection"
          help="LandingAI ADE for legend & figure extraction. Off = keyword-only, much faster but lower recall."
          checked={settings.use_ade}
          onChange={(v) => setField("use_ade", v)}
        />
        <ToggleRow
          label={`Highlight ${tradeLabel.toLowerCase()} text in PDF`}
          help="Annotate the highlighted PDF with green/purple/orange boxes for legend/figure/keyword hits."
          checked={settings.highlight_fence_text}
          onChange={(v) => setField("highlight_fence_text", v)}
        />
        {tradeMeta.supportsMeasurement && (
          <>
            <ToggleRow
              label="Run unified measurement"
              help="Auto-measure fence-line lengths in feet. Off = no per-page totals; faster."
              checked={settings.enable_unified_measurement}
              onChange={(v) => setField("enable_unified_measurement", v)}
            />
            <ToggleRow
              label="Suggest non-layer fence elements"
              help="Experimental: try to recover fence runs from pages with no CAD layer info. May add noise."
              checked={settings.enable_nonlayer_suggestions}
              onChange={(v) => setField("enable_nonlayer_suggestions", v)}
            />
          </>
        )}

        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-700 w-44 shrink-0">
            Page-image DPI
          </label>
          <input
            type="number"
            min={50}
            max={300}
            step={10}
            value={settings.display_image_dpi}
            onChange={(e) =>
              setField("display_image_dpi", Math.max(50, Math.min(300, Number(e.target.value) || 150)))
            }
            className="w-24 border rounded px-2 py-1 text-sm font-mono"
          />
          <span className="text-xs text-gray-500">
            Higher = sharper page images, slower render. Default 150.
          </span>
        </div>

        <div>
          <label className="text-sm text-gray-700 block mb-1">
            {tradeLabel} keywords ({settings.fence_keywords.length})
          </label>
          <textarea
            value={keywordsText}
            onChange={(e) => setKeywordsText(e.target.value)}
            onBlur={applyKeywords}
            rows={3}
            className="w-full border rounded px-2 py-1 text-xs font-mono"
            placeholder="comma- or newline-separated keywords"
          />
          <div className="text-xs text-gray-500 mt-1">
            Words/phrases that trigger {tradeLabel.toLowerCase()}-page classification when found in OCR or native text.
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t">
          <span className="text-xs text-gray-500">
            Settings persist in your browser and apply to new uploads.
          </span>
          <button
            type="button"
            onClick={resetAll}
            className="text-xs text-blue-600 hover:underline"
          >
            Reset to defaults
          </button>
        </div>
      </div>
    </details>
  );
}

function ToggleRow({
  label,
  help,
  checked,
  onChange,
}: {
  label: string;
  help: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
      />
      <div className="flex-1">
        <div className="text-sm text-gray-800">{label}</div>
        <div className="text-xs text-gray-500">{help}</div>
      </div>
    </label>
  );
}
