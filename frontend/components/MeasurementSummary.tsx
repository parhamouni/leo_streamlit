"use client";

import { useCallback, useEffect, useState } from "react";
import { apiJson } from "@/lib/api";

export type CategoryEntry = {
  pts: number;
  ft: number;
  auto: number;
  manual: number;
};

export type PageRow = {
  page_num: number;
  scale: number;
  has_user_edits: boolean;
  per_category: Record<string, CategoryEntry>;
};

export type SummaryResponse = {
  pages: PageRow[];
  grand_total: Record<string, CategoryEntry>;
  page_count: number;
  category_count: number;
};

const PALETTE: [number, number, number][] = [
  [0, 255, 0],
  [255, 165, 0],
  [0, 191, 255],
  [255, 0, 255],
  [255, 255, 0],
  [0, 255, 255],
  [255, 105, 180],
  [173, 255, 47],
];

function colorFor(idx: number) {
  return PALETTE[idx % PALETTE.length];
}

export function MeasurementSummary({
  jobId,
  refreshSignal = 0,
  onDataChange,
}: {
  jobId: string;
  refreshSignal?: number;
  onDataChange?: (data: SummaryResponse) => void;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SummaryResponse | null>(null);

  const fetchSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await apiJson<SummaryResponse>(
        `/api/jobs/${jobId}/measurement-summary`,
      );
      setData(d);
      onDataChange?.(d);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [jobId, onDataChange]);

  useEffect(() => {
    if (refreshSignal > 0) void fetchSummary();
  }, [fetchSummary, refreshSignal]);

  function toggle() {
    if (!open) {
      setOpen(true);
      if (!data) void fetchSummary();
    } else {
      setOpen(false);
    }
  }

  return (
    <div className="border rounded bg-white">
      <button
        type="button"
        onClick={toggle}
        className="w-full flex items-center justify-between px-4 py-2 text-left hover:bg-gray-50"
      >
        <span className="text-sm font-medium">
          📊 Cross-page measurement summary
          {data ? (
            <span className="text-gray-500 font-normal ml-2">
              · {data.page_count} fence pages · {data.category_count}{" "}
              categories
            </span>
          ) : null}
        </span>
        <span className="text-xs text-gray-500">
          {open ? "hide" : "show"}
        </span>
      </button>
      {open && (
        <div className="px-4 pb-4 pt-1 space-y-3 border-t">
          {loading && (
            <div className="text-xs text-gray-500 py-3">
              Computing summary across all pages…
            </div>
          )}
          {error && (
            <div className="text-xs text-red-600 break-all py-3">
              {error}
            </div>
          )}
          {data && (
            <>
              <div className="flex items-center justify-between">
                <div className="text-xs text-gray-600">
                  Includes manual UMT edits where saved; otherwise uses
                  pipeline-detected lines.
                </div>
                <button
                  type="button"
                  onClick={() => void fetchSummary()}
                  className="text-xs text-blue-600 hover:underline"
                  disabled={loading}
                >
                  🔄 Refresh
                </button>
              </div>
              <GrandTable data={data} />
              <PerPageBreakdown data={data} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

function GrandTable({ data }: { data: SummaryResponse }) {
  const cats = Object.entries(data.grand_total);
  if (cats.length === 0) {
    return (
      <div className="text-xs text-gray-500 py-3">
        No measurements computed yet — assign lines to categories on the
        per-page measurement canvases below to populate this summary.
      </div>
    );
  }
  const totalFt = cats.reduce((s, [, v]) => s + v.ft, 0);
  const totalAuto = cats.reduce((s, [, v]) => s + v.auto, 0);
  const totalManual = cats.reduce((s, [, v]) => s + v.manual, 0);
  return (
    <div className="overflow-x-auto border rounded">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
          <tr>
            <th className="text-left px-3 py-1.5 font-medium">Category</th>
            <th className="text-right px-3 py-1.5 font-medium">Length (ft)</th>
            <th className="text-right px-3 py-1.5 font-medium">Auto</th>
            <th className="text-right px-3 py-1.5 font-medium">Manual</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {cats.map(([name, v], i) => {
            const c = colorFor(i);
            return (
              <tr key={name} className="hover:bg-gray-50">
                <td className="px-3 py-1.5 align-top">
                  <span className="inline-flex items-center gap-2">
                    <span
                      className="inline-block w-3 h-3 rounded-sm border border-gray-300 shrink-0"
                      style={{
                        backgroundColor: `rgb(${c[0]},${c[1]},${c[2]})`,
                      }}
                    />
                    <span className="font-mono text-xs">{name}</span>
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums font-medium">
                  {v.ft.toFixed(1)}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums text-gray-600">
                  {v.auto}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums text-gray-600">
                  {v.manual}
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot className="bg-gray-50 border-t font-medium">
          <tr>
            <td className="px-3 py-1.5">Grand total</td>
            <td className="px-3 py-1.5 text-right tabular-nums">
              {totalFt.toFixed(1)}
            </td>
            <td className="px-3 py-1.5 text-right tabular-nums">{totalAuto}</td>
            <td className="px-3 py-1.5 text-right tabular-nums">
              {totalManual}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function PerPageBreakdown({ data }: { data: SummaryResponse }) {
  const grandKeys = Object.keys(data.grand_total);
  if (grandKeys.length === 0) return null;
  return (
    <details>
      <summary className="cursor-pointer text-xs text-gray-700 select-none py-1">
        Per-page breakdown ({data.pages.length} pages)
      </summary>
      <div className="mt-2 overflow-x-auto border rounded">
        <table className="w-full text-xs">
          <thead className="bg-gray-50 text-gray-500 uppercase">
            <tr>
              <th className="text-left px-2 py-1 font-medium">Page</th>
              <th className="text-right px-2 py-1 font-medium">Scale</th>
              <th className="text-left px-2 py-1 font-medium">Source</th>
              {grandKeys.map((name, i) => {
                const c = colorFor(i);
                return (
                  <th
                    key={name}
                    className="text-right px-2 py-1 font-medium font-mono"
                    title={name}
                  >
                    <span className="inline-flex items-center gap-1 justify-end">
                      <span
                        className="inline-block w-2 h-2 rounded-sm border border-gray-300 shrink-0"
                        style={{
                          backgroundColor: `rgb(${c[0]},${c[1]},${c[2]})`,
                        }}
                      />
                      <span className="text-[10px]">
                        {name.split(":")[0] || name.slice(0, 6)}
                      </span>
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y">
            {data.pages.map((p) => (
              <tr key={p.page_num} className="hover:bg-gray-50">
                <td className="px-2 py-1 font-mono">Page {p.page_num}</td>
                <td className="px-2 py-1 text-right tabular-nums text-gray-500">
                  1:{p.scale}
                </td>
                <td className="px-2 py-1 text-gray-500">
                  {p.has_user_edits ? "manual" : "auto"}
                </td>
                {grandKeys.map((name) => {
                  const v = p.per_category[name];
                  return (
                    <td
                      key={name}
                      className="px-2 py-1 text-right tabular-nums"
                    >
                      {v ? v.ft.toFixed(1) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
