"use client";

import dynamic from "next/dynamic";

// Dynamic-import the whole konva-using component as a single module.
// This avoids the named-export resolution issues we hit with per-component
// dynamic() calls on react-konva's Stage/Layer/Image/Line.
const UMTCanvasInner = dynamic(() => import("./UMTCanvasInner"), {
  ssr: false,
  loading: () => (
    <div className="text-xs text-gray-500">Loading measurement canvas…</div>
  ),
});

export function UMTCanvas(props: {
  jobId: string;
  pageNum: number;
  legendEntries: Array<{ indicator?: string; keyword?: string }>;
  initiallyOpen?: boolean;
  skipReason?: string | null;
  scaleOverride?: number | null;
  onSaved?: () => void;
}) {
  return <UMTCanvasInner {...props} />;
}
