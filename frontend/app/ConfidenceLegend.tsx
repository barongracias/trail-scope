"use client";

import { confidenceColor } from "./CanvasCompare";

// Colourbar for the model-confidence heatmap, using the exact same blue→red colourmap as
// the canvas so the legend is faithful. Qualitative, not a tunable threshold.
export default function ConfidenceLegend() {
  const stops = [0, 0.2, 0.4, 0.6, 0.8, 1]
    .map((v) => `${confidenceColor(v)} ${Math.round(v * 100)}%`)
    .join(", ");
  return (
    <div className="flex items-center gap-2" aria-hidden="true">
      <span className="text-[10px] text-slate-500">model confidence</span>
      <div className="flex flex-col">
        <div
          className="h-2 w-32 rounded"
          style={{ background: `linear-gradient(to right, ${stops})` }}
        />
        <div className="flex w-32 justify-between text-[9px] text-slate-500">
          <span>0</span>
          <span>0.5</span>
          <span>1</span>
        </div>
      </div>
    </div>
  );
}
