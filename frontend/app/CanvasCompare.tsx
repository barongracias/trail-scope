"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { InferStats } from "@/lib/api";

// Composites input_8bit.png with the model-confidence heatmap (prob.png), the predicted
// mask (recoloured), the Hough segments, and an optional component highlight onto one
// <canvas>. Layers toggle independently; an opacity slider scales the overlays. Zoom/pan
// via CSS transform; the cursor reports the raw model probability sampled from prob.png.
// All images are fetched as blobs (createImageBitmap) so the canvas stays CORS-clean.

const MASK_RGB: [number, number, number] = [255, 47, 146];
const HOUGH_RGB = "rgb(0, 200, 255)";
const HIGHLIGHT_RGB = "rgb(250, 204, 21)";

export type ProbHover = { x: number; y: number; p: number } | null;

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

function tintMask(bmp: ImageBitmap, rgb: [number, number, number]): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  const id = ctx.getImageData(0, 0, c.width, c.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] > 127) {
      d[i] = rgb[0];
      d[i + 1] = rgb[1];
      d[i + 2] = rgb[2];
      d[i + 3] = 255;
    } else {
      d[i + 3] = 0;
    }
  }
  ctx.putImageData(id, 0, 0);
  return c;
}

// Colourmap the grayscale confidence (blue→red); alpha scales with confidence so prob≈0
// is transparent. Returns both the coloured layer and the raw 0–255 grayscale for readout.
function colourmapProb(bmp: ImageBitmap): { layer: HTMLCanvasElement; gray: Uint8ClampedArray; w: number; h: number } {
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  const id = ctx.getImageData(0, 0, c.width, c.height);
  const d = id.data;
  const gray = new Uint8ClampedArray((d.length / 4) | 0);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    const g = d[i];
    gray[j] = g;
    const v = g / 255;
    const [r, gg, b] = hslToRgb((1 - v) * 240, 0.9, 0.5);
    d[i] = r;
    d[i + 1] = gg;
    d[i + 2] = b;
    d[i + 3] = Math.round(v * 230);
  }
  ctx.putImageData(id, 0, 0);
  return { layer: c, gray, w: c.width, h: c.height };
}

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${url}`);
  return createImageBitmap(await res.blob());
}

export default function CanvasCompare({
  inputUrl,
  maskUrl,
  probUrl,
  stats,
  showMask,
  showHough,
  showProb,
  opacity,
  highlight,
  onProbHover,
}: {
  inputUrl: string;
  maskUrl: string;
  probUrl: string;
  stats: InferStats;
  showMask: boolean;
  showHough: boolean;
  showProb: boolean;
  opacity: number;
  highlight: number | null;
  onProbHover?: (h: ProbHover) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [base, setBase] = useState<ImageBitmap | null>(null);
  const [maskTint, setMaskTint] = useState<HTMLCanvasElement | null>(null);
  const [prob, setProb] = useState<ReturnType<typeof colourmapProb> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 });
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [b, m, p] = await Promise.all([
          loadBitmap(inputUrl),
          loadBitmap(maskUrl),
          loadBitmap(probUrl),
        ]);
        if (cancelled) return;
        setBase(b);
        setMaskTint(tintMask(m, MASK_RGB));
        setProb(colourmapProb(p));
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : "Failed to render overlay");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [inputUrl, maskUrl, probUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !base) return;
    const [h, w] = stats.image.processed_shape;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);
    ctx.globalAlpha = 1;
    ctx.drawImage(base, 0, 0, w, h);

    ctx.globalAlpha = opacity;
    if (showProb && prob) ctx.drawImage(prob.layer, 0, 0, w, h);
    if (showHough && stats.hough.enabled) {
      ctx.strokeStyle = HOUGH_RGB;
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      for (const [x1, y1, x2, y2] of stats.hough.segments) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    if (showMask && maskTint) ctx.drawImage(maskTint, 0, 0, w, h);
    ctx.globalAlpha = 1;

    if (highlight != null) {
      const comp = stats.model_output.predicted_components.find((c) => c.index === highlight);
      if (comp && comp.bbox.length === 4) {
        const [x, y, bw, bh] = comp.bbox;
        ctx.strokeStyle = HIGHLIGHT_RGB;
        ctx.lineWidth = Math.max(2, Math.round(Math.max(w, h) / 300));
        ctx.strokeRect(x, y, bw, bh);
      }
    }
  }, [base, maskTint, prob, showMask, showHough, showProb, opacity, highlight, stats]);

  const toImagePx = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * canvas.width);
    const y = Math.floor(((clientY - rect.top) / rect.height) * canvas.height);
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
    return { x, y };
  }, []);

  const onMove = (e: React.MouseEvent) => {
    if (drag.current) {
      setView((v) => ({ ...v, tx: drag.current!.tx + (e.clientX - drag.current!.x), ty: drag.current!.ty + (e.clientY - drag.current!.y) }));
      return;
    }
    if (!onProbHover || !prob) return;
    const px = toImagePx(e.clientX, e.clientY);
    if (!px) return onProbHover(null);
    const g = prob.gray[px.y * prob.w + px.x];
    onProbHover({ x: px.x, y: px.y, p: g / 255 });
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setView((v) => {
      const scale = Math.min(8, Math.max(1, v.scale * factor));
      const rect = canvasRef.current!.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const k = scale / v.scale;
      return { scale, tx: v.tx - cx * (k - 1), ty: v.ty - cy * (k - 1) };
    });
  };

  if (err) return <div className="text-sm text-red-600">Overlay error: {err}</div>;

  return (
    <div
      className="relative overflow-hidden rounded-lg border border-slate-300 bg-black"
      onWheel={onWheel}
      onMouseLeave={() => {
        drag.current = null;
        onProbHover?.(null);
      }}
    >
      <canvas
        ref={canvasRef}
        className="w-full h-auto cursor-crosshair"
        style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`, transformOrigin: "0 0" }}
        onMouseDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
        }}
        onMouseUp={() => {
          drag.current = null;
        }}
        onMouseMove={onMove}
        onDoubleClick={() => setView({ scale: 1, tx: 0, ty: 0 })}
      />
      {view.scale > 1 && (
        <button
          onClick={() => setView({ scale: 1, tx: 0, ty: 0 })}
          className="absolute right-2 top-2 rounded bg-white/85 px-2 py-1 text-xs font-medium text-slate-700 shadow"
        >
          Reset view
        </button>
      )}
    </div>
  );
}
