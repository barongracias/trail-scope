"use client";

import { useEffect, useRef, useState } from "react";
import type { InferStats } from "@/lib/api";

// Composites input_8bit.png with the predicted mask (recoloured) and the Hough
// segments on a single <canvas>, so Mask/Hough toggles and the opacity slider work
// independently — all from the four committed result files + stats.json. Images are
// fetched as blobs (createImageBitmap) so the canvas stays CORS-clean.

const MASK_RGB: [number, number, number] = [255, 47, 146];
const HOUGH_RGB = "rgb(0, 200, 255)";

function tintBitmap(bmp: ImageBitmap, rgb: [number, number, number]): HTMLCanvasElement {
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

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${url}`);
  return createImageBitmap(await res.blob());
}

export default function CanvasCompare({
  inputUrl,
  maskUrl,
  stats,
  showMask,
  showHough,
  opacity,
}: {
  inputUrl: string;
  maskUrl: string;
  stats: InferStats;
  showMask: boolean;
  showHough: boolean;
  opacity: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [base, setBase] = useState<ImageBitmap | null>(null);
  const [maskTint, setMaskTint] = useState<HTMLCanvasElement | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [b, m] = await Promise.all([loadBitmap(inputUrl), loadBitmap(maskUrl)]);
        if (cancelled) return;
        setBase(b);
        setMaskTint(tintBitmap(m, MASK_RGB));
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : "Failed to render overlay");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [inputUrl, maskUrl]);

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
    if (showMask && maskTint) ctx.drawImage(maskTint, 0, 0, w, h);
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
    ctx.globalAlpha = 1;
  }, [base, maskTint, showMask, showHough, opacity, stats]);

  if (err) {
    return <div className="text-sm text-red-600">Overlay error: {err}</div>;
  }
  return (
    <canvas
      ref={canvasRef}
      className="w-full h-auto rounded-lg border border-slate-300 bg-black"
    />
  );
}
