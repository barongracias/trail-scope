"use client";

import { useEffect, useRef, useState } from "react";

// Rubber-band crop for the 413 ("too many patches") path: the user drags a rectangle
// over a browser-renderable image; "Crop & run" exports that region as a PNG and resubmits.
// Only offered for PNG/JPEG (the browser can't render FITS into a canvas).

type Rect = { x: number; y: number; w: number; h: number };

export default function CropView({
  file,
  onCropped,
  onCancel,
}: {
  file: File;
  onCropped: (cropped: File) => void;
  onCancel: () => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [url, setUrl] = useState<string>("");
  const [rect, setRect] = useState<Rect | null>(null);
  const start = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  const rel = (e: React.MouseEvent) => {
    const r = imgRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, rw: r.width, rh: r.height };
  };

  const crop = () => {
    const img = imgRef.current!;
    const r = img.getBoundingClientRect();
    const sel = rect ?? { x: 0, y: 0, w: r.width, h: r.height };
    const sx = (sel.x / r.width) * img.naturalWidth;
    const sy = (sel.y / r.height) * img.naturalHeight;
    const sw = (sel.w / r.width) * img.naturalWidth;
    const sh = (sel.h / r.height) * img.naturalHeight;
    if (sw < 8 || sh < 8) return;
    const c = document.createElement("canvas");
    c.width = Math.round(sw);
    c.height = Math.round(sh);
    c.getContext("2d")!.drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
    c.toBlob((blob) => {
      if (blob) onCropped(new File([blob], `crop_${file.name.replace(/\.[^.]+$/, "")}.png`, { type: "image/png" }));
    }, "image/png");
  };

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        Drag a box to select a smaller region (≤ 64 patches), then run again. Leave empty to
        use the whole image.
      </p>
      <div className="relative inline-block select-none">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={url}
          alt="Crop source"
          className="max-h-[460px] w-auto rounded border border-slate-300"
          draggable={false}
          onMouseDown={(e) => {
            const { x, y } = rel(e);
            start.current = { x, y };
            setRect({ x, y, w: 0, h: 0 });
          }}
          onMouseMove={(e) => {
            if (!start.current) return;
            const { x, y } = rel(e);
            const s = start.current;
            setRect({ x: Math.min(s.x, x), y: Math.min(s.y, y), w: Math.abs(x - s.x), h: Math.abs(y - s.y) });
          }}
          onMouseUp={() => {
            start.current = null;
          }}
        />
        {rect && rect.w > 2 && rect.h > 2 && (
          <div
            className="pointer-events-none absolute border-2 border-blue-400 bg-blue-400/20"
            style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
          />
        )}
      </div>
      <div className="flex gap-2">
        <button
          onClick={crop}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700"
        >
          Crop &amp; run
        </button>
        <button
          onClick={onCancel}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
