"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Download, Loader2, Upload } from "lucide-react";
import {
  ApiError,
  type InferResponse,
  type Tier,
  healthCheck,
  infer,
  resultUrl,
} from "@/lib/api";
import CanvasCompare from "./CanvasCompare";

// Locked, non-tunable model facts (constants, never knobs).
const MODEL_CARD = [
  ["Model", "Locked thesis U-Net"],
  ["Threshold", "0.45"],
  ["Patch", "528 × 528"],
  ["Training domain", "MeerLICHT 8-bit display PNG patches"],
];

const SCOPE =
  "A qualitative, single-image inference demo of the locked thesis detector. Not a benchmark, not a validated cross-domain tool, no training, no tunable thresholds. Qualitative inference only; no performance claims are made for uploaded images.";

const DISCLAIMER =
  "This is not a validated detector for this input unless it is from the original MeerLICHT-style domain.";

const ACCEPTED = ".fits, .fit, .fits.fz, .png, .jpg, .jpeg, .tif";

const TIER_TEXT: Record<Tier, string> = {
  in_domain_like:
    "In-domain-like input — an 8-bit display image at a plausible scale.",
  recipe_matched:
    "Recipe-matched input — FITS with a header-resolved pixel scale (the validated DECam-style recipe).",
  best_effort:
    "Best-effort input — outside the validated recipe (e.g. unknown pixel scale).",
};

// Small cropped 8-bit PNG examples derived from the public DECam frames (NOIRLab).
const DEMOS = [
  { label: "DECam — NAVSTAR-70 (crop)", file: "decam_navstar70_crop.png" },
];

type Phase = "input" | "processing" | "output";

function fmt(n: number, digits = 0): string {
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export default function Page() {
  const [phase, setPhase] = useState<Phase>("input");
  const [file, setFile] = useState<File | null>(null);
  const [hough, setHough] = useState(true);
  const [pixelScale, setPixelScale] = useState<string>("");
  const [hduIndex, setHduIndex] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState("Preparing image");
  const [result, setResult] = useState<InferResponse | null>(null);
  const [backendDown, setBackendDown] = useState(false);

  // Output overlay controls.
  const [showMask, setShowMask] = useState(true);
  const [showHough, setShowHough] = useState(true);
  const [opacity, setOpacity] = useState(0.85);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const stageTimers = useRef<number[]>([]);

  useEffect(() => {
    healthCheck()
      .then((h) => setBackendDown(!h.model_sha_ok))
      .catch(() => setBackendDown(true));
  }, []);

  const pickDemo = async (demoFile: string, label: string) => {
    setError(null);
    try {
      const res = await fetch(`/demo/${demoFile}`);
      if (!res.ok) throw new Error("Demo asset not found");
      const blob = await res.blob();
      setFile(new File([blob], demoFile, { type: "image/png" }));
    } catch {
      setError(`Could not load demo "${label}".`);
    }
  };

  const onRun = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setShowHough(hough);
    setPhase("processing");
    // Honest staged text (the request is synchronous; these are indicative).
    setStage("Preparing image");
    stageTimers.current.forEach(clearTimeout);
    stageTimers.current = [
      window.setTimeout(() => setStage("Running locked U-Net"), 600),
      window.setTimeout(() => setStage("Rendering outputs"), 2500),
    ];
    try {
      const res = await infer(file, {
        hough,
        pixelScaleArcsec: pixelScale ? Number(pixelScale) : null,
        hduIndex: hduIndex ? Number(hduIndex) : null,
      });
      setResult(res);
      setPhase("output");
    } catch (e) {
      const msg =
        e instanceof ApiError ? e.message : "Inference failed. Please try again.";
      setError(msg);
      setPhase("input");
    } finally {
      stageTimers.current.forEach(clearTimeout);
    }
  };

  const reset = () => {
    setResult(null);
    setFile(null);
    setError(null);
    setPhase("input");
  };

  return (
    <main className="min-h-screen max-w-5xl mx-auto px-4 py-8 text-slate-800">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">trail-scope</h1>
        <p className="mt-1 text-sm text-slate-600 max-w-3xl">{SCOPE}</p>
      </header>

      {backendDown && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>
            The backend is unreachable or its checkpoint failed the integrity gate.
            Inference is disabled.
          </span>
        </div>
      )}

      {phase === "input" && (
        <InputView
          file={file}
          setFile={setFile}
          hough={hough}
          setHough={setHough}
          pixelScale={pixelScale}
          setPixelScale={setPixelScale}
          hduIndex={hduIndex}
          setHduIndex={setHduIndex}
          dragging={dragging}
          setDragging={setDragging}
          fileInputRef={fileInputRef}
          error={error}
          onRun={onRun}
          pickDemo={pickDemo}
        />
      )}

      {phase === "processing" && <ProcessingView filename={file?.name ?? ""} stage={stage} />}

      {phase === "output" && result && (
        <OutputView
          result={result}
          showMask={showMask}
          setShowMask={setShowMask}
          showHough={showHough}
          setShowHough={setShowHough}
          opacity={opacity}
          setOpacity={setOpacity}
          onReset={reset}
        />
      )}

      <footer className="mt-10 border-t border-slate-200 pt-4 text-xs text-slate-500">
        <p className="font-medium text-slate-600">{DISCLAIMER}</p>
        <p className="mt-1">
          Demo data: public DECam frames. Based on observations at Cerro Tololo
          Inter-American Observatory, NSF&apos;s NOIRLab. No MeerLICHT imagery is
          distributed. Stats use &quot;predicted mask/component&quot; language and make no
          accuracy claims.
        </p>
      </footer>
    </main>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {children}
    </section>
  );
}

function InputView(props: {
  file: File | null;
  setFile: (f: File | null) => void;
  hough: boolean;
  setHough: (b: boolean) => void;
  pixelScale: string;
  setPixelScale: (s: string) => void;
  hduIndex: string;
  setHduIndex: (s: string) => void;
  dragging: boolean;
  setDragging: (b: boolean) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  error: string | null;
  onRun: () => void;
  pickDemo: (f: string, label: string) => void;
}) {
  const {
    file, setFile, hough, setHough, pixelScale, setPixelScale, hduIndex, setHduIndex,
    dragging, setDragging, fileInputRef, error, onRun, pickDemo,
  } = props;

  return (
    <div className="space-y-5">
      <Card>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition ${
            dragging ? "border-blue-400 bg-blue-50" : "border-slate-300 hover:border-slate-400"
          }`}
        >
          <Upload className="mb-2 h-7 w-7 text-slate-400" />
          <p className="text-sm font-medium text-slate-700">
            {file ? file.name : "Drag an image here, or click to browse"}
          </p>
          <p className="mt-1 text-xs text-slate-500">Accepted: {ACCEPTED} · 64 MB max</p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".fits,.fit,.fz,.png,.jpg,.jpeg,.tif,.tiff"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-slate-500">Or try a demo:</span>
          {DEMOS.map((d) => (
            <button
              key={d.file}
              onClick={() => pickDemo(d.file, d.label)}
              className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-700 hover:bg-slate-100"
            >
              {d.label}
            </button>
          ))}
        </div>
      </Card>

      <Card>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">Options</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          <label className="flex flex-col gap-1 text-xs text-slate-600">
            Pixel scale override (arcsec/px)
            <input
              type="number"
              step="0.0001"
              min="0"
              placeholder="optional"
              value={pixelScale}
              onChange={(e) => setPixelScale(e.target.value)}
              className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-600">
            FITS HDU index
            <input
              type="number"
              step="1"
              min="0"
              placeholder="optional"
              value={hduIndex}
              onChange={(e) => setHduIndex(e.target.value)}
              className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
            />
          </label>
          <label className="flex items-end gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={hough}
              onChange={(e) => setHough(e.target.checked)}
              className="h-4 w-4"
            />
            Hough overlay
          </label>
        </div>
      </Card>

      <Card className="bg-slate-50">
        <h2 className="mb-2 text-sm font-semibold text-slate-700">Locked model card</h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
          {MODEL_CARD.map(([k, v]) => (
            <div key={k}>
              <dt className="text-slate-500">{k}</dt>
              <dd className="font-medium text-slate-800">{v}</dd>
            </div>
          ))}
        </dl>
      </Card>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-slate-500">
          The model and threshold are fixed; this run does not tune parameters or estimate
          accuracy.
        </p>
        <button
          onClick={onRun}
          disabled={!file}
          className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Run inference
        </button>
      </div>
    </div>
  );
}

function ProcessingView({ filename, stage }: { filename: string; stage: string }) {
  return (
    <Card className="flex flex-col items-center gap-4 py-12 text-center">
      <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      <div>
        <p className="text-sm font-medium text-slate-700">{filename}</p>
        <p className="mt-1 text-sm text-slate-500">{stage}…</p>
      </div>
      <p className="max-w-md text-xs text-slate-400">
        Large images may take up to a minute on CPU. Images over 64 patches are rejected in
        this demo.
      </p>
      <p className="max-w-md text-xs text-slate-400">
        The model and threshold are fixed; this run does not tune parameters or estimate
        accuracy.
      </p>
    </Card>
  );
}

function OutputView(props: {
  result: InferResponse;
  showMask: boolean;
  setShowMask: (b: boolean) => void;
  showHough: boolean;
  setShowHough: (b: boolean) => void;
  opacity: number;
  setOpacity: (n: number) => void;
  onReset: () => void;
}) {
  const { result, showMask, setShowMask, showHough, setShowHough, opacity, setOpacity, onReset } =
    props;
  const { result_id, stats } = result;
  const mo = stats.model_output;

  return (
    <div className="space-y-5">
      {/* Neutral tier banner (same blue/grey style for all tiers; text differs). */}
      <div className="rounded-xl border border-blue-200 bg-blue-50 px-5 py-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
          Tier: {stats.tier.replace(/_/g, " ")}
        </p>
        <p className="mt-1 text-sm text-slate-700">{TIER_TEXT[stats.tier]}</p>
        <p className="mt-1 text-sm font-medium text-slate-700">{DISCLAIMER}</p>
        {stats.warnings.length > 0 && (
          <ul className="mt-2 list-disc pl-5 text-xs text-amber-700">
            {stats.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        )}
      </div>

      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "rgb(255,47,146)" }} />
            Predicted mask
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={showHough}
              onChange={(e) => setShowHough(e.target.checked)}
              disabled={!stats.hough.enabled}
            />
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "rgb(0,200,255)" }} />
            Hough overlay {stats.hough.enabled ? "" : "(off)"}
          </label>
          <label className="ml-auto flex items-center gap-2 text-xs text-slate-600">
            Overlay opacity
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <figure>
            <figcaption className="mb-1 text-xs text-slate-500">
              Model input (input_8bit.png — exactly what the model saw)
            </figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={resultUrl(result_id, "input_8bit.png")}
              alt="Model input"
              className="w-full rounded-lg border border-slate-300 bg-black"
            />
          </figure>
          <figure>
            <figcaption className="mb-1 text-xs text-slate-500">Overlay</figcaption>
            <CanvasCompare
              inputUrl={resultUrl(result_id, "input_8bit.png")}
              maskUrl={resultUrl(result_id, "mask.png")}
              stats={stats}
              showMask={showMask}
              showHough={showHough}
              opacity={opacity}
            />
          </figure>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h2 className="mb-3 text-sm font-semibold text-slate-700">Result summary</h2>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <Stat label="Predicted mask pixels" value={fmt(mo.predicted_mask_pixel_count)} />
            <Stat label="Predicted mask fraction" value={mo.predicted_mask_fraction.toExponential(2)} />
            <Stat label="Predicted components" value={fmt(mo.predicted_component_count)} />
            <Stat label="Max model probability" value={mo.max_model_probability.toFixed(4)} />
            <Stat label="Hough segments" value={fmt(stats.hough.segment_count)} />
            <Stat label="Processed shape" value={stats.image.processed_shape.join(" × ")} />
            <Stat label="Patches" value={fmt(stats.image.n_patches)} />
            <Stat
              label="Runtime"
              value={`${fmt(
                stats.timing_ms.preprocess + stats.timing_ms.inference + stats.timing_ms.hough,
              )} ms`}
            />
          </dl>
        </Card>

        <Card>
          <h2 className="mb-3 text-sm font-semibold text-slate-700">Provenance</h2>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <Stat label="Format" value={stats.provenance.format} small />
            <Stat label="HDU" value={stats.provenance.hdu ?? "—"} small />
            <Stat label="Stretch" value={stats.provenance.stretch} small />
            <Stat label="Stretch fallback" value={String(stats.provenance.stretch_fallback)} small />
            <Stat label="Pixel-scale source" value={stats.provenance.pixel_scale_source} small />
            <Stat
              label="Pixel scale"
              value={stats.provenance.pixel_scale_arcsec != null ? `${stats.provenance.pixel_scale_arcsec} "/px` : "—"}
              small
            />
            <Stat label="Resample factor" value={stats.provenance.resample_factor.toFixed(4)} small />
            <Stat label="Threshold" value={stats.provenance.threshold.toString()} small />
            <Stat
              label="Checkpoint SHA"
              value={`${stats.provenance.checkpoint_sha256.slice(0, 12)}…`}
              small
            />
            <Stat
              label="Vendored commit"
              value={`${stats.provenance.vendored_source_commit.slice(0, 12)}…`}
              small
            />
          </dl>
        </Card>
      </div>

      <Card>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">
          Predicted components ({mo.predicted_component_count})
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3">#</th>
                <th className="py-1 pr-3">Pixels</th>
                <th className="py-1 pr-3">bbox [x,y,w,h]</th>
                <th className="py-1 pr-3">Major axis (px)</th>
                <th className="py-1 pr-3">Orientation (°)</th>
              </tr>
            </thead>
            <tbody>
              {mo.predicted_components.slice(0, 50).map((c) => (
                <tr key={c.index} className="border-t border-slate-100">
                  <td className="py-1 pr-3">{c.index}</td>
                  <td className="py-1 pr-3">{fmt(c.pixel_count)}</td>
                  <td className="py-1 pr-3">[{c.bbox.join(", ")}]</td>
                  <td className="py-1 pr-3">{c.major_axis_px != null ? c.major_axis_px.toFixed(1) : "—"}</td>
                  <td className="py-1 pr-3">{c.orientation_deg != null ? c.orientation_deg.toFixed(1) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {mo.predicted_components.length === 0 && (
            <p className="py-2 text-xs text-slate-500">No predicted components.</p>
          )}
        </div>
      </Card>

      <Card>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">Downloads</h2>
        <div className="flex flex-wrap gap-2">
          {([
            ["overlay.png", "Overlay"],
            ["mask.png", "Mask"],
            ["input_8bit.png", "Model input"],
            ["stats.json", "Stats JSON"],
          ] as const).map(([f, label]) => (
            <a
              key={f}
              href={resultUrl(result_id, f)}
              download
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
            >
              <Download className="h-3.5 w-3.5" />
              {label}
            </a>
          ))}
        </div>
      </Card>

      <button
        onClick={onReset}
        className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
      >
        Run another image
      </button>
    </div>
  );
}

function Stat({
  label,
  value,
  small = false,
}: {
  label: string;
  value: string | number;
  small?: boolean;
}) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className={`font-medium text-slate-800 ${small ? "" : "text-base"}`}>{value}</dd>
    </div>
  );
}
