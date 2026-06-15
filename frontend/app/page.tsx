"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Copy, Download, Loader2, Upload } from "lucide-react";
import {
  ApiError,
  type FitsHdu,
  type InferResponse,
  type Tier,
  createJob,
  getJobStatus,
  healthCheck,
  infer,
  inspectFits,
  resultUrl,
} from "@/lib/api";
import CanvasCompare, { type ProbHover } from "./CanvasCompare";
import CropView from "./CropView";

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
  in_domain_like: "In-domain-like input — an 8-bit display image at a plausible scale.",
  recipe_matched:
    "Recipe-matched input — FITS with a header-resolved pixel scale (the validated DECam-style recipe).",
  best_effort: "Best-effort input — outside the validated recipe (e.g. unknown pixel scale).",
};
const TIER_DEFS =
  "in_domain_like: 8-bit display image at a plausible scale.\nrecipe_matched: FITS with a header-resolved pixel scale (the validated DECam recipe).\nbest_effort: everything else (e.g. unknown pixel scale — the model is not scale-invariant).";

const DEMOS = [{ label: "DECam — NAVSTAR-70 (crop)", file: "decam_navstar70_crop.png" }];

type Phase = "input" | "processing" | "output";

const fmt = (n: number, digits = 0) =>
  n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

const isRaster = (f: File | null) => !!f && /\.(png|jpe?g)$/i.test(f.name);
const isFits = (f: File | null) => !!f && /\.(fits|fit|fits\.fz|fz)$/i.test(f.name);

export default function Page() {
  const [phase, setPhase] = useState<Phase>("input");
  const [file, setFile] = useState<File | null>(null);
  const [hough, setHough] = useState(true);
  const [largeMode, setLargeMode] = useState(false);
  const [pixelScale, setPixelScale] = useState("");
  const [hduIndex, setHduIndex] = useState("");
  const [hduList, setHduList] = useState<FitsHdu[] | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reject413, setReject413] = useState<string | null>(null);
  const [cropping, setCropping] = useState(false);
  const [stage, setStage] = useState("Preparing image");
  const [result, setResult] = useState<InferResponse | null>(null);
  const [backendDown, setBackendDown] = useState(false);

  // Output overlay controls.
  const [showMask, setShowMask] = useState(true);
  const [showHough, setShowHough] = useState(true);
  const [showProb, setShowProb] = useState(false);
  const [opacity, setOpacity] = useState(0.85);
  const [highlight, setHighlight] = useState<number | null>(null);
  const [probHover, setProbHover] = useState<ProbHover>(null);
  const [compareHough, setCompareHough] = useState(false);
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const stageTimers = useRef<number[]>([]);

  useEffect(() => {
    healthCheck().then((h) => setBackendDown(!h.model_sha_ok)).catch(() => setBackendDown(true));
  }, []);

  // Paste-from-clipboard upload.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = e.clipboardData?.files?.[0];
      if (f) chooseFile(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  const chooseFile = (f: File | null) => {
    setFile(f);
    setError(null);
    setReject413(null);
    setCropping(false);
    setHduList(null);
    setHduIndex("");
    if (f && isFits(f)) {
      inspectFits(f)
        .then((r) => setHduList(r.hdus))
        .catch(() => setHduList(null));
    }
  };

  const pickDemo = async (demoFile: string, label: string) => {
    try {
      const res = await fetch(`/demo/${demoFile}`);
      if (!res.ok) throw new Error("Demo asset not found");
      chooseFile(new File([await res.blob()], demoFile, { type: "image/png" }));
    } catch {
      setError(`Could not load demo "${label}".`);
    }
  };

  const handleFailure = (apiErr: ApiError | null, runFile: File) => {
    if (apiErr?.status === 413) {
      setReject413(apiErr.message);
      setCropping(isRaster(runFile));
    } else {
      setError(apiErr ? apiErr.message : "Inference failed. Please try again.");
    }
    setPhase("input");
  };

  const run = async (runFile: File) => {
    setError(null);
    setReject413(null);
    setResult(null);
    setShowHough(hough);
    setHighlight(null);
    setPhase("processing");
    setStage("Preparing image");
    const opts = {
      hough,
      pixelScaleArcsec: pixelScale ? Number(pixelScale) : null,
      hduIndex: hduIndex ? Number(hduIndex) : null,
    };

    if (largeMode) {
      // Async path: submit a job and poll its status (full-frame, > 64 patches).
      try {
        const { job_id } = await createJob(runFile, opts);
        for (let i = 0; i < 1200; i++) {
          const st = await getJobStatus(job_id);
          setStage(st.n_patches ? `${st.detail} · ${st.n_patches} patches` : st.detail);
          if (st.state === "done" && st.result_id && st.stats) {
            setResult({ result_id: st.result_id, stats: st.stats });
            setPhase("output");
            return;
          }
          if (st.state === "error") {
            handleFailure(new ApiError(st.error ?? "Job failed", st.status_code ?? undefined), runFile);
            return;
          }
          await new Promise((r) => setTimeout(r, 700));
        }
        handleFailure(new ApiError("Timed out waiting for the job."), runFile);
      } catch (e) {
        handleFailure(e instanceof ApiError ? e : null, runFile);
      }
      return;
    }

    // Synchronous path (default, ≤ 64 patches).
    stageTimers.current.forEach(clearTimeout);
    stageTimers.current = [
      window.setTimeout(() => setStage("Running locked U-Net"), 600),
      window.setTimeout(() => setStage("Rendering outputs"), 2500),
    ];
    try {
      const res = await infer(runFile, opts);
      setResult(res);
      setPhase("output");
    } catch (e) {
      handleFailure(e instanceof ApiError ? e : null, runFile);
    } finally {
      stageTimers.current.forEach(clearTimeout);
    }
  };

  const reset = () => {
    setResult(null);
    setFile(null);
    setError(null);
    setReject413(null);
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
          <span>The backend is unreachable or its checkpoint failed the integrity gate. Inference is disabled.</span>
        </div>
      )}

      {phase === "input" && (
        <InputView
          file={file}
          chooseFile={chooseFile}
          hough={hough}
          setHough={setHough}
          largeMode={largeMode}
          setLargeMode={setLargeMode}
          pixelScale={pixelScale}
          setPixelScale={setPixelScale}
          hduIndex={hduIndex}
          setHduIndex={setHduIndex}
          hduList={hduList}
          dragging={dragging}
          setDragging={setDragging}
          fileInputRef={fileInputRef}
          error={error}
          reject413={reject413}
          cropping={cropping}
          setCropping={setCropping}
          onRun={() => file && run(file)}
          onCropped={(f) => {
            setFile(f);
            run(f);
          }}
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
          showProb={showProb}
          setShowProb={setShowProb}
          opacity={opacity}
          setOpacity={setOpacity}
          highlight={highlight}
          setHighlight={setHighlight}
          probHover={probHover}
          setProbHover={setProbHover}
          compareHough={compareHough}
          setCompareHough={setCompareHough}
          copied={copied}
          setCopied={setCopied}
          onReset={reset}
        />
      )}

      <footer className="mt-10 border-t border-slate-200 pt-4 text-xs text-slate-500">
        <p className="font-medium text-slate-600">{DISCLAIMER}</p>
        <p className="mt-1">
          Demo data: public DECam frames. Based on observations at Cerro Tololo Inter-American
          Observatory, NSF&apos;s NOIRLab. No MeerLICHT imagery is distributed. Stats use
          &quot;predicted mask/component&quot; language and make no accuracy claims.
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
  chooseFile: (f: File | null) => void;
  hough: boolean;
  setHough: (b: boolean) => void;
  largeMode: boolean;
  setLargeMode: (b: boolean) => void;
  pixelScale: string;
  setPixelScale: (s: string) => void;
  hduIndex: string;
  setHduIndex: (s: string) => void;
  hduList: FitsHdu[] | null;
  dragging: boolean;
  setDragging: (b: boolean) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  error: string | null;
  reject413: string | null;
  cropping: boolean;
  setCropping: (b: boolean) => void;
  onRun: () => void;
  onCropped: (f: File) => void;
  pickDemo: (f: string, label: string) => void;
}) {
  const {
    file, chooseFile, hough, setHough, largeMode, setLargeMode, pixelScale, setPixelScale,
    hduIndex, setHduIndex, hduList, dragging, setDragging, fileInputRef, error, reject413,
    cropping, setCropping, onRun, onCropped, pickDemo,
  } = props;

  if (cropping && file) {
    return (
      <Card>
        <div className="mb-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800">
          {reject413}
        </div>
        <CropView file={file} onCropped={onCropped} onCancel={() => setCropping(false)} />
      </Card>
    );
  }

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
            if (e.dataTransfer.files?.[0]) chooseFile(e.dataTransfer.files[0]);
          }}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition ${
            dragging ? "border-blue-400 bg-blue-50" : "border-slate-300 hover:border-slate-400"
          }`}
        >
          <Upload className="mb-2 h-7 w-7 text-slate-400" />
          <p className="text-sm font-medium text-slate-700">
            {file ? file.name : "Drag an image here, click to browse, or paste from clipboard"}
          </p>
          <p className="mt-1 text-xs text-slate-500">Accepted: {ACCEPTED} · 64 MB max</p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".fits,.fit,.fits.fz,.png,.jpg,.jpeg,.tif"
            onChange={(e) => chooseFile(e.target.files?.[0] ?? null)}
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
            {hduList && hduList.length > 0 ? (
              <select
                value={hduIndex}
                onChange={(e) => setHduIndex(e.target.value)}
                className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
              >
                <option value="">auto (first 2-D image)</option>
                {hduList.map((h) => (
                  <option key={h.index} value={h.index} disabled={!h.is_2d_image}>
                    [{h.index}] {h.type}
                    {h.shape ? ` ${h.shape.join("×")}` : ""}
                    {h.is_2d_image ? "" : " (not 2-D)"}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                step="1"
                min="0"
                placeholder="optional"
                value={hduIndex}
                onChange={(e) => setHduIndex(e.target.value)}
                className="rounded-md border border-slate-300 px-2 py-1.5 text-sm"
              />
            )}
          </label>
          <label className="flex items-end gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={hough} onChange={(e) => setHough(e.target.checked)} className="h-4 w-4" />
            Hough overlay
          </label>
        </div>
        <label className="mt-4 flex items-start gap-2 text-xs text-slate-600">
          <input
            type="checkbox"
            checked={largeMode}
            onChange={(e) => setLargeMode(e.target.checked)}
            className="mt-0.5 h-4 w-4"
          />
          <span>
            Process large images (over 64 patches) as a background job. Slower on CPU and
            still qualitative — the model, threshold, and recipe are unchanged.
          </span>
        </label>
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

      {reject413 && !cropping && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {reject413}
        </div>
      )}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-slate-500">
          The model and threshold are fixed; this run does not tune parameters or estimate accuracy.
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
        Large images may take up to a minute on CPU. Images over 64 patches are rejected in this demo.
      </p>
      <p className="max-w-md text-xs text-slate-400">
        The model and threshold are fixed; this run does not tune parameters or estimate accuracy.
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
  showProb: boolean;
  setShowProb: (b: boolean) => void;
  opacity: number;
  setOpacity: (n: number) => void;
  highlight: number | null;
  setHighlight: (n: number | null) => void;
  probHover: ProbHover;
  setProbHover: (h: ProbHover) => void;
  compareHough: boolean;
  setCompareHough: (b: boolean) => void;
  copied: boolean;
  setCopied: (b: boolean) => void;
  onReset: () => void;
}) {
  const {
    result, showMask, setShowMask, showHough, setShowHough, showProb, setShowProb,
    opacity, setOpacity, highlight, setHighlight, probHover, setProbHover,
    compareHough, setCompareHough, copied, setCopied, onReset,
  } = props;
  const { result_id, stats } = result;
  const mo = stats.model_output;
  const hasOriginal = stats.artifacts.includes("original_preview.png");

  const canvasProps = {
    inputUrl: resultUrl(result_id, "input_8bit.png"),
    maskUrl: resultUrl(result_id, "mask.png"),
    probUrl: resultUrl(result_id, "prob.png"),
    stats,
    opacity,
    highlight,
  };

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-blue-200 bg-blue-50 px-5 py-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
          Tier: {stats.tier.replace(/_/g, " ")}{" "}
          <span className="cursor-help text-blue-400" title={TIER_DEFS}>
            (what&apos;s this?)
          </span>
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
          <Toggle color="rgb(255,47,146)" label="Predicted mask" checked={showMask} onChange={setShowMask} />
          <Toggle
            color="rgb(0,200,255)"
            label={`Hough overlay${stats.hough.enabled ? "" : " (off)"}`}
            checked={showHough}
            onChange={setShowHough}
            disabled={!stats.hough.enabled}
          />
          <Toggle color="linear-gradient(90deg,#2563eb,#ef4444)" label="Model confidence (qualitative)" checked={showProb} onChange={setShowProb} />
          <label className="flex items-center gap-2 text-xs text-slate-600">
            <input type="checkbox" checked={compareHough} onChange={(e) => setCompareHough(e.target.checked)} />
            Compare Hough off/on
          </label>
          <label className="ml-auto flex items-center gap-2 text-xs text-slate-600">
            Overlay opacity
            <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          </label>
        </div>

        {compareHough ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <figure>
              <figcaption className="mb-1 text-xs text-slate-500">Hough off</figcaption>
              <CanvasCompare {...canvasProps} showMask={showMask} showHough={false} showProb={showProb} />
            </figure>
            <figure>
              <figcaption className="mb-1 text-xs text-slate-500">Hough on</figcaption>
              <CanvasCompare {...canvasProps} showMask={showMask} showHough={true} showProb={showProb} />
            </figure>
          </div>
        ) : (
          <div className={`grid gap-3 ${hasOriginal ? "sm:grid-cols-3" : "sm:grid-cols-2"}`}>
            {hasOriginal && (
              <figure>
                <figcaption className="mb-1 text-xs text-slate-500">As uploaded (pre-resample)</figcaption>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={resultUrl(result_id, "original_preview.png")}
                  alt="As uploaded"
                  className="w-full rounded-lg border border-slate-300 bg-black"
                />
              </figure>
            )}
            <figure>
              <figcaption className="mb-1 text-xs text-slate-500">Model input (what the model saw)</figcaption>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={resultUrl(result_id, "input_8bit.png")}
                alt="Model input"
                className="w-full rounded-lg border border-slate-300 bg-black"
              />
            </figure>
            <figure>
              <figcaption className="mb-1 flex items-center justify-between text-xs text-slate-500">
                <span>Overlay (scroll to zoom · drag to pan)</span>
                {probHover && <span className="font-mono text-slate-600">p={probHover.p.toFixed(3)} @ ({probHover.x},{probHover.y})</span>}
              </figcaption>
              <CanvasCompare {...canvasProps} showMask={showMask} showHough={showHough} showProb={showProb} onProbHover={setProbHover} />
            </figure>
          </div>
        )}
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
              value={`${fmt(stats.timing_ms.preprocess + stats.timing_ms.inference + stats.timing_ms.hough)} ms`}
            />
          </dl>
        </Card>

        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-700">Provenance</h2>
            <button
              onClick={() => {
                navigator.clipboard.writeText(JSON.stringify(stats.provenance, null, 2));
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1500);
              }}
              className="inline-flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100"
            >
              <Copy className="h-3 w-3" />
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
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
            <Stat label="Checkpoint SHA" value={`${stats.provenance.checkpoint_sha256.slice(0, 12)}…`} small />
            <Stat label="Vendored commit" value={`${stats.provenance.vendored_source_commit.slice(0, 12)}…`} small />
          </dl>
        </Card>
      </div>

      <Card>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">
          Predicted components ({mo.predicted_component_count}) — click a row to highlight
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
                <tr
                  key={c.index}
                  onClick={() => setHighlight(highlight === c.index ? null : c.index)}
                  className={`cursor-pointer border-t border-slate-100 ${
                    highlight === c.index ? "bg-yellow-100" : "hover:bg-slate-50"
                  }`}
                >
                  <td className="py-1 pr-3">{c.index}</td>
                  <td className="py-1 pr-3">{fmt(c.pixel_count)}</td>
                  <td className="py-1 pr-3">[{c.bbox.join(", ")}]</td>
                  <td className="py-1 pr-3">{c.major_axis_px != null ? c.major_axis_px.toFixed(1) : "—"}</td>
                  <td className="py-1 pr-3">{c.orientation_deg != null ? c.orientation_deg.toFixed(1) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {mo.predicted_components.length === 0 && <p className="py-2 text-xs text-slate-500">No predicted components.</p>}
        </div>
      </Card>

      <Card>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">Downloads</h2>
        <div className="flex flex-wrap gap-2">
          {([
            ["overlay.png", "Overlay"],
            ["mask.png", "Mask"],
            ["prob.png", "Confidence"],
            ["input_8bit.png", "Model input"],
            ["stats.json", "Stats JSON"],
            ["bundle.zip", "All (.zip)"],
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

function Toggle({
  color,
  label,
  checked,
  onChange,
  disabled = false,
}: {
  color: string;
  label: string;
  checked: boolean;
  onChange: (b: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-slate-700">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} disabled={disabled} />
      <span className="inline-block h-3 w-3 rounded-sm" style={{ background: color }} />
      {label}
    </label>
  );
}

function Stat({ label, value, small = false }: { label: string; value: string | number; small?: boolean }) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className={`font-medium text-slate-800 ${small ? "" : "text-base"}`}>{value}</dd>
    </div>
  );
}
