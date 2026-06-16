"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, ArrowLeft, Copy, Download, Loader2, Upload } from "lucide-react";
import {
  ApiError,
  type FitsHdu,
  type InferResponse,
  type Tier,
  cancelJob,
  createJob,
  getJobStatus,
  healthCheck,
  infer,
  inspectFits,
  resultUrl,
} from "@/lib/api";
import CanvasCompare, { type ProbHover, type ViewState } from "./CanvasCompare";
import ConfidenceLegend from "./ConfidenceLegend";
import AboutPanel from "./AboutPanel";
import CropView from "./CropView";

const SCOPE = "A qualitative, single-image inference demo of the locked thesis detector.";

const DISCLAIMER =
  "This is not a validated detector for this input unless it is from the original MeerLICHT-style domain.";

const ACCEPTED = ".fits, .fit, .fits.fz, .png, .jpg, .jpeg, .tif";

const TIER_TEXT: Record<Tier, string> = {
  in_domain_like: "In-domain-like input – an 8-bit display image at a plausible scale.",
  recipe_matched:
    "Recipe-matched input – FITS with a header-resolved pixel scale (the validated DECam-style recipe).",
  best_effort: "Best-effort input – outside the validated recipe (e.g. unknown pixel scale).",
};
const TIER_DEFS =
  "in_domain_like: 8-bit display image at a plausible scale.\nrecipe_matched: FITS with a header-resolved pixel scale (the validated DECam recipe).\nbest_effort: everything else (e.g. unknown pixel scale – the model is not scale-invariant).";

// Examples: text-only chips, image shown on hover. DECam = public NOIRLab; MeerLICHT
// examples are reproduced from the public thesis figures (with acknowledgement).
const DEMOS = [
  { label: "NAVSTAR-70", file: "decam_navstar70_crop.png" },
  { label: "STARLINK-2600", file: "decam_starlink2600_crop.png" },
  { label: "DELTA-2 R/B", file: "decam_delta2_crop.png" },
  { label: "Star field", file: "decam_starfield_crop.png" },
  { label: "Hough gap (Fig 5.4)", file: "meerlicht_hough_gap_crop.png" },
];

const TOGGLE_TIPS = {
  mask: "Pixels the U-Net scored at or above the locked 0.45 threshold (pink).",
  hough:
    "Optional probabilistic Hough line-fit over a lower-threshold canvas (cyan); can bridge short gaps the U-Net leaves.",
  prob: "Per-pixel model probability as a blue→red heatmap. Qualitative; not a tunable threshold.",
  compare: "Show the predicted mask with the Hough overlay off vs on, side by side.",
};

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
  const [jobId, setJobId] = useState<string | null>(null);
  const cancelledRef = useRef(false);

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
      inspectFits(f).then((r) => setHduList(r.hdus)).catch(() => setHduList(null));
    }
  };

  // Click an example to select; click the selected one again to unselect.
  const pickDemo = async (demoFile: string) => {
    if (file?.name === demoFile) {
      chooseFile(null);
      return;
    }
    try {
      const res = await fetch(`/demo/${demoFile}`);
      if (!res.ok) throw new Error("Demo asset not found");
      chooseFile(new File([await res.blob()], demoFile, { type: "image/png" }));
    } catch {
      setError("Could not load that example.");
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
      cancelledRef.current = false;
      try {
        const { job_id } = await createJob(runFile, opts);
        setJobId(job_id);
        for (let i = 0; i < 1200; i++) {
          if (cancelledRef.current) return;
          const st = await getJobStatus(job_id);
          setStage(st.n_patches ? `${st.detail} · ${st.n_patches} patches` : st.detail);
          if (st.state === "done" && st.result_id && st.stats) {
            setJobId(null);
            setResult({ result_id: st.result_id, stats: st.stats });
            setPhase("output");
            return;
          }
          if (st.state === "error" || st.state === "cancelled") {
            setJobId(null);
            if (st.state === "cancelled") setPhase("input");
            else handleFailure(new ApiError(st.error ?? "Job failed", st.status_code ?? undefined), runFile);
            return;
          }
          await new Promise((r) => setTimeout(r, 700));
        }
        handleFailure(new ApiError("Timed out waiting for the job."), runFile);
      } catch (e) {
        if (!cancelledRef.current) handleFailure(e instanceof ApiError ? e : null, runFile);
      }
      return;
    }

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

  const cancelRun = async () => {
    cancelledRef.current = true;
    setPhase("input");
    if (jobId) {
      try {
        await cancelJob(jobId);
      } catch {
        /* best-effort */
      }
      setJobId(null);
    }
  };

  return (
    <main className="min-h-screen max-w-5xl mx-auto px-4 py-8 text-slate-800">
      {phase !== "output" && (
        <header className="mb-6 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight">
              <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent">
                trail-scope
              </span>{" "}
              <span aria-hidden="true">🛰️</span>
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-slate-600">{SCOPE}</p>
          </div>
          {phase === "input" && (
            <button
              onClick={() => file && run(file)}
              disabled={!file}
              className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Run inference
            </button>
          )}
        </header>
      )}

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
          onCropped={(f) => {
            setFile(f);
            run(f);
          }}
          pickDemo={pickDemo}
        />
      )}

      {phase === "processing" && (
        <ProcessingView filename={file?.name ?? ""} stage={stage} onCancel={largeMode ? cancelRun : undefined} />
      )}

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
          onBack={() => setPhase("input")}
          onReset={reset}
        />
      )}

      {phase === "output" && (
        <footer className="mt-10 border-t border-slate-200/60 pt-4 text-xs text-slate-500">
          <p className="font-medium text-slate-600">{DISCLAIMER}</p>
          <p className="mt-1">
            Demo data: public DECam frames (NSF&apos;s NOIRLab) and MeerLICHT examples reproduced
            from the thesis with the consortium&apos;s acknowledgement. Stats use &quot;predicted
            mask/component&quot; language and make no accuracy claims.
          </p>
        </footer>
      )}
    </main>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`glass rounded-2xl p-5 ${className}`}>{children}</section>;
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="group/tip relative inline-flex">
      <span
        className="ml-1 inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-slate-400 text-[10px] font-medium text-slate-500"
        aria-label={text}
        role="img"
      >
        ?
      </span>
      <span className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-1.5 hidden w-52 -translate-x-1/2 rounded-lg bg-slate-800 px-2.5 py-1.5 text-[11px] leading-snug text-white shadow-lg group-hover/tip:block">
        {text}
      </span>
    </span>
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
  onCropped: (f: File) => void;
  pickDemo: (f: string) => void;
}) {
  const {
    file, chooseFile, hough, setHough, largeMode, setLargeMode, pixelScale, setPixelScale,
    hduIndex, setHduIndex, hduList, dragging, setDragging, fileInputRef, error, reject413,
    cropping, setCropping, onCropped, pickDemo,
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
          role="button"
          tabIndex={0}
          aria-label="Upload an image: drag and drop, click to browse, or paste from clipboard"
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition focus:outline focus:outline-2 focus:outline-blue-400 ${
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

        <div className="mt-4">
          <p className="mb-2 text-xs font-medium text-slate-500">Or try an example (hover to preview):</p>
          <div className="flex flex-wrap gap-2">
            {DEMOS.map((d) => (
              <div key={d.file} className="group/demo relative">
                <button
                  onClick={() => pickDemo(d.file)}
                  className={`rounded-full border px-3 py-1 text-xs transition ${
                    file?.name === d.file
                      ? "border-blue-400 bg-blue-50 text-blue-700"
                      : "border-slate-300 text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  {d.label}
                </button>
                <div className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 hidden w-56 -translate-x-1/2 group-hover/demo:block">
                  <div className="glass rounded-xl p-1.5 shadow-xl">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`/demo/${d.file}`}
                      alt={`${d.label} preview`}
                      className="w-full rounded-lg border border-white/60 bg-black"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[10px] text-slate-400">
            DECam: public NSF&apos;s NOIRLab frames. MeerLICHT examples reproduced from the thesis
            with thanks to the MeerLICHT consortium; no raw collaboration data is redistributed.
          </p>
        </div>
      </Card>

      <details className="glass rounded-2xl p-5">
        <summary className="cursor-pointer text-sm font-semibold text-slate-700">
          Options – pixel scale, FITS HDU, Hough, large images
        </summary>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
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
            Process large images (over 64 patches) as a background job. Slower on CPU and still
            qualitative; the model, threshold, and recipe are unchanged.
          </span>
        </label>
      </details>

      <AboutPanel />

      {reject413 && !cropping && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {reject413}
        </div>
      )}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      )}

      <p className="text-center text-xs text-slate-500">{DISCLAIMER}</p>
    </div>
  );
}

function ProcessingView({
  filename,
  stage,
  onCancel,
}: {
  filename: string;
  stage: string;
  onCancel?: () => void;
}) {
  return (
    <Card className="flex flex-col items-center gap-4 py-12 text-center">
      <Loader2 className="h-8 w-8 animate-spin text-blue-600" aria-hidden="true" />
      <div role="status" aria-live="polite">
        <p className="text-sm font-medium text-slate-700">{filename}</p>
        <p className="mt-1 text-sm text-slate-500">{stage}…</p>
      </div>
      <p className="max-w-md text-xs text-slate-400">
        {onCancel
          ? "Large images run as a background job and may take a while on CPU."
          : "Large images may take up to a minute on CPU. Images over 64 patches are rejected in this demo."}
      </p>
      <p className="max-w-md text-xs text-slate-400">
        The model and threshold are fixed; this run does not tune parameters or estimate accuracy.
      </p>
      {onCancel && (
        <button
          onClick={onCancel}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          Cancel
        </button>
      )}
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
  onBack: () => void;
  onReset: () => void;
}) {
  const {
    result, showMask, setShowMask, showHough, setShowHough, showProb, setShowProb,
    opacity, setOpacity, highlight, setHighlight, probHover, setProbHover,
    compareHough, setCompareHough, copied, setCopied, onBack, onReset,
  } = props;
  const { result_id, stats } = result;
  const mo = stats.model_output;
  const hasOriginal = stats.artifacts.includes("original_preview.png");
  const [view, setView] = useState<ViewState>({ scale: 1, tx: 0, ty: 0 });

  // Confidence is mutually exclusive with mask/Hough.
  const selectMask = (v: boolean) => {
    setShowMask(v);
    if (v) setShowProb(false);
  };
  const selectHough = (v: boolean) => {
    setShowHough(v);
    if (v) setShowProb(false);
  };
  const selectProb = (v: boolean) => {
    setShowProb(v);
    if (v) {
      setShowMask(false);
      setShowHough(false);
    }
  };

  const canvasProps = {
    inputUrl: resultUrl(result_id, "input_8bit.png"),
    maskUrl: resultUrl(result_id, "mask.png"),
    probUrl: resultUrl(result_id, "prob.png"),
    stats,
    opacity,
    highlight,
    view,
    onViewChange: setView,
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back
        </button>
        <button
          onClick={onReset}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700"
        >
          Run another image
        </button>
      </div>

      {/* IMAGES — the focus of the page. */}
      <Card>
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <Toggle color="rgb(255,47,146)" label="Predicted mask" tip={TOGGLE_TIPS.mask} checked={showMask} onChange={selectMask} />
          <Toggle
            color="rgb(0,200,255)"
            label={`Hough overlay${stats.hough.enabled ? "" : " (off)"}`}
            tip={TOGGLE_TIPS.hough}
            checked={showHough}
            onChange={selectHough}
            disabled={!stats.hough.enabled}
          />
          <Toggle color="linear-gradient(90deg,#2563eb,#ef4444)" label="Model confidence" tip={TOGGLE_TIPS.prob} checked={showProb} onChange={selectProb} />
          {showProb && <ConfidenceLegend />}
          <label className="flex items-center gap-1 text-xs text-slate-600">
            <input type="checkbox" checked={compareHough} disabled={showProb} onChange={(e) => setCompareHough(e.target.checked)} />
            Compare Hough off/on
            <InfoTip text={TOGGLE_TIPS.compare} />
          </label>
          <label className="ml-auto flex items-center gap-2 text-xs text-slate-600">
            Overlay opacity
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              aria-label="Overlay opacity"
              onChange={(e) => setOpacity(Number(e.target.value))}
            />
          </label>
        </div>

        {compareHough && !showProb ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <figure>
              <figcaption className="mb-1 text-xs text-slate-500">Hough off</figcaption>
              <CanvasCompare {...canvasProps} showMask={showMask} showHough={false} showProb={false} />
            </figure>
            <figure>
              <figcaption className="mb-1 text-xs text-slate-500">Hough on</figcaption>
              <CanvasCompare {...canvasProps} showMask={showMask} showHough={true} showProb={false} />
            </figure>
          </div>
        ) : (
          <div className={`grid gap-3 ${hasOriginal ? "lg:grid-cols-3 sm:grid-cols-2" : "sm:grid-cols-2"}`}>
            {hasOriginal && (
              <figure>
                <figcaption className="mb-1 text-xs text-slate-500">As uploaded (pre-resample)</figcaption>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={resultUrl(result_id, "original_preview.png")}
                  alt="The image as uploaded, before resampling"
                  className="w-full rounded-lg border border-slate-300 bg-black"
                />
              </figure>
            )}
            <figure>
              <figcaption className="mb-1 text-xs text-slate-500">Model input</figcaption>
              <CanvasCompare {...canvasProps} showMask={false} showHough={false} showProb={false} />
            </figure>
            <figure>
              <figcaption className="mb-1 flex items-center justify-between text-xs text-slate-500">
                <span>Overlay (scroll to zoom · drag to pan)</span>
                {probHover && (
                  <span className="font-mono text-slate-600">
                    p={probHover.p.toFixed(3)} @ ({probHover.x},{probHover.y})
                  </span>
                )}
              </figcaption>
              <CanvasCompare {...canvasProps} showMask={showMask} showHough={showHough} showProb={showProb} onProbHover={setProbHover} />
            </figure>
          </div>
        )}

        <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-slate-100 pt-3 text-sm sm:grid-cols-4">
          <Stat label="Predicted mask pixels" value={fmt(mo.predicted_mask_pixel_count)} small />
          <Stat label="Predicted components" value={fmt(mo.predicted_component_count)} small />
          <Stat label="Max model probability" value={mo.max_model_probability.toFixed(4)} small />
          <Stat label="Hough segments" value={fmt(stats.hough.segment_count)} small />
          <Stat label="Mask fraction" value={mo.predicted_mask_fraction.toExponential(2)} small />
          <Stat label="Processed shape" value={stats.image.processed_shape.join(" × ")} small />
          <Stat label="Patches" value={fmt(stats.image.n_patches)} small />
          <Stat
            label="Runtime"
            value={`${fmt(stats.timing_ms.preprocess + stats.timing_ms.inference + stats.timing_ms.hough)} ms`}
            small
          />
        </dl>
      </Card>

      {/* PREDICTED COMPONENTS — main section below the images. */}
      <details className="glass rounded-2xl p-5" open>
        <summary className="cursor-pointer text-sm font-semibold text-slate-700">
          Predicted components ({mo.predicted_component_count}) – click a row to highlight on the overlay
        </summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3">#</th>
                <th className="py-1 pr-3">Pixels</th>
                <th className="py-1 pr-3">bbox [x,y,w,h]</th>
                <th className="py-1 pr-3">Major axis (px)</th>
                <th className="py-1 pr-3">Orientation (°)</th>
                <th className="py-1 pr-3">Mean conf.</th>
                <th className="py-1 pr-3">Max conf.</th>
              </tr>
            </thead>
            <tbody>
              {mo.predicted_components.slice(0, 50).map((c) => (
                <tr
                  key={c.index}
                  tabIndex={0}
                  aria-label={`Component ${c.index}, ${c.pixel_count} pixels`}
                  onClick={() => setHighlight(highlight === c.index ? null : c.index)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setHighlight(highlight === c.index ? null : c.index);
                    }
                  }}
                  className={`cursor-pointer border-t border-slate-100 focus:outline focus:outline-2 focus:outline-blue-400 ${
                    highlight === c.index ? "bg-yellow-100" : "hover:bg-slate-50"
                  }`}
                >
                  <td className="py-1 pr-3">{c.index}</td>
                  <td className="py-1 pr-3">{fmt(c.pixel_count)}</td>
                  <td className="py-1 pr-3">[{c.bbox.join(", ")}]</td>
                  <td className="py-1 pr-3">{c.major_axis_px != null ? c.major_axis_px.toFixed(1) : "–"}</td>
                  <td className="py-1 pr-3">{c.orientation_deg != null ? c.orientation_deg.toFixed(1) : "–"}</td>
                  <td className="py-1 pr-3">{c.mean_probability.toFixed(3)}</td>
                  <td className="py-1 pr-3">{c.max_probability.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {mo.predicted_components.length === 0 && (
            <p className="py-2 text-xs text-slate-500">No predicted components.</p>
          )}
          <p className="mt-2 text-[11px] text-slate-400">
            &quot;Confidence&quot; is the model&apos;s probability within the component&apos;s pixels – a
            qualitative model output, not a likelihood that a real object is present.
          </p>
        </div>
      </details>

      {/* Secondary detail – collapsed by default. */}
      <details className="glass rounded-2xl p-5">
        <summary className="cursor-pointer text-sm font-semibold text-blue-700">
          Tier: {stats.tier.replace(/_/g, " ")}{" "}
          <span className="cursor-help text-blue-400" title={TIER_DEFS}>
            (what&apos;s this?)
          </span>
        </summary>
        <p className="mt-2 text-sm text-slate-700">{TIER_TEXT[stats.tier]}</p>
        <p className="mt-1 text-sm font-medium text-slate-700">{DISCLAIMER}</p>
        {stats.warnings.length > 0 && (
          <ul className="mt-2 list-disc pl-5 text-xs text-amber-700">
            {stats.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        )}
      </details>

      <details className="glass rounded-2xl p-5">
        <summary className="flex cursor-pointer items-center justify-between text-sm font-semibold text-slate-700">
          <span>Provenance</span>
          <button
            onClick={(e) => {
              e.preventDefault();
              navigator.clipboard.writeText(JSON.stringify(stats.provenance, null, 2));
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            }}
            aria-label="Copy provenance JSON to clipboard"
            className="inline-flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs font-normal text-slate-600 hover:bg-slate-100"
          >
            <Copy className="h-3 w-3" aria-hidden="true" />
            {copied ? "Copied" : "Copy"}
          </button>
        </summary>
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
          <Stat label="Format" value={stats.provenance.format} small />
          <Stat label="HDU" value={stats.provenance.hdu ?? "–"} small />
          <Stat label="Stretch" value={stats.provenance.stretch} small />
          <Stat label="Stretch fallback" value={String(stats.provenance.stretch_fallback)} small />
          <Stat label="Pixel-scale source" value={stats.provenance.pixel_scale_source} small />
          <Stat
            label="Pixel scale"
            value={stats.provenance.pixel_scale_arcsec != null ? `${stats.provenance.pixel_scale_arcsec} "/px` : "–"}
            small
          />
          <Stat label="Resample factor" value={stats.provenance.resample_factor.toFixed(4)} small />
          <Stat label="Threshold" value={stats.provenance.threshold.toString()} small />
          <Stat label="RGB→luminance" value={String(stats.provenance.rgb_to_luminance)} small />
          <Stat label="Non-finite cleaned" value={fmt(stats.provenance.nonfinite_pixels_cleaned)} small />
          <Stat label="Checkpoint SHA" value={`${stats.provenance.checkpoint_sha256.slice(0, 12)}…`} small />
          <Stat label="Vendored commit" value={`${stats.provenance.vendored_source_commit.slice(0, 12)}…`} small />
        </dl>
      </details>

      <details className="glass rounded-2xl p-5">
        <summary className="cursor-pointer text-sm font-semibold text-slate-700">Downloads</summary>
        <div className="mt-3 flex flex-wrap gap-2">
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
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              {label}
            </a>
          ))}
        </div>
      </details>
    </div>
  );
}

function Toggle({
  color,
  label,
  tip,
  checked,
  onChange,
  disabled = false,
}: {
  color: string;
  label: string;
  tip?: string;
  checked: boolean;
  onChange: (b: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="flex items-center gap-1.5 text-sm text-slate-700">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} disabled={disabled} />
      <span className="inline-block h-3 w-3 rounded-sm" style={{ background: color }} />
      {label}
      {tip && <InfoTip text={tip} />}
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
