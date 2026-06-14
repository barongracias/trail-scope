// Typed client for the trail-scope backend. Mirrors the InterPyApp api.ts pattern
// (ApiError + env base URL + a `handle` helper), retargeted to the inference endpoints.

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Tier = "in_domain_like" | "recipe_matched" | "best_effort";

export type Provenance = {
  format: string;
  hdu: number | null;
  stretch: string;
  stretch_fallback: boolean;
  pixel_scale_source: string;
  pixel_scale_arcsec: number | null;
  resample_factor: number;
  rgb_to_luminance: boolean;
  nonfinite_pixels_cleaned: number;
  checkpoint_sha256: string;
  threshold: number;
  vendored_source_commit: string;
};

export type PredictedComponent = {
  index: number;
  pixel_count: number;
  bbox: number[];
  major_axis_px: number | null;
  orientation_deg: number | null;
};

export type ResultFile =
  | "input_8bit.png"
  | "mask.png"
  | "overlay.png"
  | "prob.png"
  | "original_preview.png"
  | "stats.json"
  | "bundle.zip";

export type InferStats = {
  qualitative_only: boolean;
  benchmark: boolean;
  threshold_tuned: boolean;
  training_domain: string;
  scope: string;
  disclaimer: string;
  tier: Tier;
  warnings: string[];
  provenance: Provenance;
  artifacts: string[];
  image: { input_shape: number[]; processed_shape: number[]; n_patches: number };
  model_output: {
    predicted_mask_pixel_count: number;
    predicted_mask_fraction: number;
    predicted_component_count: number;
    max_model_probability: number;
    predicted_components: PredictedComponent[];
  };
  hough: { enabled: boolean; segment_count: number; segments: number[][] };
  timing_ms: { preprocess: number; inference: number; hough: number };
};

export type InferResponse = { result_id: string; stats: InferStats };

export type ModelInfo = {
  architecture: string;
  param_count: number;
  threshold: number;
  patch_size: number;
  normalisation: string;
  training_domain: string;
  tier_definitions: Record<string, string>;
  thesis_repo: string;
  checkpoint_sha256: string;
  scope: string;
  disclaimer: string;
  user_facing_knobs: string[];
};

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI errors come back as { detail: ... }.
    const detail =
      (data && (data.detail || data.error)) || `Request failed with ${res.status}`;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
  }
  return data as T;
}

export async function healthCheck(baseUrl: string = API_BASE_URL) {
  const res = await fetch(`${baseUrl}/health`, { cache: "no-store" });
  return handle<{ status: string; version: string; model_sha_ok: boolean }>(res);
}

export async function getModelInfo(baseUrl: string = API_BASE_URL) {
  const res = await fetch(`${baseUrl}/model`, { cache: "no-store" });
  return handle<ModelInfo>(res);
}

export type InferOptions = {
  hough: boolean;
  pixelScaleArcsec?: number | null;
  hduIndex?: number | null;
};

export async function infer(
  file: File,
  opts: InferOptions,
  baseUrl: string = API_BASE_URL,
): Promise<InferResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("hough", String(opts.hough));
  if (opts.pixelScaleArcsec != null && !Number.isNaN(opts.pixelScaleArcsec)) {
    form.append("pixel_scale_arcsec", String(opts.pixelScaleArcsec));
  }
  if (opts.hduIndex != null && !Number.isNaN(opts.hduIndex)) {
    form.append("hdu_index", String(opts.hduIndex));
  }
  const res = await fetch(`${baseUrl}/infer`, { method: "POST", body: form });
  return handle<InferResponse>(res);
}

export type FitsHdu = {
  index: number;
  type: string;
  shape: number[] | null;
  is_2d_image: boolean;
};

export async function inspectFits(
  file: File,
  baseUrl: string = API_BASE_URL,
): Promise<{ hdus: FitsHdu[] }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${baseUrl}/inspect`, { method: "POST", body: form });
  return handle<{ hdus: FitsHdu[] }>(res);
}

export function resultUrl(
  resultId: string,
  filename: ResultFile,
  baseUrl: string = API_BASE_URL,
): string {
  return `${baseUrl}/results/${resultId}/${filename}`;
}
