# trail-scope — post-v1 improvement plan (v1.1+)

Companion to `agents/spec.md` (the **frozen v1** spec). Nothing here changes v1; these are
sequenced enhancements to pick up after the v1 build (P1–P4, complete 2026-06-14).

**Binding constraints for every item below** (from `CLAUDE.md`):
- Locked invariants stay constants, never knobs: checkpoint (SHA-gated), `THRESHOLD=0.45`,
  Hough `0.1/50/100/250/3`, `PATCH_SIZE=528`, `full_image` norm, `TARGET=0.56″/px`. User
  knobs remain only: Hough on/off, pixel-scale override, HDU index.
- Honesty/scope: qualitative only. No precision/recall/accuracy, no benchmark claims, no
  threshold tuning for uploads. "predicted mask/component" vocabulary, never
  "trail"/"detection". Disclaimer + neutral tier stay on input page, output page, README,
  `/model`, `stats.json`.
- Data policy: public DECam only; never MeerLICHT.

Effort key: **S** ≤½ day · **M** ~1 day · **L** multi-day. ⟐ = already a spec v2 candidate.

> **Status (2026-06-16): ALL roadmap items are now BUILT and verified.** v1.1, v1.2, v1.3
> (incl. #10 TorchScript, #11 async jobs, #12 full-frame, #14 CI Playwright) and the v1.4
> vendored-checksum guard are complete. #11/#12 were landed **additively** — the
> synchronous `/infer` path is unchanged; large images opt in to the async `/jobs` path
> (status-file polling) with a hard `MAX_JOB_PATCH_BUDGET` ceiling, never unbounded. The
> "keep vendored" decision stands; the checksum guard makes accidental edits loud. Build
> details in `agents/progress.md`.

---

## Phase v1.1 — high-value, low-risk (in the spirit of the frozen contract)

### 1. Demo-result caching — S
**Why:** the demo picker re-runs the U-Net (~1–2 s CPU) on every click.
**Approach (backend):** in `main.py` `/infer`, compute `key = sha256(file_bytes + repr(hough,
pixel_scale, hdu))`; keep a small dict `key → result_id` (and reuse the existing per-id
result dir). If hit and the dir still exists, return `{result_id, stats}` from the cached
`stats.json` (read it back) without re-inferring; else run and record. Bound the map (e.g.
LRU, 64 entries) and clear it in the lifespan startup alongside the dirs.
**Notes:** cache is best-effort; correctness is unaffected by eviction. No schema change.
**Acceptance:** second identical demo request returns the same `result_id` and skips the
`infer.completed` U-Net path (assert via a timing/log check in a test).

### 2. Probability heatmap toggle — M
**Why:** the single most *honest* addition — shows where and how confident the model is.
**Approach (backend):** in `artifacts.run_inference`, after the prob canvas, render a
colormapped 0→1 PNG (`prob.png`) with a perceptually-uniform map (e.g. `cv2.applyColorMap`
on `(prob*255)`), and add `"prob.png"` to `_RESULT_FILES` in `main.py` and to the
`resultUrl` filename union in `lib/api.ts`. Add `max/mean` already in stats; optionally add
a tiny legend.
**Approach (frontend):** new overlay layer in `CanvasCompare.tsx` (toggle "Model
confidence") drawn *under* mask/Hough with its own opacity; or a third image panel.
**Honesty:** label it "model confidence (qualitative)", NOT a tunable threshold. Do **not**
add a client-side re-threshold slider over it (that implies tuning — out of scope).
**Acceptance:** `prob.png` served + rendered; values monotonic with `prob` (spot-check).

### 3. Click-a-component-to-highlight — S (frontend only)
**Why:** make the components table actionable.
**Approach:** lift `selectedComponent` state into `OutputView`; pass to `CanvasCompare`;
when set, stroke that component's `bbox` (already in `stats.model_output.predicted_components`)
in a highlight colour on the canvas. Row click toggles selection; hover highlights.
**Acceptance:** clicking row N boxes component N on the overlay; clicking again clears.

### 4. Crop-to-fit for the 413 path — M
**Why:** turn the "image too large (>64 patches)" dead end into a workflow.
**Approach (frontend):** when `/infer` returns 413, instead of only showing the message,
render the uploaded image on a canvas with a draggable/resizable crop rectangle; on confirm,
crop client-side (canvas `toBlob`) and resubmit. Show the live post-resample patch estimate
(replicate the budget math: `ceil(h/528)*ceil(w/528)` against the chosen scale) so the user
sees when they're under 64.
**Honesty:** cropping is a user action on their own image; provenance already records
`input_shape` vs `processed_shape`. No locked-path change.
**Acceptance:** an image that 413s can be cropped in-browser and run successfully.

---

## Phase v1.2 — usability

### 5. FITS HDU picker dropdown — M
**Why:** typing an index is opaque; the backend already lists HDUs on error.
**Approach:** add `GET /inspect` (multipart `file`) that opens the FITS and returns
`[{index, type, shape}]` (reuse `_hdu_summary`/`select_image_hdu` logic from `preprocess`).
Frontend: on a FITS upload, call `/inspect`, populate a dropdown; pass the chosen
`hdu_index` to `/infer`. Non-FITS uploads skip it.
**Acceptance:** multi-HDU FITS shows a dropdown; selection drives `/infer`.

### 6. Original vs model-input view (FITS/16-bit) — S
**Why:** make the ZScale/resample domain shift explicit; strengthens the honesty story.
**Approach (backend):** optionally also write `original_preview.png` (a downsized ZScale of
the *pre-resample* cleaned image) and add to the result set; (frontend) a third panel
"As uploaded → what the model saw". For 8-bit passthrough the two are identical (skip).
**Acceptance:** FITS results show both panels; provenance dims line up.

### 7. Overlay UX cluster — S (frontend)
Zoom/pan on the comparison (wheel + drag) for large frames; a **cursor probability readout**
(needs `prob.png` from #2, or a served prob array — sample on mousemove); **paste-from-
clipboard** upload (`onPaste` → file); **copy-provenance** button (JSON to clipboard);
**"why this tier?"** tooltip wired to `/model`'s `tier_definitions`.
**Acceptance:** each control works; no a11y regressions (keyboard-focusable).

### 8. Bundled ZIP download — S
**Why:** one click for all artifacts + stats.
**Approach (backend):** `GET /results/{id}/bundle.zip` streams the four (or six) files via
`zipfile`/`StreamingResponse`. Frontend: a "Download all (.zip)" button.
**Acceptance:** zip contains exactly the served artifacts + `stats.json`.

### 9. Hough off/on side-by-side view — S ⟐
**Why:** the spec's one sanctioned comparison feature; shows the classical aid's effect.
**Approach (frontend only):** the canvas already toggles Hough independently. Add a
"Compare" mode that renders two `CanvasCompare` panels — left `showHough=false`, right
`showHough=true` — sharing opacity. No backend change; segments already in `stats`.
**Honesty:** purely visual; both use the locked Hough params. Do not expose Hough
parameters as knobs.
**Acceptance:** toggling Compare shows mask-only vs mask+Hough side by side.

---

## Phase v1.3 — performance & ops

### 10. CPU inference speedups — S→M
`torch.inference_mode()` in `preprocess_core.infer_probability_canvas` (replace `no_grad`);
tune `torch.set_num_threads(os.cpu_count())` at startup; **export the locked weights to
TorchScript or ONNX** once and load that for faster CPU inference (keep the SHA gate on the
source `.pth`; derive + checksum the exported artifact at build).
**Acceptance:** measured wall-clock drop on the DECam frame with identical mask pixel count
(±tolerance) — no behavioural change.

### 11. Async job queue + progress — L ⟐
**Why:** large frames block the synchronous request; also the path to >64-patch support.
**Approach:** v2-style status files in the per-result dir (`status.json`: queued→preprocess
→inferring(patch k/N)→rendering→done|error); `/infer` returns `{result_id}` immediately;
add `GET /results/{id}/status` and stream patch progress (SSE) to the processing view.
A single worker (CPU-bound) with a bounded queue; reject/queue-full → honest message.
**Acceptance:** UI shows live patch progress; a slow frame doesn't hold the request open.

### 12. Full-frame support beyond 64 patches — L ⟐ (depends on #11)
Lift `MAX_PATCH_BUDGET` behind the async path, with a hard ceiling and a clear runtime/RAM
warning. Keep the 64-patch synchronous fast path as the default. **Stays out of scope until
#11 exists** — never make the synchronous path unbounded.

### 13. Result-dir TTL cleanup — S
**Why:** dirs are only cleared on startup; a long-running server accumulates them.
**Approach:** record `created_at` per result; a periodic asyncio task (or check on each
`/infer`) removes dirs older than `RESULTS_TTL_SECONDS` (env, default e.g. 3600). Keep the
startup wipe.
**Acceptance:** an aged result dir is removed; a fresh one survives.

### 14. CI Playwright smoke + rate limiting — M
Add a CI job that boots backend+frontend and runs the verified browser flow (input → demo →
infer → output assertions) headless. Add a small concurrency guard on `/infer` (semaphore;
CPU-bound) returning a polite 429 when saturated.
**Acceptance:** CI fails if the end-to-end UI flow breaks; concurrent floods get 429 not OOM.

---

## Phase v1.4 — vendoring: KEEP IT (decided 2026-06-15) — optional integrity hardening — S

**Decision: leave the inference core vendored. Do not extract, do not package, do not touch
`bg492`.** Reviewed by the thesis-repo agent (which mapped the coupling) and the trail-scope
agent; both concur. The prior "no pip package" decision stands and the coupling map makes
the case *stronger*, not weaker.

**Why (this is the textbook-correct use of vendoring, not a smell):**
- The "silently drifting copy" risk that normally motivates de-vendoring assumes an
  **actively developed upstream**. Here the upstream is a **frozen thesis submission we have
  explicitly decided not to touch** — a frozen source cannot drift. Vendoring a thin, frozen
  harness away from a repo whose release cadence you deliberately don't control is exactly
  what vendoring is *for*.
- The genuine source of truth — the trained model — is already **singular and SHA-gated**.
  The code around it is a harness, **behaviourally pinned by the e2e DECam test** (if the
  vendored path breaks, NAVSTAR-70's 13113 px moves).

**Coupling map (why extraction would be clean — which cuts both ways).** The core is
leaf-level: `unet.py` → torch only; `hough_runner.py` → cv2/numpy only; `loading.py`'s only
`src.*` edge was the attention_unet dispatch (already cut in the vendored copy); the 10
`decam_cold_inference.py` functions → numpy/cv2 + one constant (`PATCH_SIZE=528`). The
apparent `src.utils.imaging.resize_for_display` edge is used **only** in the montage/figure
code (lines ~592–637), **not** by any of the 10 inference functions. So the entire `src.*`
footprint of the inference core is one integer + an attention branch already removed. It's a
clean lift precisely **because the vendored copy is already a clean, complete lift** — there
is nothing tangled left to fix.

**Options considered and rejected:**
- **(b) Separate shared package both repos depend on** — only delivers "one source of truth"
  if `bg492` *also* consumes it, which means ripping the modules out of `bg492/src/` and
  replacing them with imports = exactly the restructuring of submission-ready code the prior
  decision rejected (1–2 days + a new repo/CI/release to maintain + re-verifying the thesis
  reproduces). Real submission risk; post-submission only.
- **(c) Make `bg492` a pip dependency of trail-scope** — drags the entire research codebase
  (training, sweeps, Optuna, attention, evaluation) and its heavy deps in to use four leaf
  files, binds the demo to an experiment-shaped unstable API, and re-introduces the exact
  `src.*` coupling `test_no_src_imports.py` forbids. Architecturally the worst option.

**Optional, proportionate hardening (the only thing worth doing — and not urgent).** If the
"silent" part still nags: record the **per-file SHA-256 of each vendored source** in
`VENDOR_MANIFEST.md` and add a test asserting the vendored files still match those hashes.
That converts an accidental local edit of the copy into a **loud failure** instead of silent
drift. ~1 hour, entirely within trail-scope, touches nothing in `bg492`. Even this is
optional — the e2e DECam test already guards faithfulness behaviourally. **Don't do it
mid-annotation**; it changes nothing about correctness.

**Status: settled — keep vendored.** No action required. The optional checksum guard above
is the only future work, and it is low-priority.

---

## Phase v1.5 — interpretability, honest stats, and UX depth (COMPLETE 2026-06-16)

Additive polish after the roadmap's core was completed. Same guardrails: qualitative only,
no accuracy/benchmark/tuning, "predicted mask/component" vocabulary, locked invariants.
**Status: all seven items (1, 2, 3, 7, 8, 9, 10) built and browser-verified, zero console
errors. Deferred sub-items #4/#5/#6 remain noted below.**

### v1.5-1. Confidence colourbar/legend — S (frontend)
The `prob.png` heatmap ships with no visible scale. Add a 0→1 colourbar (matching the
canvas blue→red colourmap exactly) beside the overlay when "Model confidence" is on, so the
map is interpretable. Pairs with the existing cursor `p=…` readout.

### v1.5-2. About / method explainer — S (frontend)
A collapsible "About this demo" panel: one honest paragraph on the locked U-Net + Hough
pipeline, the MeerLICHT training domain, the three tiers, the thesis link, and the
disclaimer. Gives a first-time visitor context; reinforces the honesty framing.

### v1.5-3. Per-component confidence — S (backend + frontend)
Each predicted component already reports pixels/bbox/major-axis/orientation. Add **mean and
max model probability within the component's pixels** (sampled from the prob canvas — pass
it into `_component_stats`). New `PredictedComponent` fields `mean_probability`,
`max_probability`; two more columns in the components table. Honest model-output detail, no
"trail" claim.

### v1.5-7. Cancel a running async job — S (backend + frontend)
`DELETE /jobs/{id}` cancels the asyncio task (new terminal state `cancelled`);
`JobManager.cancel`. Frontend: a Cancel button in the processing view during large-mode
runs. Closes the gap that a multi-minute 256-patch job can't be stopped.

### v1.5-8. Accessibility + responsive pass — S (frontend)
ARIA labels on icon-only buttons and sliders, keyboard operability for toggles/tabs, focus
states, and a mobile-layout check (the output grid is desktop-first). No new features —
quality only.

### v1.5-9. Tabbed output + synchronized split-view — M (frontend)
The output page is dense. Group it into tabs (Overview / Components / Provenance /
Downloads). Make the comparison a **synchronized split-view**: lift the zoom/pan state out
of `CanvasCompare` so the model-input and overlay panels zoom/pan together (render the
model-input panel as a base-only `CanvasCompare`). The "as uploaded" preview stays a static
panel (different resolution).

### v1.5-10. Demo gallery — S (assets)
Ship 2–3 small cropped 8-bit PNGs from additional public DECam frames (via
`download_demo_assets.py`), keeping the in-repo total small. The picker already supports
multiple entries. NOIRLab acknowledgement unchanged; never MeerLICHT.

**Deferred from the original list (not in this slice):** #4 extra shape descriptors,
#5 physical-unit (arcsec) measurements, #6 confidence histogram. Reconsider after v1.5.

## Explicitly NOT planned (would break scope — do not build)
Threshold slider on uploads; any precision/recall/accuracy or benchmark output; a
model-selection dropdown; exposing Hough/stretch/patch params as knobs; "detection"/"trail"
vocabulary; multi-image batch (v1 is single-image by design). A client-side **re-threshold**
slider over the probability map is specifically excluded — it implies tuning the locked
detector.
