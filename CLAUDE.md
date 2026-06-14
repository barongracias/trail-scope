# trail-scope — project instructions (read first)

`trail-scope` is a **single-image qualitative inference demo** for the satellite-trail
detector built in the sibling MPhil thesis project (`../bg492`). FastAPI + Next.js +
Docker Compose; it takes an astronomical image, runs the locked thesis U-Net (+ optional
Hough), and returns an overlay, a binary mask, and a stats JSON. It is **not** a
benchmark, not a validated cross-domain tool, no training, no tunable thresholds.

The complete, frozen build specification is **`agents/spec.md`**. Concrete facts a
builder needs (checkpoint SHA, exact files to vendor, demo-data source, architecture
constants) are in **`agents/build_context.md`**. Record progress in
**`agents/progress.md`** as you go.

## Hard rules (do not violate)

1. **NEVER edit, stage, commit, or push anything under `../bg492`.** That is the
   MPhil thesis submission repository. It is **read-only reference only** — you may
   *read* it to copy the checkpoint and the files listed for vendoring
   (`agents/build_context.md`), nothing more. Do not run its scripts, do not modify
   its files, do not touch its git history.
2. **This repo (`trail-scope`) is push-allowed.** Once the GitHub remote exists, you
   (Claude or any agent) may commit and push code changes here freely. Until the
   remote is configured, commit locally.
3. **Vendor, do not depend.** Copy the minimal inference core into
   `backend/trailscope/vendored/` (per the spec) with a `VENDOR_MANIFEST.md` recording
   source repo/commit/file. **No imports from `../bg492` `src.*`** anywhere — enforced
   by a test.
4. **Locked invariants are constants, never API parameters:** checkpoint
   `model-best.pth` (SHA in `build_context.md`, asserted at startup — refuse to serve
   on mismatch), threshold `0.45`, normalisation `full_image`, Hough
   `0.1/50/100/250/3` (incl. `line_thickness=3`), `PATCH_SIZE=528`. User-facing knobs
   are ONLY: Hough on/off, optional pixel-scale override, optional FITS HDU index.
5. **Data policy.** Never ship, commit, or redistribute MeerLICHT imagery (it is
   collaboration data in `../bg492`). Demo data is the **public DECam frames only**
   (`build_context.md`). Include the NOIRLab acknowledgement where DECam data appears.
6. **Honesty / scope.** Qualitative inference only; no precision/recall/accuracy or
   benchmark claims for uploaded images. The disclaimer and neutral tier
   (`in_domain_like` / `recipe_matched` / `best_effort`) must appear on the input
   page, output page, README, `/model`, and `stats.json`. Stats use "predicted
   mask/component" language, never "trail"/"detection".

## Build approach

- Walking skeleton first: P1 (scaffold + checkpoint SHA gate + `/health`, `/model`)
  then P2 (the preprocessing contract — the only genuinely hard part). The milestone
  that retires the risk is **one DECam demo frame end-to-end through the backend → mask
  + stats JSON on CPU**. Do not build the frontend until the backend produces a correct
  mask on a real frame. Then P3 (`/infer` + artifacts + frontend) and P4 (Docker/CI/
  demo assets/README). Phasing detail in `agents/spec.md`.
- Use **this repo's own** `.venv` (gitignored) — do **not** share or install into the
  sibling `../astro_venv`, which is the thesis test environment and must not be polluted.
  Install **CPU torch wheels** (`--index-url https://download.pytorch.org/whl/cpu`);
  `opencv-python-headless`, not `opencv-python`. No GPU, no CSD3 — everything runs locally
  on the committed checkpoint.
- Keep `main.py` thin (routes/validation/CORS/lifespan/errors); real logic lives in
  `backend/trailscope/`.

## Status

GitHub remote: **not yet created** — the author will create it and confirm. Until then,
commit locally; do not attempt to push. Tests must pass and the checkpoint SHA gate must
be wired before the first push.
