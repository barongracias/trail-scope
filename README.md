# trail-scope

A single-image **qualitative inference demo** for satellite-trail detection: upload an
astronomical image, run a locked U-Net (+ optional probabilistic Hough), and get back an
overlay, a binary mask, and a stats summary. FastAPI backend + Next.js frontend + Docker
Compose.

> **Scope.** Qualitative inference only — **not** a benchmark, not a validated
> cross-domain detector. No training, no tunable thresholds. The model was trained on
> MeerLICHT 8-bit display renders; results on other inputs are out-of-domain and
> qualitative. No performance claims are made for uploaded images.

The detector is the locked model from an MPhil dissertation (a PyTorch replication of the
ASTA U-Net + Hough pipeline of Stoppa et al. 2024). DECam demo frames are public NSF
NOIRLab Astro Data Archive products.

## Status

Under construction — see `agents/spec.md` (build specification), `agents/build_context.md`
(concrete facts), and `agents/progress.md` (build log). Project rules are in `CLAUDE.md`.
