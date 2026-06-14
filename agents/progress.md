# Build progress log

Append a dated entry per work session: what was built, decisions made, what's verified,
and what's next. Keep newest at the bottom.

## Phasing (from agents/spec.md §Phasing)

- [ ] **P1** — repo scaffold (backend/ + frontend/ + docker-compose), vendoring +
  `VENDOR_MANIFEST.md`, checkpoint copy + SHA gate (startup refusal on mismatch),
  `/health`, `/model`.
- [ ] **P2** — `preprocess.py` dispatch (FITS/PNG), pixel-scale/tier logic, patch-budget
  check after resample, + tests. **Milestone: one DECam demo frame end-to-end through the
  backend → mask + stats JSON on CPU.**
- [ ] **P3** — `/infer` (sync, 64-patch cap, 413 over), `artifacts.py` (overlay/mask/
  input_8bit/stats), frontend results page.
- [ ] **P4** — Docker/CI, demo assets + `download_demo_assets.py`, README with scope
  sentence + disclaimer.

## Log

### 2026-06-14 — scaffolding created (no code yet)
- `trail-scope/` created as a sibling of `bg492` with `CLAUDE.md` (rules), `agents/spec.md`
  (frozen v1 spec), and `agents/build_context.md` (checkpoint SHA `ff680804…`, vendoring
  source paths, demo source, architecture constants). No application code written yet.
- GitHub remote not yet created; commit locally until the author confirms it exists.
- Next: P1.
