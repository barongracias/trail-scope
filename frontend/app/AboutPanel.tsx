"use client";

// Collapsible "About this demo" explainer — gives a first-time visitor honest context for
// the locked detector and the neutral tiers, and links the thesis. No performance claims.
export default function AboutPanel() {
  return (
    <details className="rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-700 shadow-sm">
      <summary className="cursor-pointer font-semibold text-slate-800">
        About this demo & method
      </summary>
      <div className="mt-3 space-y-3 leading-relaxed">
        <p>
          trail-scope runs a <strong>locked U-Net</strong> (485,673 parameters) trained on
          MeerLICHT 8-bit display-PNG patches, followed by an optional probabilistic Hough
          transform. An uploaded image is tiled into 528×528 patches, each scored by the
          network; pixels above a fixed probability threshold of <strong>0.45</strong> form
          the predicted mask, and Hough lines are drawn over a lower-threshold canvas. The
          model, threshold, normalisation, and Hough parameters are <strong>fixed
          constants</strong> — there is nothing to tune.
        </p>
        <p>
          Because the model is not scale-invariant, FITS inputs with a known pixel scale are
          resampled toward the training scale (~0.56″/px); 8-bit display images pass through.
          Every result is tagged with a neutral tier:
        </p>
        <ul className="list-disc space-y-1 pl-5">
          <li>
            <strong>in_domain_like</strong> — an 8-bit display image at a plausible scale.
          </li>
          <li>
            <strong>recipe_matched</strong> — FITS with a header-resolved pixel scale (the
            validated DECam-style recipe).
          </li>
          <li>
            <strong>best_effort</strong> — everything else (e.g. unknown pixel scale).
          </li>
        </ul>
        <p className="font-medium text-slate-700">
          This is a qualitative inference demo only. It is not a benchmark and makes no
          accuracy claims; the model is not a validated detector for an input unless it is
          from the original MeerLICHT-style domain.
        </p>
        <p className="text-xs text-slate-500">
          Detector and method:{" "}
          <a
            href="https://github.com/barongracias/bg492"
            className="underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            MPhil thesis repository
          </a>
          . Demo frames are public NSF NOIRLab DECam products.
        </p>
      </div>
    </details>
  );
}
