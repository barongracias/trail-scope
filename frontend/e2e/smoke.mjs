// End-to-end smoke test: drives the real UI through the demo inference flow and asserts
// the output page renders. Run by CI (chromium) and locally. Exits non-zero on failure.
//
//   node e2e/smoke.mjs            # uses chromium installed by `npx playwright install`
//   PW_CHANNEL=chrome node ...    # use system Chrome instead
//
// Requires backend on :8000 and frontend on :3000 already running.

import { chromium } from "playwright";

const BASE = process.env.E2E_BASE_URL || "http://localhost:3000";
const channel = process.env.PW_CHANNEL || undefined;

const fail = (msg) => {
  console.error("FAIL:", msg);
  process.exitCode = 1;
};

const browser = await chromium.launch({ channel, headless: true });
const errors = [];
try {
  const page = await (await browser.newContext({ viewport: { width: 1280, height: 1400 } })).newPage();
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto(BASE, { waitUntil: "networkidle" });

  // Input page renders with the disclaimer.
  if (!(await page.content()).includes("not a validated detector")) fail("disclaimer missing on input page");

  // Pick the DECam demo and run.
  await page.locator("button:has-text('NAVSTAR-70')").first().click();
  await page.waitForFunction(
    () => {
      const b = [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === "Run inference");
      return b && !b.disabled;
    },
    { timeout: 20000 },
  );
  await page.locator("button:has-text('Run inference')").click();

  // Output page.
  await page.waitForSelector("text=Tier:", { timeout: 90000 });
  await page.waitForTimeout(1500);

  // Output page: tier banner, summary, the synced split-view canvases, and the
  // always-visible predicted-components section.
  const checks = {
    "tier banner": await page.locator("text=/Tier:/").count(),
    "result summary": await page.locator("text=Predicted mask pixels").count(),
    canvas: await page.locator("canvas").count(),
    "components section": await page.locator("text=/Predicted components/").count(),
  };
  for (const [k, v] of Object.entries(checks)) {
    if (!v) fail(`missing on output page: ${k}`);
  }

  // The OVERLAY canvas (the 2nd panel in the split-view) drew coloured mask/Hough pixels.
  // canvas[0] is the grayscale "Model input" panel.
  const colored = await page.evaluate(() => {
    const canvases = document.querySelectorAll("canvas");
    const c = canvases[1] || canvases[0];
    if (!c) return 0;
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4 * 997) {
      if (Math.abs(d[i] - d[i + 1]) > 30 || Math.abs(d[i + 1] - d[i + 2]) > 30) n++;
    }
    return n;
  });
  if (colored <= 0) fail("overlay canvas has no coloured overlay pixels");

  // Downloads tab holds the .zip bundle link.
  await page.locator("[role=tab]:has-text('Downloads')").click();
  await page.waitForTimeout(300);
  if (!(await page.locator("a:has-text('All (.zip)')").count())) {
    fail("missing zip bundle link in Downloads tab");
  }

  if (errors.length) fail("console errors: " + JSON.stringify(errors));

  if (!process.exitCode) console.log("E2E smoke PASS (checks:", JSON.stringify(checks), "colored:", colored, ")");
} catch (e) {
  fail("exception: " + (e instanceof Error ? e.stack : String(e)));
} finally {
  await browser.close();
}
