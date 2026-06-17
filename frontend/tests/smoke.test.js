import assert from "node:assert";
import { test } from "node:test";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();

test(".env.example documents the API URLs", () => {
  const env = fs.readFileSync(path.join(root, ".env.example"), "utf8");
  assert.ok(env.includes("NEXT_PUBLIC_API_URL="), "NEXT_PUBLIC_API_URL documented");
  assert.ok(env.includes("API_URL="), "API_URL documented");
});

test("the disclaimer appears on the input page", () => {
  const page = fs.readFileSync(path.join(root, "app", "page.tsx"), "utf8");
  assert.ok(
    page.includes("not a validated detector"),
    "disclaimer sentence must be present in the UI",
  );
});

test("input page uses 'predicted mask' vocabulary", () => {
  const page = fs.readFileSync(path.join(root, "app", "page.tsx"), "utf8");
  assert.ok(page.includes("Predicted mask"), "uses 'predicted mask' vocabulary");
});

test("no 'detection' wording anywhere in the UI (all app/*.tsx)", () => {
  const appDir = path.join(root, "app");
  for (const f of fs.readdirSync(appDir).filter((n) => n.endsWith(".tsx"))) {
    const src = fs.readFileSync(path.join(appDir, f), "utf8");
    assert.ok(!/\bdetections?\b/i.test(src), `no 'detection' wording in ${f}`);
  }
});

test("the cropped DECam demo asset is shipped", () => {
  assert.ok(
    fs.existsSync(path.join(root, "public", "demo", "decam_navstar70.png")),
    "demo PNG must exist",
  );
});
