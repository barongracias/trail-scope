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

test("vocabulary stays neutral — no trail/detection wording in stats UI", () => {
  const page = fs.readFileSync(path.join(root, "app", "page.tsx"), "utf8");
  assert.ok(page.includes("Predicted mask"), "uses 'predicted mask' vocabulary");
  assert.ok(!/\bdetections?\b/i.test(page), "must not use 'detection' wording");
});

test("the cropped DECam demo asset is shipped", () => {
  assert.ok(
    fs.existsSync(path.join(root, "public", "demo", "decam_navstar70_crop.png")),
    "demo PNG must exist",
  );
});
