#!/usr/bin/env node
/**
 * Save backup screenshots of the DEMO MODE presets and comparison panel.
 *
 * Playwright is intentionally not a project dependency (the demo must run
 * offline). Install it only when preparing slides:
 *
 *   cd frontend
 *   npm install --no-save playwright
 *   npx playwright install chromium
 *   node scripts/demo_screenshots.mjs
 *
 * The page must already be running with DEMO_INJECTION_ENABLED and
 * NEXT_PUBLIC_DEMO_INJECTION_ENABLED set to true.
 */

import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const BASE =
  process.env.DEMO_BASE_URL ?? "http://localhost:3000/demo/injection";
const OUT = fileURLToPath(new URL("../demo-screenshots/", import.meta.url));

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  console.error(
    "Playwright is not installed. Run:\n" +
      "  npm install --no-save playwright\n" +
      "  npx playwright install chromium\n" +
      "then retry.",
  );
  process.exit(2);
}

const PRESETS = [
  "No injection (real series)",
  "Slow retreat",
  "Below the gate",
  "Borderline",
  "Fast retreat",
  "Landward move",
];

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });

try {
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByLabel("Speed").selectOption("60");

  for (const label of PRESETS) {
    await page.getByRole("button", { name: new RegExp(label) }).first().click();
    await page.getByRole("button", { name: "Start" }).click();
    // Let the stream run far enough to show the outcome card.
    await page.waitForTimeout(12_000);
    const safe = label.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    await page.screenshot({
      path: `${OUT}preset-${safe}.png`,
      fullPage: true,
    });
    await page.getByRole("button", { name: "Reset" }).click();
  }

  await page.getByRole("button", { name: "Compare" }).click();
  await page.waitForTimeout(20_000);
  await page.screenshot({ path: `${OUT}compare-recordings.png`, fullPage: true });
  console.log(`Saved screenshots to ${OUT}`);
} finally {
  await browser.close();
}
