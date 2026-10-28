const fs = require('node:fs');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const ROOT = 'E:/998/CPR+AGENT/CODE';
const BASE = 'http://127.0.0.1:8013';

(async () => {
  const output = `${ROOT}/data/mainline-delivery-v1/browser.json`;
  if (fs.existsSync(output)) throw new Error('Preserve browser acceptance');
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
  });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  const external = [];
  context.on('page', (page) => page.on('pageerror', (error) => errors.push(String(error))));
  context.on('request', (request) => {
    if (new URL(request.url()).origin !== BASE) external.push(request.url());
  });
  try {
    const page = await context.newPage();
    await page.goto(`${BASE}/demo`, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: /单视角 CPR/ }).waitFor();
    const links = await page.locator('a').evaluateAll((elements) => elements.map((element) => element.getAttribute('href')));
    for (const id of ['D01', 'D02', 'D03']) {
      if (!links.includes(`/demo/open/${id}`)) throw new Error(`case link missing: ${id}`);
    }
    await page.goto(`${BASE}/demo/open/D01`, { waitUntil: 'domcontentloaded' });
    await page.waitForURL(`${BASE}/`);
    const workspace = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace')));
    const runId = await page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
    if (!workspace?.session_id || !runId) throw new Error('D01 did not load a verified workspace');
    await page.reload({ waitUntil: 'networkidle' });
    await page.getByRole('complementary', { name: 'Practice review assistant' }).waitFor();
    await page.setViewportSize({ width: 390, height: 950 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    if (overflow) throw new Error('mobile horizontal overflow');
    await page.screenshot({ path: `${ROOT}/data/mainline-delivery-v1/browser-mobile.png`, fullPage: true });
    const result = { passed: true, case_links: 3, d01_loaded: true, refresh_restored: true, mobile_width: 390, browser_errors: errors, external_requests: external.length };
    fs.writeFileSync(output, JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
