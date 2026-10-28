/* Current-mainline offline UI acceptance. Reuses frozen public-video reports; no CV calls. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const ROOT = path.resolve(__dirname, '..');
const BASE = 'http://127.0.0.1:8013';
const state = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/mainline-delivery-v1/demo-state.json'), 'utf8'));
const output = path.join(ROOT, 'data/mainline-delivery-v1/mainline-ui-acceptance.json');

async function main() {
  assert(!fs.existsSync(output), 'Preserve existing UI acceptance');
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    headless: true,
  });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
  const errors = [];
  const external = [];
  context.on('page', (page) => page.on('pageerror', (error) => errors.push(String(error))));
  context.on('request', (request) => {
    if (new URL(request.url()).origin !== BASE) external.push(request.url());
  });
  const api = context.request;
  const get = async (url) => {
    const response = await api.get(`${BASE}/api${url}`);
    assert(response.ok(), `${url}: ${response.status()}`);
    return response.json();
  };
  const waitRun = async (page, runId, seconds = 30) => {
    const started = Date.now();
    while (Date.now() - started < seconds * 1000) {
      const run = await get(`/runs/${runId}`);
      if (['completed', 'partially_completed', 'failed', 'rejected', 'cancelled'].includes(run.status)) {
        assert.equal(run.status, 'completed', JSON.stringify(run));
        return run;
      }
      await page.waitForTimeout(250);
    }
    throw new Error(`Timed out waiting for ${runId}`);
  };
  const currentRun = (page) => page.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId')));
  try {
    const health = await get('/health');
    assert.equal(health.llm, 'disabled');
    const page = await context.newPage();
    await page.goto(`${BASE}/demo/open/D01`, { waitUntil: 'domcontentloaded' });
    await page.waitForURL(`${BASE}/`);
    await page.getByRole('region', { name: 'Practice findings', exact: true }).waitFor();
    const initialRun = await currentRun(page);
    assert.equal(initialRun, state['D01-full'].run_id);
    const video = page.getByLabel('Practice video player', { exact: true });
    await video.evaluate(async (element) => { element.muted = true; await element.play(); });
    await page.waitForTimeout(700);
    const playback = await video.evaluate((element) => ({ time: element.currentTime, width: element.videoWidth, height: element.videoHeight, error: element.error?.code || null }));
    assert(playback.time > 0 && playback.width > 0 && !playback.error);
    const explanationButton = page.getByRole('button', { name: /Explain/ }).first();
    await explanationButton.click();
    await page.waitForFunction((old) => JSON.parse(localStorage.getItem('cpr.runId')) !== old, initialRun);
    const explanationRunId = await currentRun(page);
    const explanationRun = await waitRun(page, explanationRunId);
    assert.deepEqual(await get(`/runs/${explanationRunId}/tools`), []);
    await page.getByLabel('Evidence-constrained explanation', { exact: true }).waitFor();
    await page.getByText('Saved source checked', { exact: false }).waitFor();
    const findings = page.getByRole('region', { name: 'Practice findings', exact: true });
    await findings.getByRole('button', { name: 'Make a recap', exact: true }).click();
    await page.waitForFunction((old) => JSON.parse(localStorage.getItem('cpr.runId')) !== old, explanationRunId);
    const recapRunId = await currentRun(page);
    const recapRun = await waitRun(page, recapRunId);
    assert.deepEqual(await get(`/runs/${recapRunId}/tools`), []);
    await page.getByRole('heading', { name: 'Your review recap', exact: true }).waitFor();
    const exportLink = page.locator('a.export-review');
    const exportHref = await exportLink.getAttribute('href');
    assert(exportHref && exportHref.includes(`/api/reports/${recapRun.report_id}/export`));
    const exportResponse = await api.get(`${BASE}${exportHref}`);
    assert.equal(exportResponse.status(), 200);
    assert((await exportResponse.text()).includes('Evidence-constrained review'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.getByRole('heading', { name: 'Your review recap', exact: true }).waitFor();
    assert.equal(await currentRun(page), recapRunId);
    await page.setViewportSize({ width: 390, height: 900 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({ path: path.join(ROOT, 'data/mainline-delivery-v1/mainline-ui-mobile.png'), fullPage: true });

    await page.goto(`${BASE}/demo/open/D02`, { waitUntil: 'domcontentloaded' });
    await page.waitForURL(`${BASE}/`);
    await page.getByRole('region', { name: 'Practice findings', exact: true }).waitFor();
    const bodyRun = await get(`/runs/${state['D02-body'].run_id}`);
    const fullRun = await get(`/runs/${state['D02-full'].run_id}`);
    const bodyTools = (await get(`/runs/${state['D02-body'].run_id}/tools`)).map((tool) => tool.request.tool_id);
    const fullTools = (await get(`/runs/${state['D02-full'].run_id}/tools`)).map((tool) => tool.request.tool_id);
    assert.deepEqual(bodyTools, ['pose.extract', 'skeleton.predict', 'fusion.predict']);
    assert.deepEqual(fullTools, ['rgb.predict', 'fusion.predict']);
    const scheduling = fullRun.response.scheduling;
    assert.equal(scheduling.algorithm, undefined); // algorithm is recorded inside each decision audit
    assert(scheduling.history.some((item) => item.audit?.selected?.profile_id === 'full-v15'));
    assert(scheduling.history.some((item) => item.audit?.covered_labels?.length));
    await page.screenshot({ path: path.join(ROOT, 'data/mainline-delivery-v1/d02-agent-routing.png'), fullPage: true });
    const result = {
      passed: true,
      default_llm_mode: health.llm,
      initial_run_id: initialRun,
      explanation_run_id: explanationRunId,
      recap_run_id: recapRunId,
      playback,
      explanation_reused_visual_tools: true,
      recap_reused_visual_tools: true,
      export_passed: true,
      refresh_restored_recap: true,
      mobile_width: 390,
      d02_body_tools: bodyTools,
      d02_expansion_tools: fullTools,
      d02_scheduling_history_entries: scheduling.history.length,
      d02_body_profile: bodyRun.response.scheduling.history.find((item) => item.audit?.selected?.profile_id)?.audit?.selected?.profile_id || null,
      d02_expansion_profile: scheduling.history.find((item) => item.audit?.selected?.profile_id === 'full-v15')?.audit?.selected?.profile_id || null,
      new_vision_calls: 0,
      browser_errors: errors,
      external_browser_requests: external.length,
    };
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    fs.writeFileSync(output, JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
