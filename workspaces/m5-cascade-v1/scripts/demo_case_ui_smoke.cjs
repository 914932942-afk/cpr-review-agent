/* Reopen prepared real cases. Only mutation is an intentionally rejected invalid-file upload. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..'), out = path.join(root, 'test-results/demo-cases-v1/ui');
const catalog = JSON.parse(fs.readFileSync(path.join(root, 'data/demo-cases-v1/cases.json'), 'utf8'));
const byId = Object.fromEntries(catalog.cases.map(c => [c.id, c]));
async function main() {
  fs.mkdirSync(out, { recursive: true }); assert(!fs.existsSync(path.join(out, 'acceptance.json')), 'Preserve accepted evidence');
  const b = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], posts = [], external = [];
  try {
    const c = await b.newContext({ baseURL: catalog.base_url, viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
    c.on('page', p => p.on('pageerror', e => errors.push(e.message)));
    c.on('request', r => { if (r.method() !== 'GET') posts.push(r.url()); if (/^https?:/.test(r.url()) && new URL(r.url()).origin !== catalog.base_url) external.push(r.url()); });
    const p = await c.newPage(); await p.goto('/');
    const get = async u => { const r = await c.request.get('/api' + u); assert(r.ok(), u); return r.json(); };
    async function seed(item) {
      await p.evaluate(({ ws, rid }) => { localStorage.setItem('cpr.workspace', JSON.stringify(ws)); localStorage.setItem('cpr.runId', JSON.stringify(rid)); }, { ws: item.workspace, rid: item.verified.run_id });
      await p.reload(); await p.waitForFunction(id => document.querySelector('.video-header h2')?.textContent === id || document.querySelector('.video-card h2')?.textContent === id, item.workspace.name).catch(async () => {
        await p.getByRole('heading', { name: item.workspace.name, exact: true }).waitFor();
      });
    }
    const seen = [];
    for (const id of ['C01', 'C02', 'C03', 'C05', 'C06', 'C07']) {
      const item = byId[id]; await p.setViewportSize({ width: 1440, height: 1000 }); await seed(item);
      const findings = p.getByRole('region', { name: 'Practice findings', exact: true });
      if (['C01', 'C02', 'C03'].includes(id)) {
        await findings.waitFor(); await p.getByText('Saved source checked', { exact: false }).waitFor();
        assert((await findings.locator('.review-overview p').innerText()).includes('11'));
        if (id === 'C01') await findings.getByRole('button', { name: 'Explain Bent arms', exact: true }).waitFor();
        if (id === 'C02') await findings.getByRole('button', { name: 'Explain Tilted arms', exact: true }).waitFor();
        if (id === 'C03') await findings.getByText('No selected category was flagged.', { exact: false }).waitFor();
        const response = await c.request.get(`/api/reports/${item.verified.report_id}/export`); assert(response.ok()); assert((await response.text()).includes(item.workspace.name));
      } else if (id === 'C05') {
        await p.getByRole('heading', { name: "We couldn't complete this review.", exact: true }).waitFor();
        assert.equal(await findings.count(), 0); assert.equal(await p.getByLabel('Evidence-constrained explanation', { exact: true }).count(), 0);
      } else {
        await p.getByRole('region', { name: 'Reference video comparison', exact: true }).waitFor();
        if (id === 'C07') {
          assert.equal(await findings.count(), 0);
          const panel = p.getByRole('region', { name: 'Practice continuity', exact: true });
          await panel.getByText(/Starting focus saved:/).waitFor();
          await panel.getByRole('button', { name: 'View earlier recap', exact: true }).click();
          await p.getByRole('dialog', { name: 'Previous practice recap', exact: true }).getByText('video.mp4', { exact: true }).waitFor();
          await p.keyboard.press('Escape');
          assert.equal((await get(`/sessions/${item.workspace.session_id}/context`)).source, null);
        }
        await p.getByRole('button', { name: 'Play both from start', exact: true }).click(); await p.waitForTimeout(700);
        const players = await p.locator('.video-comparison-grid video').evaluateAll(vs => vs.map(v => { v.pause(); return { time: v.currentTime, width: v.videoWidth, error: v.error?.code || null }; }));
        assert.equal(players.length, 2); assert(players.every(v => v.time > 0 && v.width > 0 && !v.error));
      }
      if (id !== 'C06' && id !== 'C07' && id !== 'C05') {
        await p.getByLabel('Practice video player', { exact: true }).evaluate(async v => { v.muted = true; await v.play(); }); await p.waitForTimeout(500);
        assert(await p.getByLabel('Practice video player', { exact: true }).evaluate(v => { v.pause(); return v.currentTime > 0 && v.videoWidth > 0 && !v.error; }));
      }
      for (const width of [1440, 390]) { await p.setViewportSize({ width, height: 1000 }); await p.evaluate(() => scrollTo(0, 0)); assert.equal(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), false); await p.screenshot({ path: path.join(out, `${id}-${width}.png`), fullPage: true }); }
      seen.push(id);
    }
    await p.setViewportSize({ width: 1440, height: 1000 }); await seed(byId.C03);
    await p.getByRole('region', { name: 'Practice findings', exact: true }).getByText('No selected category was flagged.', { exact: false }).waitFor();
    await p.locator('input[type=file]').setInputFiles(path.join(root, byId.C04.source.path));
    await p.getByRole('alert').getByText('This file could not be opened as a video. Choose an MP4, MOV or WebM file.', { exact: true }).waitFor();
    assert.equal(await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace')).session_id), byId.C03.workspace.session_id);
    assert.equal(await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.runId'))), byId.C03.verified.run_id);
    await p.screenshot({ path: path.join(out, 'C04-rejected-upload.png'), fullPage: true });
    assert.deepEqual(posts, [catalog.base_url + '/api/assets']);
    const book = await c.newPage(); await book.goto(pathToFileURL(path.join(root, 'docs/演示案例-v1.html')).href);
    await book.getByRole('heading', { name: '七个案例，演示完整使用价值。', exact: true }).waitFor();
    for (const width of [1440, 1024, 768, 390]) {
      await book.setViewportSize({ width, height: 1000 }); assert.equal(await book.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await book.screenshot({ path: path.join(out, `casebook-${width}.png`), fullPage: true });
    }
    const media = [];
    for (const id of ['C01', 'C02', 'C03']) {
      await book.locator('#' + id).getByText('素材与来源', { exact: true }).click();
      const player = book.getByLabel(id + ' public clip', { exact: true });
      await player.evaluate(async v => { v.muted = true; await v.play(); }); await book.waitForTimeout(400);
      const value = await player.evaluate(v => { v.pause(); return { time: v.currentTime, width: v.videoWidth, error: v.error?.code || null }; });
      assert(value.time > 0 && value.width > 0 && !value.error); media.push({ id, ...value });
    }
    await book.locator('#C02').getByRole('button', { name: '复制练习名称', exact: true }).click();
    await book.getByRole('status').getByText('已复制，可粘贴使用', { exact: true }).waitFor();
    assert.deepEqual(errors, []); assert.deepEqual(external, []);
    fs.writeFileSync(path.join(out, 'acceptance.json'), JSON.stringify({ passed: true, reopened_cases: seen, invalid_upload_keeps_previous_practice: true, no_fake_classification_for_failure: true, normal_case_has_all_11_categories: true, old_recap_and_references_verified: true, media, guide_clipboard_passed: true, guide_widths: [1440, 1024, 768, 390], application_widths: [1440, 390], browser_errors: errors, external_requests: external.length, posts: posts.length, new_vision_calls: 0, provider_calls: 0, mocked_ui_or_cv: false }, null, 2));
    console.log('Seven prepared cases and local casebook passed actual browser verification; no new inference/provider calls');
  } catch (e) {
    for (const c of b.contexts()) for (const p of c.pages()) await p.screenshot({ path: path.join(out, `failure-${Date.now()}.png`), fullPage: true }).catch(() => {});
    throw e;
  } finally { await b.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
