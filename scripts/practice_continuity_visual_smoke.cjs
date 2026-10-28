/* Read-only final layout and source isolation check; no new CV or provider requests. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..'), out = path.join(root, 'test-results/practice-continuity-v1/visual-final');
async function main() {
  fs.mkdirSync(out, { recursive: true }); assert(!fs.existsSync(path.join(out, 'acceptance.json')));
  const b = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const errors = [], writes = [];
  try {
    const c = await b.newContext({ baseURL: 'http://127.0.0.1:8007', reducedMotion: 'reduce' });
    c.on('page', p => p.on('pageerror', e => errors.push(e.message)));
    c.on('request', r => { if (r.method() !== 'GET') writes.push(r.url()); });
    const p = await c.newPage(); await p.goto('/');
    for (const type of ['ui', 'live']) {
      const saved = JSON.parse(fs.readFileSync(path.join(root, `test-results/practice-continuity-v1/${type}/acceptance.json`), 'utf8'));
      await p.setViewportSize({ width: 1440, height: 1000 });
      await p.locator(`[data-session-id="${saved.workspace.session_id}"]`).click();
      await p.waitForFunction(id => JSON.parse(localStorage.getItem('cpr.runId')) === id, saved.analysis_run_id || saved.run_id);
      const panel = p.getByRole('region', { name: 'Practice continuity', exact: true });
      await panel.getByText(/Starting focus saved:/).waitFor();
      if (type === 'ui') {
        await p.getByRole('region', { name: 'Practice findings', exact: true }).waitFor();
        assert(await p.evaluate(() => document.querySelector('.findings-card').getBoundingClientRect().top < document.querySelector('.practice-continuity').getBoundingClientRect().top));
        assert.equal(await panel.getByRole('button', { name: 'Use this focus', exact: true }).count(), 0);
      } else {
        assert.equal(await p.getByRole('region', { name: 'Practice findings', exact: true }).count(), 0);
        assert(await panel.getByRole('button', { name: 'Use this focus', exact: true }).isDisabled());
        await p.getByRole('region', { name: 'Reference video comparison', exact: true }).waitFor();
      }
      for (const width of [1440, 1024, 768, 390]) {
        await p.setViewportSize({ width, height: 1000 }); await p.waitForTimeout(80); await p.evaluate(() => scrollTo(0, 0));
        assert.equal(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
        await p.screenshot({ path: path.join(out, `${type}-${width}.png`), fullPage: true });
      }
      await panel.getByRole('button', { name: 'View earlier recap', exact: true }).click();
      const modal = p.getByRole('dialog', { name: 'Previous practice recap', exact: true });
      await modal.getByText('video.mp4', { exact: true }).waitFor();
      assert.equal(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await p.screenshot({ path: path.join(out, `${type}-recap-390.png`), fullPage: true });
      await p.keyboard.press('Escape');
      assert.equal(await p.evaluate(() => JSON.parse(localStorage.getItem('cpr.workspace')).session_id), saved.workspace.session_id);
    }
    assert.deepEqual(errors, []); assert.deepEqual(writes, []);
    fs.writeFileSync(path.join(out, 'acceptance.json'), JSON.stringify({ passed: true, findings_before_history: true, review_and_unanalysed_practice_verified: true, widths: [1440, 1024, 768, 390], old_recap_does_not_switch_video: true, browser_errors: errors, new_posts: 0, new_vision_calls: 0, real_provider_requests: 0 }, null, 2));
    console.log('Final cross-practice layout and source isolation passed; read-only');
  } finally { await b.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
