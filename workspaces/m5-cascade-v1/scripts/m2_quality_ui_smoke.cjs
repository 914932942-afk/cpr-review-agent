/* Saved actual M2 runs, real video playback, and shared-scope comparison. No mocked APIs. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test-results/m2-quality-v1/ui'),baseURL='http://127.0.0.1:8008';
const state=JSON.parse(fs.readFileSync(path.join(root,'test-results/m2-quality-v1/http/state.json'),'utf8'));
async function main(){
  fs.mkdirSync(out,{recursive:true});assert(!fs.existsSync(path.join(out,'acceptance.json')),'Preserve accepted browser evidence');
  const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  const errors=[],external=[],mutations=[];
  try{
    const c=await browser.newContext({baseURL,viewport:{width:1440,height:1000},reducedMotion:'reduce'});
    c.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));
    c.on('request',r=>{if(/^https?:/.test(r.url()) && new URL(r.url()).origin!==baseURL)external.push(r.url());if(r.method()!=='GET')mutations.push(r.url());});
    const p=await c.newPage();await p.goto('/');
    const cases=Object.values(state);
    for(const [index,r] of cases.entries()){
      const ws={session_id:r.session_id,asset_id:r.asset_id,clip_id:r.clip_id,name:r.name};
      await p.evaluate(({ws,rid})=>{localStorage.setItem('cpr.workspace',JSON.stringify(ws));localStorage.setItem('cpr.runId',JSON.stringify(rid));},{ws,rid:r.run_id});await p.reload();
      const findings=p.getByRole('region',{name:'Practice findings',exact:true});await findings.waitFor();
      await p.getByText('Saved source checked',{exact:false}).waitFor();
      if(index===0)await findings.getByRole('button',{name:'Explain Bent arms',exact:true}).waitFor();
      else await findings.getByText('No selected category was flagged.',{exact:false}).waitFor();
      const player=p.getByLabel('Practice video player',{exact:true});await player.evaluate(async v=>{v.muted=true;await v.play();});await p.waitForTimeout(350);
      assert(await player.evaluate(v=>{v.pause();return v.currentTime>0 && v.videoWidth>0 && !v.error;}));
      await p.getByRole('button',{name:'Technical details',exact:true}).click();
      await p.getByText('Algorithm modules',{exact:true}).click();
      await p.getByText('quality-gated-disagreement-v1',{exact:false}).waitFor();
      await p.keyboard.press('Escape');
      for(const width of [1440,390]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await p.screenshot({path:path.join(out,`case-${index}-${width}.png`),fullPage:true});}
      await p.setViewportSize({width:1440,height:1000});
    }
    await p.getByRole('button',{name:'Compare practice records',exact:true}).click();
    const modal=p.getByRole('dialog',{name:'Practice record comparison',exact:true});
    const offered=await c.request.get(`/api/reports/${cases[1].report_id}/comparison-options`);assert(offered.ok());
    const previous=(await offered.json()).items.find(r=>r.session_id===cases[0].session_id && r.source_available && r.scope_label_ids.length===11);
    assert(previous,'Choose the latest verified full-scope report offered for the earlier practice');
    await modal.getByLabel('Compare with',{exact:true}).selectOption(previous.report_id);
    await modal.getByRole('heading',{name:'11 shared categories',exact:true}).waitFor();
    assert.deepEqual(await modal.locator('tbody tr').filter({hasText:'Bent arms'}).getByRole('cell').allTextContents(),['Flagged for review','Not flagged']);
    await p.screenshot({path:path.join(out,'comparison.png'),fullPage:true});await p.keyboard.press('Escape');
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);assert.deepEqual(mutations,[]);
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:true,real_browser:true,mocked_apis:false,restored_cases:cases.length,m2_registration_visible:true,real_video_playback:true,shared_scope_comparison:true,widths:[1440,390],browser_errors:errors,external_requests:external.length,http_mutations:mutations.length,new_model_executions:0,provider_calls:0},null,2));
    console.log('M2 real browser replay and comparison passed');
  }catch(e){for(const c of browser.contexts())for(const p of c.pages())await p.screenshot({path:path.join(out,`failure-${Date.now()}.png`),fullPage:true}).catch(()=>{});throw e;}finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
