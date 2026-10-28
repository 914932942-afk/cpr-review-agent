/* Real saved M4 routes, actual video playback, source scope and responsive UI. Read-only APIs. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test-results/m4-scheduling-v1/ui'),baseURL='http://127.0.0.1:8011';
async function main(){
  const state=JSON.parse(fs.readFileSync(path.join(root,'test-results/m4-scheduling-v1/http/state.json'),'utf8'));
  fs.mkdirSync(out,{recursive:true});assert(!fs.existsSync(path.join(out,'acceptance.json')),'Preserve browser evidence');
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const errors=[],external=[],writes=[];
  try{
    const c=await browser.newContext({baseURL,viewport:{width:1440,height:1000},reducedMotion:'reduce'});
    c.on('request',r=>{if(/^https?:/.test(r.url()) && new URL(r.url()).origin!==baseURL)external.push(r.url());if(r.method()!=='GET')writes.push(r.url())});
    const p=await c.newPage();p.on('pageerror',e=>errors.push(e.message));await p.goto('/');
    const cases=[['focused',state.body,state.body.run_id],['expanded',state.body,state.body.expand_run_id],['full',state.full,state.full.run_id]];
    for(const [name,row,rid] of cases){
      await p.evaluate(({row,rid})=>{localStorage.setItem('cpr.workspace',JSON.stringify({session_id:row.session_id,clip_id:row.clip_id,asset_id:row.asset_id,name:row.name}));localStorage.setItem('cpr.runId',JSON.stringify(rid));},{row,rid});await p.reload();
      const findings=p.getByRole('region',{name:'Practice findings',exact:true});await findings.waitFor();await p.getByText('Saved source checked',{exact:false}).waitFor();
      if(name==='focused'){
        await p.getByRole('region',{name:'Scope of this analysis',exact:true}).waitFor();
        await p.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Expand to all categories')&&!b.disabled));
        assert.equal(await findings.getByRole('button',{name:'Explain Clenched hands',exact:true}).count(),0);
        await findings.getByRole('button',{name:'Explain Bent arms',exact:true}).waitFor();
      }else assert.equal(await p.getByRole('region',{name:'Scope of this analysis',exact:true}).count(),0);
      if(name==='full')await findings.getByRole('region',{name:'Categories needing another look',exact:true}).waitFor();
      const player=p.getByLabel('Practice video player',{exact:true});await player.evaluate(async v=>{v.muted=true;await v.play()});await p.waitForTimeout(300);assert(await player.evaluate(v=>{v.pause();return v.currentTime>0&&v.videoWidth>0&&!v.error}));
      await p.getByRole('button',{name:'Technical details',exact:true}).click();await p.getByText('Algorithm modules',{exact:true}).click();await p.getByRole('dialog').locator('p').filter({hasText:'goal-cost-dag-v1'}).waitFor();await p.keyboard.press('Escape');
      for(const width of [1440,390]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await p.screenshot({path:path.join(out,`${name}-${width}.png`),fullPage:true})}await p.setViewportSize({width:1440,height:1000});
    }
    const comparison=await c.request.get(`/api/review-comparison?left_report_id=${state.body.report_id}&right_report_id=${state.full.report_id}`);assert(comparison.ok());assert.deepEqual((await comparison.json()).common_label_ids,[4]);
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);assert.deepEqual(writes,[]);
    fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:true,real_browser:true,read_only_api:true,mocked_apis:false,restored_routes:3,partial_scope_explicit:true,expand_control_enabled:true,uncertain_state_visible:true,partial_full_comparison_common_scope:[4],widths:[1440,390],browser_errors:errors,provider_calls:0,new_vision_executions:0},null,2));console.log('M4 focused/expanded/full UI, scope and real playback passed');
  }catch(e){await pSafeShot(browser);throw e}finally{await browser.close()}
}
async function pSafeShot(browser){for(const c of browser.contexts())for(const p of c.pages())await p.screenshot({path:path.join(out,`failure-${Date.now()}.png`),fullPage:true}).catch(()=>{})}
main().catch(e=>{console.error(e);process.exitCode=1});
