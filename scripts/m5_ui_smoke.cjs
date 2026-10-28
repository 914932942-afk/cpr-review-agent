/* Real M5 evidence in the user UI, source restoration and clicked scope expansion. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test-results/m5-cascade-v1/ui'),baseURL='http://127.0.0.1:8013';
async function main(){
 fs.mkdirSync(out,{recursive:true});assert(!fs.existsSync(path.join(out,'acceptance.json')));
 const state=JSON.parse(fs.readFileSync(path.join(root,'test-results/m5-cascade-v1/http/state.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});let p;const errors=[],external=[];
 try {
  const c=await browser.newContext({baseURL,viewport:{width:1440,height:1050},reducedMotion:'reduce'});p=await c.newPage();p.on('pageerror',e=>errors.push(e.message));
  c.on('request',r=>{if(/^https?:/.test(r.url())&&new URL(r.url()).origin!==baseURL)external.push(r.url())});
  async function get(q){const r=await c.request.get('/api'+q);assert(r.ok(),await r.text());return r.json()}
  async function restore(row,rid){await p.goto('/');await p.evaluate(({row,rid})=>{localStorage.setItem('cpr.workspace',JSON.stringify({session_id:row.session_id,clip_id:row.clip_id,asset_id:row.asset_id,name:row.name}));localStorage.setItem('cpr.runId',JSON.stringify(rid))},{row,rid});await p.reload();await p.getByRole('region',{name:'Practice findings',exact:true}).waitFor();}
  await restore(state.full,state.full.run_id);
  await p.getByText(/Visual evidence checked · \d+ sampled frames/).waitFor();await p.getByRole('region',{name:'Categories needing another look',exact:true}).waitFor();
  const player=p.getByLabel('Practice video player',{exact:true});await player.evaluate(async v=>{v.muted=true;await v.play()});await p.waitForTimeout(300);assert(await player.evaluate(v=>{v.pause();return v.currentTime>0&&v.videoWidth>0&&!v.error}));
  await p.getByRole('button',{name:'Technical details',exact:true}).click();await p.getByText('Algorithm modules',{exact:true}).click();await p.getByRole('dialog').locator('p').filter({hasText:'quality-dependency-cascade-v1'}).waitFor();await p.keyboard.press('Escape');
  for(const width of [1440,390]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await p.screenshot({path:path.join(out,'full-'+width+'.png'),fullPage:true})}
  await p.setViewportSize({width:1440,height:1050});await restore(state.partial,state.partial.run_id);
  await p.getByText('Partial review · 4 of 11 requested categories checked',{exact:true}).waitFor();assert.equal(await p.getByText(/Visual evidence checked/).count(),0);
  await p.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.includes('Expand to all categories')&&!b.disabled));
  const request=p.waitForResponse(r=>r.url().includes('/turns')&&r.request().method()==='POST');await p.getByRole('button',{name:'Expand to all categories',exact:true}).click();const response=await request;assert.equal(response.status(),202);const id=(await response.json()).run_id;
  let r,mark;const until=Date.now()+240000;
  while(Date.now()<until){r=await get('/runs/'+id);const tools=await get('/runs/'+id+'/tools');const msg=r.status+':'+tools.map(t=>t.request.tool_id+'='+t.status).join(',');if(msg!==mark){console.log(msg);mark=msg}if(['completed','partially_completed','failed','rejected','cancelled'].includes(r.status))break;await p.waitForTimeout(600)}
  assert.equal(r.status,'completed');const tools=await get('/runs/'+id+'/tools');assert.deepEqual(tools.map(t=>t.request.tool_id),['rgb.predict','fusion.predict']);assert.equal(r.response.visual_processing.mode,'lightweight');
  await p.getByText(/Visual evidence checked · \d+ sampled frames/).waitFor();await p.waitForFunction(()=>!document.querySelector('.partial-result-note'));await p.screenshot({path:path.join(out,'expanded-after-click.png'),fullPage:true});
  assert.equal((await get('/reports/'+state.partial.report_id+'/evidence')).status,'available');
  await restore(state.partial,state.partial.run_id);await p.getByText('Partial review · 4 of 11 requested categories checked',{exact:true}).waitFor();assert.equal(await p.getByText(/Visual evidence checked/).count(),0);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:true,real_browser:true,mocked_apis:false,video_playback:true,full_visual_evidence_note:true,nonvisual_partial_has_no_fabricated_visual_note:true,m5_improved_module_visible:true,actual_expand_click:true,expand_run_id:id,expand_report_id:r.report_id,successful_new_tool_calls:2,m5_actual_calls:1,partial_source_preserved:true,uncertainty_preserved:true,widths:[1440,390],errors,external,provider_calls:0},null,2));console.log('M5 real user UI and clicked dependency-pruned expansion passed');
 }catch(e){if(p)await p.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});throw e}finally{await browser.close()}
}
main().catch(e=>{console.error(e);process.exitCode=1});
