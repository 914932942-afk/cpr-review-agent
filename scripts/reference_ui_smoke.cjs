/* Real saved-video UI and reference playback. No API/CV mocks and no provider calls. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'test-results/reference-v1/ui'),baseURL='http://127.0.0.1:8007';
async function main(){
 fs.mkdirSync(out,{recursive:true});assert(!fs.existsSync(path.join(out,'acceptance.json')),'Preserve accepted evidence');
 const source=JSON.parse(fs.readFileSync(path.join(root,'test-results/product-demo-v2/browser/acceptance.json'),'utf8'));
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const errors=[],external=[];
 try{
  const c=await browser.newContext({baseURL,viewport:{width:1440,height:1000},reducedMotion:'reduce'});c.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));c.on('request',r=>{if(new URL(r.url()).origin!==baseURL)external.push(r.url());});
  const get=async u=>{const r=await c.request.get('/api'+u);assert(r.ok(),u);return r.json();};
  assert.deepEqual((await get('/capabilities')).reference_tools,['reference.search']);
  const p=await c.newPage();await p.goto('/');
  const seed=async(id)=>{await p.evaluate(({ws,id})=>{localStorage.setItem('cpr.workspace',JSON.stringify(ws));localStorage.setItem('cpr.runId',JSON.stringify(id));},{ws:source.workspace,id});await p.reload();};
  const id=()=>p.evaluate(()=>JSON.parse(localStorage.getItem('cpr.runId')));
  async function reference(){const old=await id();await p.getByRole('button',{name:'Compare references',exact:true}).click();await p.waitForFunction(old=>JSON.parse(localStorage.getItem('cpr.runId'))!==old,old);const rid=await id();let run;for(let n=0;n<150;n++){run=await get('/runs/'+rid);if(run.status==='completed')break;assert(!['failed','rejected','cancelled'].includes(run.status));await p.waitForTimeout(100);}assert.equal(run.status,'completed');assert.equal(run.reference_requested,true);assert.equal(run.response.prediction,null);assert.deepEqual(run.response.classification_claims,[]);assert.deepEqual(await get('/runs/'+rid+'/tools'),[]);await p.getByRole('region',{name:'Reference video comparison',exact:true}).waitFor();return run;}
  await seed(source.summary_run_id);await p.getByRole('heading',{name:'Your review recap',exact:true}).waitFor();
  const first=await reference();let data=await get('/runs/'+first.id+'/references');
  assert.deepEqual(data.requested_label_ids,[4]);assert.deepEqual(data.items.map(i=>i.label_id),[0,4]);assert.equal(data.items[1].id,'cpr-c04-r01-ch0');
  assert.equal(await p.getByRole('region',{name:'Practice findings'}).count(),1);
  const original=await get('/runs/'+source.analysis_run_id);assert.deepEqual(original.response.prediction,source.prediction);
  await p.getByRole('button',{name:'Play both from start',exact:true}).click();await p.waitForTimeout(800);
  const playback=await p.locator('.video-comparison-grid video').evaluateAll(players=>players.map(v=>{v.pause();return{time:v.currentTime,width:v.videoWidth,height:v.videoHeight,error:v.error?.code||null};}));assert.equal(playback.length,2);assert(playback.every(v=>v.time>0&&v.width>0&&!v.error));
  await p.getByText('Source and use information',{exact:false}).click();assert.equal(await p.getByRole('link',{name:'Dataset source',exact:false}).getAttribute('href'),'https://huggingface.co/datasets/ShunliWang/CPR-Coach');
  await p.evaluate(()=>scrollTo(0,0));await p.screenshot({path:path.join(out,'comparison-1440.png'),fullPage:true});
  await p.getByLabel('Reference clip',{exact:true}).selectOption(data.items[1].id);await p.getByText('Dataset-labelled “Bending Arms”. An error example.',{exact:true}).waitFor();
  const range=await c.request.get(`/api/runs/${first.id}/reference-clips/${data.items[1].id}/media`,{headers:{Range:'bytes=0-1023'}});assert.equal(range.status(),206);
  assert.equal((await c.request.get(`/api/runs/${first.id}/reference-clips/cpr-c05-r00-ch0/media`)).status(),410);
  await p.getByRole('button',{name:'Close comparison',exact:true}).click();assert.equal(await p.getByLabel('Dataset reference video',{exact:true}).count(),0);
  await p.getByRole('button',{name:'Open side-by-side',exact:true}).click();await p.getByLabel('Dataset reference video',{exact:true}).waitFor();
  await p.reload();await p.getByRole('region',{name:'Reference video comparison',exact:true}).waitFor();assert.equal(await id(),first.id);
  for(const width of [1024,768,390]){await p.setViewportSize({width,height:1000});await p.waitForTimeout(100);assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await p.evaluate(()=>scrollTo(0,0));await p.screenshot({path:path.join(out,`comparison-${width}.png`),fullPage:true});}
  await p.getByRole('button',{name:'Play both from start',exact:true}).click();await p.waitForTimeout(500);assert(await p.getByLabel('Dataset reference video',{exact:true}).evaluate(v=>v.currentTime>0&&!v.error));await p.locator('.video-comparison-grid video').evaluateAll(vs=>vs.forEach(v=>v.pause()));
  await p.setViewportSize({width:1440,height:1000});
  await p.locator('input[type=file]').setInputFiles({name:'new-practice-no-reference.mp4',mimeType:'video/mp4',buffer:fs.readFileSync(path.join(root,'data/public_cpr/rgb-acceptance-v1/records/SupC04--r00--ch0/video.mp4'))});
  await p.waitForFunction(old=>JSON.parse(localStorage.getItem('cpr.workspace'))?.session_id!==old,source.workspace.session_id);
  assert.equal(await p.getByRole('region',{name:'Reference video comparison',exact:true}).count(),0);assert.equal(await p.getByRole('region',{name:'Reference search result',exact:true}).count(),0);
  await seed(source.summary_run_id);await p.getByRole('heading',{name:'Your review recap',exact:true}).waitFor();const final=await reference();
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:true,workspace:source.workspace,source_analysis_run_id:source.analysis_run_id,source_summary_run_id:source.summary_run_id,reference_run_id:final.id,first_reference_run_id:first.id,requested_labels:[4],reference_ids:data.items.map(i=>i.id),normal_and_error_reference_distinct:true,identical_practice_video_excluded:true,media_range_status:206,context_cleared_for_new_video:true,refresh_restored:true,playback,widths:[1440,1024,768,390],browser_errors:errors,external_browser_requests:external.length,new_vision_calls:0,paid_calls:0,mocked_ui_or_cv:false},null,2));
  console.log('Reference UI passed: source-bound lookup, actual side-by-side playback, switch/close/restore, new-video isolation; zero vision/provider calls');
 }catch(e){for(const c of browser.contexts())await c.pages()[0]?.screenshot({path:path.join(out,'failure-'+Date.now()+'.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
