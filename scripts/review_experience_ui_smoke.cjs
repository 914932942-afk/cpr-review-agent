/* Real product browser checks. CPU decoding only; no new CV runs or provider calls. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { chromium } = require('C:/Users/gjj33/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '..'), baseOut = path.join(root, 'test-results/review-experience-v1/ui');
const out = process.argv.includes('--recheck') ? path.join(baseOut, 'recheck') : baseOut;
const baseURL = 'http://127.0.0.1:8007', restore = process.argv.includes('--restore');
const book = JSON.parse(fs.readFileSync(path.join(root, 'data/demo-cases-v1/cases.json'), 'utf8'));
const cases = Object.fromEntries(book.cases.map(c => [c.id, c]));
const statePath = path.join(baseOut, 'state.json');
const saved = fs.existsSync(statePath) ? JSON.parse(fs.readFileSync(statePath, 'utf8')) : {};
const persist = () => fs.writeFileSync(statePath, JSON.stringify(saved, null, 2));
async function main() {
  fs.mkdirSync(out, {recursive:true});
  const target=path.join(out,restore?'restore.json':'acceptance.json');
  assert(!fs.existsSync(target),'Preserve accepted evidence');
  const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  const errors=[], external=[], mutations=[];
  try {
    const c=await browser.newContext({baseURL,viewport:{width:1440,height:1000},reducedMotion:'reduce'});
    c.on('page',p=>{p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>{if(m.type()==='error' && /same key|unique.*key/i.test(m.text()))errors.push(m.text());});});
    c.on('request',r=>{if(/^https?:/.test(r.url()) && new URL(r.url()).origin!==baseURL)external.push(r.url());if(r.method()!=='GET')mutations.push(r.url());});
    const p=await c.newPage();p.setDefaultTimeout(15000);await p.goto('/');
    const get=async u=>{const r=await c.request.get('/api'+u);assert(r.ok(),u);return r.json();};
    const post=async(u,value)=>{const r=await c.request.post('/api'+u,{data:value,headers:{'Idempotency-Key':crypto.randomUUID()}});assert(r.ok(),await r.text());return r.json();};
    const seed=async(ws,rid=null)=>{
      await p.evaluate(({ws,rid})=>{localStorage.setItem('cpr.workspace',JSON.stringify(ws));localStorage.setItem('cpr.runId',JSON.stringify(rid));},{ws,rid});await p.reload();
      await p.getByRole('heading',{name:ws.name,exact:true}).waitFor();
    };
    const panel=()=>p.getByRole('region',{name:'Personal review notes',exact:true});
    const note=()=>panel().getByLabel(/^Personal note/);
    const caps=await get('/capabilities');assert.equal(caps.upload_preflight,true);assert.equal(caps.review_workspace_tools,true);
    if(restore){
      assert(saved.source && saved.target);
      await seed(cases.C02.workspace,cases.C02.verified.run_id);
      const findings=p.getByRole('region',{name:'Practice findings',exact:true});
      await findings.getByRole('button',{name:'Explain Tilted arms',exact:true}).waitFor();
      await p.getByRole('button',{name:'Compare practice records',exact:true}).waitFor();
      assert.equal(await p.getByRole('button',{name:'Compare practice records',exact:true}).count(),1);
      await p.getByRole('button',{name:'Compare practice records',exact:true}).click();
      const comparison=p.getByRole('dialog',{name:'Practice record comparison',exact:true});
      await comparison.getByLabel('Compare with',{exact:true}).selectOption(cases.C01.verified.report_id);
      await comparison.getByRole('heading',{name:'11 shared categories',exact:true}).waitFor();
      await p.keyboard.press('Escape');
      await seed(cases.C03.workspace,cases.C03.verified.run_id);
      await p.getByText('No selected category was flagged.',{exact:false}).waitFor();
      await p.getByRole('button',{name:'Compare practice records',exact:true}).waitFor();
      assert.equal(await p.getByRole('button',{name:'Compare practice records',exact:true}).count(),1);
      await seed(saved.source);await panel().getByText('Saved for this practice',{exact:true}).waitFor();assert.equal(await note().inputValue(),saved.note);
      await panel().getByLabel('Status for Bent arms',{exact:true}).selectOption('next_focus');
      await seed(saved.target);await p.getByRole('button',{name:'Focus: Bent arms',exact:true}).waitFor();
      const ctx=await get(`/sessions/${saved.target.session_id}/context`);assert.equal(ctx.source,null);assert.deepEqual(ctx.active_label_ids,[4]);
      assert.equal(await note().inputValue(),'');
      await p.screenshot({path:path.join(out,'after-restart.png'),fullPage:true});
      const guide=await c.newPage();await guide.goto(require('node:url').pathToFileURL(path.join(root,'docs/演示案例-v2.html')).href);
      await guide.getByRole('heading',{name:'新增的三个用户功能',exact:true}).waitFor();
      await guide.getByText(/当前8007已启用上传预检/).waitFor();
      for(const width of [1440,390]){await guide.setViewportSize({width,height:1000});assert.equal(await guide.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await guide.screenshot({path:path.join(out,`casebook-v2-${width}.png`),fullPage:true});}
    }else{
      console.log('Compare two existing public-video reports');
      await seed(cases.C02.workspace,cases.C02.verified.run_id);
      await p.getByRole('button',{name:'Compare practice records',exact:true}).click();
      const modal=p.getByRole('dialog',{name:'Practice record comparison',exact:true});
      await modal.getByLabel('Compare with',{exact:true}).selectOption(cases.C01.verified.report_id);
      await modal.getByRole('heading',{name:'11 shared categories',exact:true}).waitFor();
      let row=modal.locator('tbody tr').filter({hasText:'Bent arms'});
      assert.deepEqual(await row.getByRole('cell').allTextContents(),['Flagged for review','Not flagged']);
      row=modal.locator('tbody tr').filter({hasText:'Tilted arms'});
      assert.deepEqual(await row.getByRole('cell').allTextContents(),['Not flagged','Flagged for review']);
      const playback=[];
      for(const label of ['Other practice video','Current practice video']){
        const player=modal.getByLabel(label,{exact:true});await player.evaluate(async v=>{v.muted=true;await v.play();});await p.waitForTimeout(350);
        const facts=await player.evaluate(v=>{v.pause();return {time:v.currentTime,width:v.videoWidth,error:v.error?.code||null};});assert(facts.time>0 && facts.width>0 && !facts.error);playback.push(facts);
      }
      for(const width of [1440,390]){await p.setViewportSize({width,height:1000});assert.equal(await modal.evaluate(e=>e.scrollWidth>e.clientWidth),false);await p.screenshot({path:path.join(out,`comparison-${width}.png`),fullPage:true});}
      await p.keyboard.press('Escape');await p.setViewportSize({width:1440,height:1000});
      assert.equal(await p.evaluate(()=>JSON.parse(localStorage.getItem('cpr.workspace')).session_id),cases.C02.workspace.session_id);
      console.log('Reject too-short upload before workspace/task creation');
      await p.locator('input[type=file]').setInputFiles(path.join(root,cases.C05.source.path));
      await p.getByRole('alert').getByText('This video is too short. Choose a complete clip with at least 8 frames.',{exact:true}).waitFor();
      assert.equal(await p.evaluate(()=>JSON.parse(localStorage.getItem('cpr.workspace')).session_id),cases.C02.workspace.session_id);
      await p.screenshot({path:path.join(out,'upload-rejected.png'),fullPage:true});
      async function upload(name,video){
        const old=await p.evaluate(()=>JSON.parse(localStorage.getItem('cpr.workspace'))?.session_id);
        await p.locator('input[type=file]').setInputFiles({name,mimeType:'video/mp4',buffer:fs.readFileSync(path.join(root,video))});
        await p.waitForFunction(old=>JSON.parse(localStorage.getItem('cpr.workspace'))?.session_id!==old,old,{timeout:45000});
        const ws=await p.evaluate(()=>JSON.parse(localStorage.getItem('cpr.workspace')));
        await p.getByText(/File check passed ·/).waitFor();assert.equal((await get('/assets/'+ws.asset_id)).validation_level,'decoded_before_session');return ws;
      }
      if(!saved.source){saved.source=await upload('checklist-practice.mp4',cases.C01.source.video_rel);persist();}else await seed(saved.source);
      await panel().getByRole('button',{name:'Save notes',exact:true}).waitFor();
      let initial=await get(`/sessions/${saved.source.session_id}/notebook`);
      if(initial.revision){await post(`/sessions/${saved.source.session_id}/notebook`,{revision:initial.revision,note:'',items:[]});await p.reload();await panel().getByRole('button',{name:'Save notes',exact:true}).waitFor();}
      console.log('Notes: local draft recovery, private per-practice scope, optimistic concurrency');
      const draft='Check my arm position next time. <script>plain note, never executed</script>';
      await note().fill(draft);await panel().getByLabel('Checklist category',{exact:true}).selectOption('4');await panel().getByRole('button',{name:'Add item',exact:true}).click();
      await panel().getByLabel('Status for Bent arms',{exact:true}).selectOption('next_focus');
      await p.waitForFunction(sid=>JSON.parse(localStorage.getItem('cpr.notebookDraft.'+sid)||'null')?.items?.[0]?.status==='next_focus',saved.source.session_id);
      await p.reload();await panel().getByText('Unsaved draft · kept in this browser',{exact:true}).waitFor();assert.equal(await note().inputValue(),draft);
      await panel().getByRole('button',{name:'Save notes',exact:true}).click();await panel().getByText('Saved for this practice',{exact:true}).waitFor();
      const first=await get(`/sessions/${saved.source.session_id}/notebook`);assert.equal(first.note,draft);assert.deepEqual(first.items,[{label_id:4,status:'next_focus'}]);
      await note().fill('Local draft should survive a revision conflict.');
      const conflictRevision=await post(`/sessions/${saved.source.session_id}/notebook`,{revision:first.revision,note:'Saved in another window.',items:first.items});
      await panel().getByRole('button',{name:'Save notes',exact:true}).click();
      await panel().getByText(/These notes changed in another window/).waitFor();
      assert.equal(await note().inputValue(),'Local draft should survive a revision conflict.');
      assert.equal((await get(`/sessions/${saved.source.session_id}/notebook`)).revision,conflictRevision.revision);
      p.once('dialog',d=>d.accept());await panel().getByRole('button',{name:'Reload saved notes',exact:true}).click();
      await panel().getByText('Saved for this practice',{exact:true}).waitFor();assert.equal(await note().inputValue(),'Saved in another window.');
      saved.note='Revisit Bent arms before the next practice. Personal checklist only.';
      await note().fill(saved.note);await panel().getByLabel('Checklist category',{exact:true}).selectOption('5');await panel().getByRole('button',{name:'Add item',exact:true}).click();await panel().getByLabel('Status for Tilted arms',{exact:true}).selectOption('reviewed');
      await panel().getByRole('button',{name:'Save notes',exact:true}).click();await panel().getByText('Saved for this practice',{exact:true}).waitFor();persist();
      for(const width of [1440,768,390]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await panel().scrollIntoViewIfNeeded();await p.screenshot({path:path.join(out,`notebook-${width}.png`),fullPage:true});}
      assert.equal(await p.getByRole('region',{name:'Practice continuity',exact:true}).count(),1,'One continuity panel after repeated note saves');
      assert.equal(await panel().count(),1,'One notebook panel');
      await p.setViewportSize({width:1440,height:1000});
      if(!saved.target){saved.target=await upload('next-practice-from-checklist.mp4',cases.C03.source.video_rel);persist();}else await seed(saved.target);
      assert.equal(await note().inputValue(),'');
      const before=await get(`/sessions/${saved.target.session_id}/context`);assert.equal(before.source,null);assert.deepEqual(before.active_label_ids,[]);
      const continuity=p.getByRole('region',{name:'Practice continuity',exact:true});await continuity.getByLabel('Saved next-practice goals',{exact:true}).selectOption(saved.source.session_id);
      await continuity.getByText('Your chosen goals: Bent arms',{exact:true}).waitFor();
      await continuity.getByRole('button',{name:'Use these goals',exact:true}).click();
      await p.getByRole('button',{name:'Focus: Bent arms',exact:true}).waitFor();
      let ctx=await get(`/sessions/${saved.target.session_id}/context`);assert.equal(ctx.source,null);assert.deepEqual(ctx.active_label_ids,[4]);assert.deepEqual(ctx.recent_goals,[]);assert(!JSON.stringify(ctx).includes(saved.note));
      await continuity.getByRole('button',{name:'Start fresh',exact:true}).click();await p.getByRole('button',{name:'Focus: All categories',exact:true}).waitFor();
      await continuity.getByRole('button',{name:'Use these goals',exact:true}).click();await p.getByRole('button',{name:'Focus: Bent arms',exact:true}).waitFor();
      await p.reload();await p.getByRole('button',{name:'Focus: Bent arms',exact:true}).waitFor();
      assert.equal(await p.getByRole('region',{name:'Practice findings',exact:true}).count(),0);
      assert.equal((await get('/sessions/'+saved.target.session_id+'/history')).items.length,0);
      for(const width of [1440,390]){await p.setViewportSize({width,height:1000});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await continuity.scrollIntoViewIfNeeded();await p.screenshot({path:path.join(out,`next-practice-${width}.png`),fullPage:true});}
      saved.playback=playback;persist();
    }
    assert.equal(await p.getByRole('region',{name:'Practice continuity',exact:true}).count(),1);
    assert.equal(await panel().count(),1);
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
    assert(!mutations.some(u=>/\/turns$|\/resume$|\/llm\/probe$/.test(u)));
    fs.writeFileSync(target,JSON.stringify({passed:true,restore,real_browser:true,mocked_apis:false,source_workspace:saved.source,target_workspace:saved.target,playback:saved.playback,shared_scope_comparison:true,upload_preflight_before_session:true,notes_revision_conflict_detected:true,draft_and_scope_isolation:true,explicit_goal_adoption:true,browser_errors:errors,external_requests:external.length,new_vision_calls:0,provider_calls:0},null,2));
    console.log('Review experience '+(restore?'restart recovery':'product acceptance')+' passed');
  }catch(e){for(const c of browser.contexts())for(const p of c.pages())await p.screenshot({path:path.join(out,`failure-${Date.now()}.png`),fullPage:true}).catch(()=>{});throw e;}finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
