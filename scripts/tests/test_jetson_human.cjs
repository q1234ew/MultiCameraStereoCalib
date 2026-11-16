// Run with: NODE_PATH=<directory containing playwright> node --test scripts/tests/test_jetson_human.cjs
// Uses a local HTTP fixture; never connects to cameras or changes runtime settings.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {createServer}=require('node:http');
const {readFile}=require('node:fs/promises');
const path=require('node:path');
const artifactDirectory=require('node:os').tmpdir();
let chromium;
try { ({chromium}=require('playwright')); } catch (_) { /* Optional browser dependency. */ }

function snapshot(){
  return {state:'running',session_id:'test-session',cameras:[],
    cross_camera:{state:'blocked',geometry_ready:true,world_frame:'floor:test',reason:'测试：未核验时钟',
      camera_views:[{camera_id:'device_1',world_frame:'floor:test',camera_xyz_m:[0,0,2],
                     polygon_xy_m:[[-2,-1],[3,-1],[3,3],[-2,3]],clip_extent_m:20}]},
    tracks:[{track_uuid:'person-a',global_id:null,observed:true,world_xy_m:[2,1],
      sources:[{camera_id:'device_1',epoch:'one',local_id:1,age_s:0}],
      trajectory:[{age_s:7,world_xy_m:[0,1]},{age_s:4,world_xy_m:[1,1]},
                  {age_s:1,world_xy_m:[1.5,1]},{age_s:0,world_xy_m:[2,1]}]}]};
}

test('real-time trails: controls, views, identity reset, disconnect and mobile layout',
  {skip:!chromium&&'Playwright is not installed; set NODE_PATH to an existing installation'},async()=>{
  let data=snapshot();
  const assets=path.resolve(__dirname,'../jetson/web');
  const server=createServer(async(req,res)=>{
    if(req.url==='/api/tracking/state'){
      res.setHeader('Content-Type','application/json');res.end(JSON.stringify({...data,updated_at_unix_ms:Date.now()}));return;
    }
    if(req.url.startsWith('/api/')){
      res.setHeader('Content-Type','application/json');
      res.end(JSON.stringify(req.url==='/api/devices'?{devices:[]}:{}));return;
    }
    const name=req.url.split('?')[0]==='/'?'index.html':req.url.split('?')[0].slice(1);
    if(path.basename(name)!==name){res.writeHead(404);res.end();return;}
    try{
      const types={'.html':'text/html','.css':'text/css','.js':'text/javascript','.svg':'image/svg+xml'};
      res.setHeader('Content-Type',types[path.extname(name)]||'application/octet-stream');
      res.end(await readFile(path.join(assets,name)));
    }catch(_){res.writeHead(404);res.end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    page.on('console',message=>{if(['error','warning'].includes(message.type()))errors.push(message.text());});
    const url=`http://127.0.0.1:${server.address().port}/#human`;
    await page.goto(url);
    assert.equal(page.url(),url);
    assert.match(await page.title(),/MultiCamera/);
    await page.locator('.live-trail').waitFor();
    assert.equal(await page.locator('#human-show-trails').isEnabled(),true);
    assert.equal(await page.locator('.live-trail').getAttribute('data-trail-uuid'),'person-a');
    const pointCount=()=>page.locator('.live-trail').evaluate(el=>el.points.numberOfItems);
    assert.equal(await pointCount(),4);
    assert.equal(await page.locator('#human-show-fov').isEnabled(),true);
    assert.equal(await page.locator('.live-fov').count(),1);
    await page.locator('#human-show-fov').uncheck();
    assert.equal(await page.locator('.live-fov').count(),0);
    assert.equal(await page.locator('.live-trail').count(),1);
    await page.locator('#human-show-fov').check();
    assert.equal(await page.locator('.live-fov').count(),1);
    await page.locator('#human-show-trails').uncheck();
    assert.equal(await page.locator('.live-trail').count(),0);
    assert.equal(await page.locator('#human-scene .person-marker').count(),1);
    await page.locator('#human-show-trails').check();
    await page.locator('#human-settings-open').click();
    assert.equal(await page.locator('#human-trail-length').isEnabled(),true);
    await page.locator('#human-trail-length').selectOption('short');
    assert.equal(await pointCount(),2);
    await page.locator('#human-settings button[value="done"]').click();
    const iso=await page.locator('.live-trail').getAttribute('points');
    await page.locator('#human-top-view').click();
    assert.notEqual(await page.locator('.live-trail').getAttribute('points'),iso);
    await page.locator('#human-reset-view').click();
    assert.equal(await page.locator('.live-trail').getAttribute('points'),iso);
    await page.locator('#human-settings-open').click();
    await page.locator('#human-trail-length').selectOption('long');
    await page.locator('#human-settings button[value="done"]').click();
    assert.equal(await pointCount(),4);
    await page.screenshot({path:path.join(artifactDirectory,'multicamera-trails-desktop.png'),fullPage:true});
    await page.locator('.scene-panel').screenshot({path:path.join(artifactDirectory,'multicamera-real-time-trails.png')});
    data={state:'paused',session_id:'test-session',cameras:[],tracks:[]};
    await page.waitForFunction(()=>!document.querySelector('#human-scene .person-marker'));
    assert.equal(await page.locator('.live-trail').count(),0);
    data=snapshot();data.session_id='new-session';data.tracks[0].track_uuid='person-b';
    data.tracks[0].sources[0].epoch='reconnected';data.tracks[0].trajectory=[{age_s:0,world_xy_m:[2,1]}];
    await page.waitForFunction(()=>document.querySelector('[data-track-uuid="person-b"]'));
    assert.equal(await page.locator('.live-trail').count(),0);
    data.tracks[0].trajectory.push({age_s:0,world_xy_m:[2.1,1]});
    await page.locator('[data-trail-uuid="person-b"]').waitFor();
    await page.locator('#human-demo-mode').click();
    assert.equal(await page.locator('.live-trail').count(),0);
    await page.locator('#human-status-mode').click();
    await page.locator('[data-trail-uuid="person-b"]').waitFor();
    assert.equal(await page.locator('[data-trail-uuid="person-a"]').count(),0);
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(artifactDirectory,'multicamera-trails-mobile.png'),fullPage:true});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    data.tracks=[];
    await page.waitForFunction(()=>!document.querySelector('#human-scene .person-marker'));
    assert.equal(await page.locator('.live-fov').count(),1);
    data.cross_camera.camera_views[0].world_frame='floor:wrong';
    await page.waitForFunction(()=>!document.querySelector('.live-fov'));
    assert.equal(await page.locator('#human-show-fov').isEnabled(),true);
    data={state:'unavailable',cameras:[],tracks:[]};
    await page.waitForFunction(()=>!document.querySelector('.live-trail'));
    assert.deepEqual(errors,[]);
  }finally{
    if(browser)await browser.close();
    server.closeAllConnections();
    await new Promise(resolve=>server.close(resolve));
  }
});

// Delays below are injected into this browser's fixture, never the live service.
test('cached scene survives failed and pending requests without replacing SVG or current nodes',
 {skip:!chromium&&'Playwright is not installed'},async()=>{
 let data=snapshot(),mode='healthy',sample=0;const pending=[];
 data.options_revision='rev';data.control_csrf='test';data.options={enabled:true,tracker:'bytetrack',camera_fps:3,imgsz:640,new_track_conf:.3,draw_boxes:true,draw_skeleton:true,jpeg_quality:78};
 data.cameras=[{id:'device_1',epoch:'one',state:'running',age_s:0,image_size:[640,480]}];
 const assets=path.resolve(__dirname,'../jetson/web');
 const server=createServer(async(req,res)=>{
  if(req.url==='/api/tracking/state'){
   if(mode==='hold')await new Promise(resolve=>pending.push(resolve));
   res.setHeader('Content-Type','application/json');
   if(mode==='unavailable'){res.end(JSON.stringify({state:'unavailable',cameras:[],tracks:[],cross_camera:{geometry_ready:false}}));return;}
   if(mode==='http-error'){res.writeHead(503);res.end('{}');return;}
   res.end(JSON.stringify({...data,updated_at_unix_ms:Date.now(),cameras:data.cameras.map(c=>({...c,sample_id:++sample}))}));return;
  }
  if(req.url.startsWith('/api/')){res.setHeader('Content-Type','application/json');res.end(JSON.stringify(req.url==='/api/devices'?{devices:[]}:{}));return;}
  const name=req.url.split('?')[0]==='/'?'index.html':req.url.split('?')[0].slice(1);
  try{
   assert.equal(path.basename(name),name);
   res.setHeader('Content-Type',({'.html':'text/html','.css':'text/css','.js':'text/javascript','.svg':'image/svg+xml'})[path.extname(name)]||'application/octet-stream');
   res.end(await readFile(path.join(assets,name)));
  }catch(_){res.writeHead(404);res.end();}
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
 try{
  browser=await chromium.launch({channel:'chrome',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/#human`);
  await page.waitForFunction(()=>document.querySelector('#human-metric-people').textContent==='1');
  await page.evaluate(()=>{window.originalScene=document.querySelector('#human-scene svg');window.originalFloor=originalScene.querySelector('polygon');window.originalFov=document.querySelector('.live-fov');window.originalHead=document.querySelector('#human-scene .person-marker');originalHead.focus();});
  mode='hold';
  await page.waitForFunction(()=>document.querySelector('#human-metric-people').textContent==='—');
  assert.equal(await page.locator('.live-fov').count(),1);
  assert.equal(await page.locator('#human-run-toggle').isDisabled(),true);
  assert.equal(await page.locator('#human-metric-cameras').innerText(),'— / 3');
  assert.match(await page.locator('#human-track-rows').innerText(),/待确认/);
  assert.equal(await page.locator('#human-scene .person-marker').getAttribute('data-live'),'false');
  assert.equal(await page.evaluate(()=>originalScene===document.querySelector('#human-scene svg')&&originalFloor===originalScene.querySelector('polygon')&&originalFov===document.querySelector('.live-fov')&&originalHead===document.querySelector('#human-scene .person-marker')&&document.activeElement===originalHead),true);
  mode='healthy';pending.splice(0).forEach(resolve=>resolve());
  await page.waitForFunction(()=>document.querySelector('#human-metric-people').textContent==='1');
  assert.equal(await page.locator('#human-run-toggle').isEnabled(),true);
  assert.equal(await page.evaluate(()=>originalHead===document.querySelector('#human-scene .person-marker')),true);
  for(const failure of ['unavailable','http-error']){
   mode=failure;
   await page.waitForFunction(()=>document.querySelector('.live-scene-status').textContent.includes('连接中断'));
   assert.equal(await page.locator('.live-fov').count(),1);
   assert.equal(await page.locator('#human-metric-people').innerText(),'—');
   assert.equal(await page.evaluate(()=>originalScene===document.querySelector('#human-scene svg')&&originalFov===document.querySelector('.live-fov')),true);
   mode='healthy';await page.waitForFunction(()=>document.querySelector('#human-metric-people').textContent==='1');
  }
  mode='unavailable';
  await page.waitForFunction(()=>document.querySelector('.live-scene-status').textContent.includes('连接中断'));
  await page.waitForFunction(()=>!document.querySelector('#human-scene .person-marker'),undefined,{timeout:5000});
  assert.equal(await page.locator('.live-fov').count(),1);assert.equal(await page.locator('.live-trail').count(),1);
  await page.locator('.scene-panel').screenshot({path:path.join(artifactDirectory,'multicamera-cached-scene.png')});
  mode='healthy';await page.waitForFunction(()=>document.querySelector('#human-metric-people').textContent==='1');
  data.tracks=[];
  await page.waitForFunction(()=>document.querySelector('#human-metric-people').textContent==='0');
  assert.equal(await page.locator('#human-scene .person-marker').count(),0);assert.equal(await page.locator('.live-trail').count(),1);
  assert.equal(await page.evaluate(()=>originalFov===document.querySelector('.live-fov')),true);
  const next=snapshot();data.tracks=next.tracks;data.session_id='session-b';data.tracks[0].track_uuid='person-b';data.tracks[0].trajectory=[{age_s:0,world_xy_m:[2,1]}];
  await page.waitForFunction(()=>document.querySelector('#human-scene [data-track-uuid="person-b"]'));
  assert.equal(await page.locator('[data-trail-uuid="person-a"]').count(),0);
  assert.equal(await page.evaluate(()=>originalFov===document.querySelector('.live-fov')),true);
  data.cross_camera.geometry_revision='geometry-new';
  data.cross_camera.camera_views[0].polygon_xy_m=[[10,10],[13,10],[13,13],[10,13]];
  data.tracks=[];
  await page.waitForFunction(()=>!document.querySelector('#human-scene .person-marker'));
  assert.equal(await page.locator('.live-trail').count(),0);
  data.cross_camera.geometry_ready=false;
  await page.waitForFunction(()=>!document.querySelector('.live-fov'));
  assert.equal(await page.evaluate(()=>originalScene===document.querySelector('#human-scene svg')),true);
  assert.deepEqual(errors,[]);
 }finally{
  pending.splice(0).forEach(resolve=>resolve());if(browser)await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
 }
});
