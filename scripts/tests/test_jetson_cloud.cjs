// Run with: node --test scripts/tests/test_jetson_cloud.cjs
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const source=readFileSync(require('node:path').join(__dirname,'../jetson/web/cloud.js'),'utf8');
function viewer(){
  const events={}, pixels=[];
  const ctx=new Proxy({fillRect:(...args)=>pixels.push([...args,ctx.fillStyle])}, {get:(o,k)=>o[k]||(()=>{})});
  const canvas={dataset:{},setAttribute(){},addEventListener:(name,f)=>events[name]=f,
    getBoundingClientRect:()=>({width:800,height:500}),getContext:()=>ctx,
    focus(){},setPointerCapture(){}};
  const sandbox={ResizeObserver:class{observe(){}},devicePixelRatio:1,requestAnimationFrame:f=>f()};
  vm.createContext(sandbox);vm.runInContext(source+'\nglobalThis.View=CloudView;',sandbox);
  return {view:new sandbox.View(canvas),canvas,events,pixels};
}
test('fit contains an offset metric scene in every preset without anchoring to origin',()=>{
  const {view,canvas}=viewer();view.worldAxes=true;
  const points=[];
  for(const x of [9,11])for(const y of [18,22])for(const z of [-3,3])points.push([x,y,z]);
  const original=JSON.stringify(points);view.update(points,[]);
  assert.deepEqual(Array.from(view.center),[10,20,0]);
  for(const preset of ['front','top','side','iso']){
    view.setView(preset);assert.equal(Number(canvas.dataset.visibleCount),8,preset);
  }
  assert.equal(JSON.stringify(points),original);
});
test('refresh preserves user navigation; fit and reset recover a lost scene',()=>{
  const {view,canvas}=viewer();view.update([[0,0,1],[1,1,2]],[]);
  view.yaw=.9;view.zoom=4;view.pan=[5000,0];view.draw();
  assert.equal(Number(canvas.dataset.visibleCount),0);
  view.update([[0,0,1],[1,1,2]],[]);
  assert.equal(view.zoom,4);assert.equal(view.yaw,.9);
  view.fit();assert.equal(Number(canvas.dataset.visibleCount),2);assert.equal(view.yaw,.9);
  view.reset();assert.equal(view.yaw,0);assert.equal(view.zoom,1);
});
test('invalid samples and empty results clear counts safely',()=>{
  const {view,canvas}=viewer();view.update([[NaN,0,1],[0,0,1],[Infinity,0,0]],[]);
  assert.equal(canvas.dataset.pointCount,'1');view.update([],[]);
  assert.equal(canvas.dataset.pointCount,'0');assert.equal(canvas.dataset.visibleCount,'0');
});
test('right drag pans, pointer cancellation stops moving, zoom stays bounded',()=>{
  const {view,events}=viewer();view.update([[0,0,1]],[]);
  events.pointerdown({button:2,pointerId:1,clientX:100,clientY:100});
  events.pointermove({pointerId:1,clientX:140,clientY:120,buttons:2});
  assert.deepEqual(Array.from(view.pan),[40,20]);assert.equal(view.yaw,0);
  events.pointercancel({pointerId:1});
  events.pointermove({pointerId:1,clientX:500,clientY:120,buttons:2});
  assert.deepEqual(Array.from(view.pan),[40,20]);
  view.scale(1e6);assert.equal(view.zoom,12);view.scale(1e-6);assert.equal(view.zoom,.15);
});
test('two finger gesture zooms and pans without rotating; orbit can pass both poles',()=>{
  const {view,events}=viewer();view.update([[0,0,1]],[]);
  events.pointerdown({button:0,pointerId:1,clientX:100,clientY:100});
  events.pointerdown({button:0,pointerId:2,clientX:200,clientY:100});
  events.pointermove({pointerId:2,clientX:250,clientY:100,buttons:1});
  assert.equal(view.zoom,1.5);assert.equal(view.pan[0],25);assert.equal(view.yaw,0);
  events.pointerup({pointerId:2});
  events.pointermove({pointerId:1,clientX:100,clientY:9000,buttons:1});
  assert.ok(view.pitch>Math.PI*2);
});

test('world top view looks down positive Z with raised surfaces in front',()=>{
  const {view,pixels}=viewer();view.worldAxes=true;
  view.update([[0,0,0],[0,0,.4]],[[0,0,255],[255,0,0]]);
  pixels.length=0;view.setView('top');
  const points=pixels.filter(p=>p[4]==='rgb(0,0,255)'||p[4]==='rgb(255,0,0)');
  assert.equal(points.length,2);
  assert.equal(points[1][4],'rgb(255,0,0)');
});
test('device layers filter real source labels without moving the common reference frame',()=>{
  const {view,canvas,pixels}=viewer();
  view.deviceColors={device_1:[1,2,3],device_2:[4,5,6]};view.colorMode='device';
  view.update([[0,0,1],[1,1,2]],[[255,255,255],[255,255,255]],['device_2','device_1']);
  const center=JSON.stringify(view.center),radius=view.radius;
  view.visibleDevices=new Set(['device_2']);pixels.length=0;view.draw();
  assert.equal(canvas.dataset.enabledCount,'1');assert.equal(canvas.dataset.visibleCount,'1');
  assert.ok(pixels.some(p=>p[4]==='rgb(4,5,6)'));assert.ok(!pixels.some(p=>p[4]==='rgb(1,2,3)'));
  assert.equal(JSON.stringify(view.center),center);assert.equal(view.radius,radius);
  view.visibleDevices.clear();view.draw();assert.equal(canvas.dataset.visibleCount,'0');
  view.visibleDevices.add('device_1');view.draw();assert.equal(canvas.dataset.visibleCount,'1');
});
test('filtering invalid coordinates preserves point/color/source correspondence',()=>{
  const {view,canvas,pixels}=viewer();view.deviceColors={device_2:[4,5,6]};view.colorMode='device';
  view.update([[NaN,0,1],[0,0,1]],[[255,0,0],[0,255,0]],['device_1','device_2']);
  view.visibleDevices=new Set(['device_2']);pixels.length=0;view.draw();
  assert.equal(canvas.dataset.enabledCount,'1');assert.ok(pixels.some(p=>p[4]==='rgb(4,5,6)'));
});

test('roll changes screen orientation without modifying metric coordinates, and level restores Z up',()=>{
  const {view,canvas,pixels,events}=viewer();view.worldAxes=true;
  const points=[[0,0,0],[0,0,.4]], original=JSON.stringify(points);
  view.update(points,[[0,0,255],[255,0,0]]);view.setView('front');
  view.mode='roll';
  events.pointerdown({button:0,pointerId:1,clientX:100,clientY:100});
  events.pointermove({pointerId:1,clientX:200,clientY:100,buttons:1});
  assert.equal(view.roll,.8);assert.equal(view.pitch,0);assert.equal(view.yaw,0);
  pixels.length=0;view.draw();
  const dots=pixels.filter(p=>p[4]==='rgb(0,0,255)'||p[4]==='rgb(255,0,0)');
  assert.notEqual(dots[0][0],dots[1][0]);
  view.level();assert.equal(view.roll,0);
  view.pitch=Math.PI;view.roll=.7;view.level();assert.equal(view.roll,Math.PI);
  pixels.length=0;view.draw();
  const level=pixels.filter(p=>p[4]==='rgb(0,0,255)'||p[4]==='rgb(255,0,0)');
  const raised=level.find(p=>p[4]==='rgb(255,0,0)'),base=level.find(p=>p[4]==='rgb(0,0,255)');
  assert.ok(raised[1]<base[1]);
  view.setView('top');assert.equal(view.roll,0);view.level();assert.equal(view.pitch,0);
  assert.equal(JSON.stringify(points),original);
  assert.equal(JSON.stringify(view.points),original);assert.equal(canvas.dataset.roll,'0');
});
test('middle drag pans; keyboard roll and full pitch work and reset clears navigation',()=>{
  const {view,events}=viewer();view.update([[0,0,1],[1,1,2]],[]);
  events.pointerdown({button:1,pointerId:1,clientX:100,clientY:100});
  events.pointermove({pointerId:1,clientX:170,clientY:130,buttons:4});
  assert.deepEqual(Array.from(view.pan),[70,30]);assert.equal(view.yaw,0);
  events.pointerup({pointerId:1});
  const key=(key,shiftKey=false)=>events.keydown({key,shiftKey,preventDefault(){}});
  key('e');assert.ok(view.roll>0);key('q');assert.equal(view.roll,0);
  for(let i=0;i<30;i++)key('ArrowDown');assert.ok(view.pitch>Math.PI);
  key('ArrowRight',true);assert.equal(view.pan[0],90);
  key('Home');assert.equal(view.roll,0);assert.equal(view.pitch,0);assert.equal(view.zoom,1);
  assert.deepEqual(Array.from(view.pan),[0,0]);
});
