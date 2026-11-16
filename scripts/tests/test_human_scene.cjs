const {test}=require('node:test');
const assert=require('node:assert/strict');
const {HumanSceneStore}=require('../jetson/web/human-scene.js');
function data(sample=1){
 return {state:'running',session_id:'session-a',updated_at_unix_ms:sample,
  cameras:[{id:'device_1',epoch:'epoch-a',sample_id:sample}],
  cross_camera:{geometry_ready:true,geometry_revision:'geometry-a',world_frame:'floor:a',
   camera_views:[{camera_id:'device_1',world_frame:'floor:a',camera_xyz_m:[0,0,2],polygon_xy_m:[[0,0],[4,0],[0,4]]}]},
  tracks:[{track_uuid:'track-a',observed:true,world_frame:'floor:a',world_xy_m:[1,2],
   sources:[{camera_id:'device_1',epoch:'epoch-a',local_id:1,age_s:0}],
   trajectory:[{age_s:7,world_xy_m:[0,0]},{age_s:2,world_xy_m:[.5,1]},{age_s:0,world_xy_m:[1,2]}]}]};
}
test('transport failure preserves geometry, ghosts age out, histories expire without new data',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);const geometry=store.geometry;
 assert.equal(store.snapshot(10).heads[0].live,true);
 store.fail();let scene=store.snapshot(10.2);
 assert.equal(scene.geometry,geometry);assert.equal(scene.heads[0].live,false);
 assert.match(scene.status,/连接中断/);assert.equal(scene.trails.length,1);
 assert.equal(store.snapshot(13.1).heads.length,0);
 assert.equal(store.snapshot(18.1).trails.length,0);assert.equal(store.tracks.size,0);
 assert.equal(store.snapshot(100).geometry,geometry);
});
test('pending request exceeds freshness limit before HTTP timeout',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);
 assert.equal(store.snapshot(10.7).heads[0].live,true);
 assert.equal(store.snapshot(10.81).heads[0].live,false);
 assert.equal(store.fresh(10.81),false);assert.match(store.snapshot(11).status,/更新延迟/);
});
test('authoritative empty frames remove current people immediately and retain bounded history',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);
 const next=data(2);next.tracks=[];store.accept(next,10.1);
 const scene=store.snapshot(10.1);assert.equal(scene.heads.length,0);assert.equal(scene.trails.length,1);
 assert(scene.geometry);assert.equal(scene.fresh,true);assert.equal(store.snapshot(18.1).trails.length,0);
});
test('gateway unavailable is not evidence that calibration or people disappeared',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);const geometry=store.geometry;
 store.accept({state:'unavailable',cameras:[],tracks:[],cross_camera:{geometry_ready:false}},10.1);
 assert.equal(store.geometry,geometry);assert.equal(store.snapshot(10.1).heads.length,1);
 assert.equal(store.snapshot(10.1).heads[0].live,false);
 store.accept(data(2),10.3);assert.equal(store.snapshot(10.3).heads[0].live,true);
});
test('duplicate samples neither refresh a person nor append or rejuvenate trajectory points',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);
 const points=store.tracks.get('track-a').points;
 store.accept(data(),10.7);store.accept(data(),11);
 assert.equal(store.tracks.get('track-a').points,points);assert.equal(points.length,3);
 assert.equal(store.snapshot(11).heads[0].live,false);
 assert.equal(store.snapshot(11.1).trails[0].points.length,2);
});
test('source age and short/long trail settings use the original sample times',()=>{
 const store=new HumanSceneStore();const old=data();old.tracks[0].sources[0].age_s=1;
 store.accept(old,10);assert.equal(store.snapshot(10).heads[0].live,false);
 assert.equal(store.snapshot(10,3).trails[0].points.length,2);
 assert.equal(store.snapshot(10,8).trails[0].points.length,3);
});
for(const change of ['session','geometry','epoch','membership'])test(change+' changes never bridge old history',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);const next=data(2);
 if(change==='session')next.session_id='session-b';
 if(change==='geometry')next.cross_camera.geometry_revision='geometry-b';
 if(change==='epoch'){next.cameras[0].epoch='epoch-b';next.tracks[0].sources[0].epoch='epoch-b';}
 if(change==='membership')next.tracks[0].sources[0].local_id=2;
 next.tracks[0].trajectory=[{age_s:0,world_xy_m:[5,5]}];next.tracks[0].world_xy_m=[5,5];
 store.accept(next,10.1);assert.equal(store.tracks.size,1);
 assert.equal(store.tracks.get('track-a').points.length,1);assert.equal(store.snapshot(10.1).trails.length,0);
});
test('explicit calibration invalidation clears projections; pause clears dynamic history only',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);const paused=data(2);paused.state='paused';
 store.accept(paused,10.1);assert(store.geometry);assert.equal(store.tracks.size,0);
 const invalid=data(3);invalid.cross_camera.geometry_ready=false;
 store.accept(invalid,10.2);assert.equal(store.geometry,null);assert.equal(store.tracks.size,0);
});
test('an old camera epoch cannot leave a historical trail when its replacement has no people',()=>{
 const store=new HumanSceneStore();store.accept(data(),10);const next=data(2);
 next.cameras[0].epoch='new';next.tracks=[];store.accept(next,10.1);
 assert.equal(store.snapshot(10.1).heads.length,0);assert.equal(store.snapshot(10.1).trails.length,0);
});
test('display storage has a bounded number of tracks and points',()=>{
 const store=new HumanSceneStore();const snapshot=data();
 snapshot.tracks=Array.from({length:600},(_,i)=>({...snapshot.tracks[0],track_uuid:'id-'+i,
  trajectory:Array.from({length:300},(_,j)=>({age_s:j/100,world_xy_m:[j,j]}))}));
 store.accept(snapshot,10);assert.equal(store.tracks.size,512);
 for(const entry of store.tracks.values())assert.equal(entry.points.length,256);
});
