'use strict';

// Display-only memory. Never supplies observations to tracking, controls or MQTT.
(function(global){
const finiteXY=p=>Array.isArray(p)&&p.length>=2&&p.every(Number.isFinite);
const sourcesOf=t=>(t.sources||[]).map(s=>[s.camera_id,s.epoch,s.local_id]).sort();
class HumanSceneStore{
 constructor(){this.geometry=null;this.geometryKey=null;this.session=null;this.epochs=new Map();this.tracks=new Map();this.lastAccepted=null;this.transport='connecting';}
 clearTracks(){this.tracks.clear();}
 fail(){this.transport='unavailable';}
 fresh(now){return this.lastAccepted!==null&&this.transport!=='unavailable'&&now-this.lastAccepted<=.8;}
 accept(data,now){
  if(!data||!['running','paused','loading','failed'].includes(data.state)){this.fail();return false;}
  this.transport=data.state;this.lastAccepted=now;
  if(data.session_id&&this.session!==data.session_id){this.clearTracks();this.epochs.clear();this.session=data.session_id;}
  const cross=data.cross_camera;
  if(cross&&Object.prototype.hasOwnProperty.call(cross,'geometry_ready')){
   if(!cross.geometry_ready){this.geometry=null;this.geometryKey=null;this.clearTracks();}
   else if(cross.world_frame){
    const views=(cross.camera_views||[]).filter(v=>v.world_frame===cross.world_frame&&finiteXY(v.camera_xyz_m)&&Array.isArray(v.polygon_xy_m)&&v.polygon_xy_m.length>=3&&v.polygon_xy_m.every(finiteXY));
    const key=JSON.stringify([cross.geometry_revision||null,cross.world_frame,views]);
    if(key!==this.geometryKey){this.clearTracks();this.geometryKey=key;}
    this.geometry={world_frame:cross.world_frame,views,linked:cross.state==='ready'};
   }
  }
  for(const c of data.cameras||[]){
   if(c.epoch===undefined)continue;
   if(this.epochs.has(c.id)&&this.epochs.get(c.id)!==c.epoch){
    for(const [id,entry] of this.tracks)if((entry.track.sources||[]).some(s=>s.camera_id===c.id))this.tracks.delete(id);
   }
   this.epochs.set(c.id,c.epoch);
  }
  if(data.state==='paused'){this.clearTracks();return true;}
  if(data.state!=='running')return true;
  const present=new Set();
  for(const track of data.tracks||[]){
   if(!track.observed||!track.track_uuid||!finiteXY(track.world_xy_m))continue;
   if(!this.geometry||(track.world_frame&&track.world_frame!==this.geometry.world_frame))continue;
   const id=track.track_uuid,signature=JSON.stringify(sourcesOf(track));
   present.add(id);
   let entry=this.tracks.get(id);
   if(entry&&entry.signature!==signature){this.tracks.delete(id);entry=null;}
   const ages=(track.sources||[]).map(s=>s.age_s).filter(v=>Number.isFinite(v)&&v>=0);
   const age=ages.length?Math.min(...ages):0;
   const samples=(track.sources||[]).map(s=>{
    const camera=(data.cameras||[]).find(c=>c.id===s.camera_id);
    return camera?.sample_id===undefined?null:[s.camera_id,s.epoch,s.local_id,camera.sample_id];
   });
   const sample=samples.length&&samples.every(s=>s!==null)?JSON.stringify(samples):
    JSON.stringify([data.updated_at_unix_ms??null,track.world_xy_m,track.trajectory||[]]);
   const seen=now-age;
   if(!entry||entry.sample!==sample){
    const points=(track.trajectory||[]).filter(p=>finiteXY(p.world_xy_m)&&Number.isFinite(p.age_s)&&p.age_s>=0&&p.age_s<=8)
     .map(p=>({at:now-p.age_s,xy:p.world_xy_m.slice(0,2)})).sort((a,b)=>a.at-b.at).slice(-256);
    if(!points.length)points.push({at:seen,xy:track.world_xy_m.slice(0,2)});
    entry={track,signature,sample,seen,points,present:true};this.tracks.set(id,entry);
   }else{entry.seen=Math.min(entry.seen,seen);entry.track=track;entry.present=true;}
  }
  for(const [id,entry] of this.tracks)if(!present.has(id))entry.present=false;
  while(this.tracks.size>512)this.tracks.delete(this.tracks.keys().next().value);
  return true;
 }
 snapshot(now,duration=8){
  const fresh=this.fresh(now),heads=[],trails=[];
  for(const [id,entry] of this.tracks){
   entry.points=entry.points.filter(p=>now-p.at<=8);
   const age=Math.max(0,now-entry.seen),live=fresh&&this.transport==='running'&&entry.present&&age<=.8;
   if(entry.present&&age<=3)heads.push({id,track:entry.track,xy:entry.track.world_xy_m,age,live});
   const points=entry.points.filter(p=>now-p.at<=duration);
   if(points.length>1)trails.push({id,points,live});
   if(age>8&&!entry.points.length)this.tracks.delete(id);
  }
  const age=this.lastAccepted===null?null:Math.max(0,now-this.lastAccepted);
  let status=fresh?'实时更新':'更新延迟 · 保留上次场景';
  if(this.transport==='paused')status='追踪已暂停 · 保留已核验场景';
  else if(this.transport==='failed')status='追踪异常 · 保留上次场景';
  else if(this.transport==='unavailable')status='连接中断 · 保留上次场景';
  if(age!==null)status+=' · '+age.toFixed(1)+' 秒前更新';
  else status=(this.transport==='unavailable'?'连接中断 · ':'')+'等待首次有效场景';
  return {geometry:this.geometry,geometryKey:this.geometryKey,heads,trails,fresh,status};
 }
}

const ns='http://www.w3.org/2000/svg';
function svgNode(tag,attrs={}){const node=document.createElementNS(ns,tag);setAttrs(node,attrs);return node;}
function setAttrs(node,attrs){for(const [key,value] of Object.entries(attrs))if(node.getAttribute(key)!==String(value))node.setAttribute(key,String(value));}
function setText(node,text){if(node.textContent!==text)node.textContent=text;}
function reconcile(parent,map,items,key,create,update){
 const wanted=new Set();
 for(const item of items){const id=key(item);wanted.add(id);let node=map.get(id);if(!node){node=create(item);map.set(id,node);parent.append(node);}update(node,item);}
 for(const [id,node] of map)if(!wanted.has(id)){node.remove();map.delete(id);}
}
class HumanSceneView{
 constructor(container){this.container=container;this.svg=null;this.fit=null;this.fitKey=null;}
 resetFit(){this.fit=null;}
 ensure(){
  if(this.svg?.parentNode===this.container)return;
  this.container.classList.add('live-scene');
  this.svg=svgNode('svg',{viewBox:'0 0 640 420','aria-label':'地面位置与最近有效轨迹'});
  this.floor=svgNode('polygon',{fill:'#1a2c3b',stroke:'#4b647e'});
  this.title=svgNode('text',{x:20,y:48,fill:'#9dacbe','font-size':11});
  this.fovLayer=svgNode('g');this.trailLayer=svgNode('g');this.headLayer=svgNode('g');
  this.fovNodes=new Map();this.trailNodes=new Map();this.headNodes=new Map();
  this.note=svgNode('text',{x:20,y:405,fill:'#9dacbe','font-size':10});
  this.svg.append(this.floor,this.fovLayer,this.trailLayer,this.headLayer,this.title,this.note);
  this.status=document.createElement('p');this.status.className='live-scene-status';this.status.setAttribute('role','status');this.status.setAttribute('aria-live','off');
  this.empty=document.createElement('div');this.empty.className='scene-empty live-scene-empty';
  const title=document.createElement('strong'),note=document.createElement('p');
  title.textContent='等待有效地面场景';note.textContent='接入已核验的地面标定后显示相机视野及人体位置。';this.empty.append(title,note);
  this.container.replaceChildren(this.svg,this.status,this.empty);
 }
 update(model,options){
  this.ensure();
  if(this.fitKey!==model.geometryKey){this.fitKey=model.geometryKey;this.fit=null;}
  const geometry=model.geometry;
  const bounds=this.fit?[]:[...(geometry?.views||[]).flatMap(v=>[v.camera_xyz_m,...v.polygon_xy_m]),...model.heads.map(h=>h.xy),...model.trails.flatMap(t=>t.points.map(p=>p.xy))];
  if(!this.fit&&bounds.length){
   const xs=bounds.map(p=>p[0]),ys=bounds.map(p=>p[1]);
   this.fit={cx:(Math.min(...xs)+Math.max(...xs))/2,cy:(Math.min(...ys)+Math.max(...ys))/2,span:Math.max(6,Math.max(...xs)-Math.min(...xs)+2,Math.max(...ys)-Math.min(...ys)+2)};
  }
  const fit=this.fit||{cx:0,cy:0,span:6};
  const screen=p=>{const x=4+(p[0]-fit.cx)*6/fit.span,y=3+(p[1]-fit.cy)*6/fit.span;return options.top?[74+x*62,49+(6-y)*55]:[320+(x-4)*40-(y-3)*34,205+(x-4)*20+(y-3)*21];};
  const plane=[[1,0],[7,0],[7,6],[1,6]].map(([x,y])=>options.top?[74+x*62,49+(6-y)*55]:[320+(x-4)*40-(y-3)*34,205+(x-4)*20+(y-3)*21]);
  setAttrs(this.floor,{points:plane.map(p=>p.join(',')).join(' ')});
  setText(this.title,geometry?geometry.world_frame+' · 单位 m'+(geometry.linked?'':' · 未跨机去重'):'地面标定待就绪');
  setText(this.status,model.status);this.status.classList.toggle('is-stale',!model.fresh);
  this.empty.hidden=!!geometry||model.heads.length>0||model.trails.length>0;
  setText(this.empty.children[0],model.fresh?'等待有效地面标定':'等待有效场景数据');
  setText(this.empty.children[1],model.fresh?'接入已核验的地面标定后显示相机视野及人体位置。':'连接恢复后自动加载场景；暂不判断标定或人员状态。');
  const views=options.fov?(geometry?.views||[]):[];
  setText(this.note,views.length?'标定视野地面投影 · 世界X/Y ±20 m内 · 未计遮挡':'');
  reconcile(this.fovLayer,this.fovNodes,views,v=>v.camera_id,()=>{const g=svgNode('g',{'class':'live-fov'});g.append(svgNode('polygon'),svgNode('rect'),svgNode('text'));return g;},(g,v)=>{
   const index=(geometry?.views||[]).indexOf(v),color=['#73abff','#f4c276','#c99bff'][index%3],[x,y]=screen(v.camera_xyz_m);
   setAttrs(g,{'data-camera-fov':v.camera_id});
   setAttrs(g.children[0],{points:v.polygon_xy_m.map(p=>screen(p).join(',')).join(' '),fill:color,'fill-opacity':.10,stroke:color,'stroke-dasharray':'5 4'});
   setAttrs(g.children[1],{x:x-5,y:y-4,width:10,height:8,fill:color});
   setAttrs(g.children[2],{x:x+9,y:y-8-index*15,fill:color,'font-size':11});setText(g.children[2],v.camera_id.replace('device_','C'));
  });
  reconcile(this.trailLayer,this.trailNodes,options.trails?model.trails:[],t=>t.id,()=>svgNode('polyline',{'class':'live-trail',fill:'none','stroke-width':2}),(node,t)=>{
   setAttrs(node,{'data-trail-uuid':t.id,points:t.points.map(p=>screen(p.xy).join(',')).join(' '),stroke:t.live?'#55d7c2':'#8b9aab',opacity:t.live ? .65 : .35});
  });
  reconcile(this.headLayer,this.headNodes,model.heads,h=>h.id,()=>{const g=svgNode('g',{role:'button',tabindex:0});g.append(svgNode('circle',{r:9}),svgNode('text',{'font-size':12}));return g;},(g,h)=>{
   const [x,y]=screen(h.xy),name=options.name(h.track),label=h.live?name:name+'（历史 · '+h.age.toFixed(1)+'s前）';
   setAttrs(g,{'class':'person-marker'+(h.live?'':' is-stale'),'data-track-uuid':h.id,'aria-label':'选择 '+label,'data-live':h.live,opacity:h.live?1:.55});
   setAttrs(g.children[0],{cx:x,cy:y,fill:h.live?'#55d7c2':'#8b9aab'});
   setAttrs(g.children[1],{x:x+14,y:y+4,fill:'#edf3fa'});setText(g.children[1],label);
  });
 }
}
global.HumanSceneStore=HumanSceneStore;global.HumanSceneView=HumanSceneView;
if(typeof module!=='undefined')module.exports={HumanSceneStore};
})(typeof globalThis==='undefined'?window:globalThis);
