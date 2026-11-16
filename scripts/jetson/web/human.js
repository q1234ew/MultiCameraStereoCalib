"use strict";

(() => {
const root = document.getElementById("human-workspace");
let active = document.body.dataset.previewMode === "human";
let liveData = null, liveSelected = null, liveBusy = false, liveController = null,liveRequestEpoch=0;
const people = [
  {id:"P-001", color:"#55d7c2", x:2.84,y:3.16, sources:[[1,12],[2,7]], score:96, status:"稳定关联", elapsed:"持续观察 18 秒", key:"linked"},
  {id:"P-002", color:"#73abff", x:5.32,y:2.28, sources:[[2,9],[3,4]], score:91, status:"稳定关联", elapsed:"持续观察 12 秒", key:"linked"},
  {id:"P-003", color:"#f4c276", x:6.55,y:4.48, sources:[[3,8]], score:null, status:"单路观察", elapsed:"持续观察 4 秒", key:"single"},
];
const state = {selected:0, demo:true, top:false, playing:true, time:18, speed:1, trails:true, fov:true, skeleton:true, boxes:true, trailLength:"long",filter:"all",focused:false};
const $ = (id) => document.getElementById("human-" + id);
let toastTimeout;
const cameraFrames = new Map();
let cameraDialog = null;
let enlargedCamera = null;
let previewMode = 'smooth';
let liveReceivedAt = 0;
const liveSceneStore = new HumanSceneStore();
const liveSceneView = new HumanSceneView($('scene'));
const sceneNow = () => performance.now()/1000;
function toast(message){$("toast").textContent=message;$("toast").classList.add("visible");clearTimeout(toastTimeout);toastTimeout=setTimeout(()=>$("toast").classList.remove("visible"),3500);}
function locationOf(p){const phase=state.time-18;return {x:p.x+Math.sin(phase/14)*.4,y:p.y+Math.sin(phase/20)*.22};}
function skeleton(x,y,h,color){return `<g class="skeleton" stroke="${color}" stroke-width="2" stroke-linecap="round" fill="none"><circle cx="${x}" cy="${y-h*.84}" r="${h*.065}"/><path d="M${x},${y-h*.76} v${h*.33} M${x-h*.17},${y-h*.65} l${h*.17},${-h*.07} ${h*.17},${h*.07} M${x-h*.17},${y-h*.65} l${-h*.04},${h*.2} M${x+h*.17},${y-h*.65} l${h*.04},${h*.2} M${x},${y-h*.43} l${-h*.1},${h*.22} ${-h*.04},${h*.2} M${x},${y-h*.43} l${h*.1},${h*.22} ${h*.04},${h*.2}"/></g>`;}
function roomSVG(camera){const offsets=[0,14,-10],offset=offsets[camera-1];return `<rect width="320" height="180" fill="#1e2a36"/><path d="M0 0H320V82L180 120 0 95Z" fill="#283744"/><path d="M0 95L180 120 320 82V180H0Z" fill="#303b44"/><path d="M${175+offset} 0V119M0 94L180 120 320 82M55 101L30 180M112 110L109 180M234 106L280 180" stroke="#425260" fill="none" stroke-width="1"/><rect x="27" y="28" width="65" height="49" rx="1" fill="#15232e" stroke="#4b5d6d"/><path d="M59 28V77M27 52H92" stroke="#3c4d60"/><path d="M240 75l41-12v52l-41 13z" fill="#13222f" stroke="#4b5c6d"/><path d="M104 120l50-12 29 13-49 14z" fill="#5c6367"/><path d="M104 120v17l30 16v-18M134 135l49-14v16l-49 16" fill="#3f4a53" stroke="#1c2b36"/><path d="M17 150h58v9H17zM22 159v15M70 159v15" fill="#4d5a65" stroke="#72818c"/><rect x="0" y="0" width="320" height="180" fill="#0b1523" opacity=".1"/>`;}
function cameraSVG(camera){const group=people.flatMap((p,index)=>p.sources.some(s=>s[0]===camera)?[{p,index}]:[]);return `<svg viewBox="0 0 320 180" aria-label="相机 ${camera} 模拟画面">${roomSVG(camera)}${group.map(({p,index},order)=>{const x=100+order*113+(camera===2?12:0),y=157-order*12,h=88-order*7,selected=index===state.selected;return `<g class="camera-person" data-person="${index}" role="button" tabindex="0" aria-label="选择 ${p.id}"><path d="M${x-12} ${y-h*.7}q12-7 24 0l7 34-14 5-3 32h-8l-4-32-12-5z" fill="${p.color}" opacity=".13"/><rect class="detection-box" x="${x-29}" y="${y-h-5}" width="58" height="${h+10}" rx="2" fill="none" stroke="${p.color}" stroke-width="${selected?2.4:1}" opacity="${selected?1:.7}"/><g class="detection-label"><rect x="${x-29}" y="${y-h-20}" width="75" height="15" rx="2" fill="${p.color}"/><text x="${x-25}" y="${y-h-9}" font-family="system-ui" font-size="8" font-weight="700" fill="#10222d">${p.id} / L${p.sources.find(s=>s[0]===camera)[1]}</text></g>${skeleton(x,y,h,p.color)}</g>`;}).join("")}<text x="8" y="171" fill="#b2c4d9" font-size="7" font-family="system-ui">SIMULATED IMAGE · ${formatTime(state.time)}</text></svg>`;}
function renderCameras(){if(!state.demo){renderLiveCameras();return;}$("camera-list").innerHTML=[1,2,3].map(n=>`<article class="camera-card"><div class="camera-head"><strong>相机 0${n}</strong><span>${state.demo?"演示画面 · 1280×720":"设备接入状态待查询"}</span></div><div class="camera-scene ${state.skeleton?"":"hide-skeleton"} ${state.boxes?"":"hide-boxes"}">${state.demo?cameraSVG(n):'<div class="empty-camera">尚未连接实时输入</div>'}</div><div class="camera-foot"><span>${state.demo?people.filter(p=>p.sources.some(s=>s[0]===n)).length+" 人 · 本机追踪":"追踪服务待接入"}</span><span>${state.demo?"模拟 15 FPS":"— FPS"}</span></div></article>`).join("");}
function project(x,y,z=0){return state.top?[74+x*62,49+(6-y)*55-z*22]:[320+(x-4)*40-(y-3)*34,205+(x-4)*20+(y-3)*21-z*44];}
function line(points,attrs=""){return `<polyline points="${points.map(p=>project(...p).join(",")).join(" ")}" fill="none" ${attrs}/>`;}
function polygon(points,attrs=""){return `<polygon points="${points.map(p=>project(...p).join(",")).join(" ")}" ${attrs}/>`;}
function renderScene(){if(!state.demo){renderLiveScene();return;}
 $("scene").classList.remove("live-scene");
 let room=polygon([[0,0],[8,0],[8,6],[0,6]],'fill="#1a2c3b" stroke="#4b647e" stroke-width="1.2"');
 for(let x=0;x<=8;x++)room+=line([[x,0],[x,6]],'stroke="#3a5066" stroke-width=".7" opacity=".6"');
 for(let y=0;y<=6;y++)room+=line([[0,y],[8,y]],'stroke="#3a5066" stroke-width=".7" opacity=".6"');
 if(!state.top)room+=polygon([[0,0],[8,0],[8,0,1.6],[0,0,1.6]],'fill="#426078" opacity=".12" stroke="#6f8ba4" stroke-width="1"')+polygon([[0,0],[0,6],[0,6,1.6],[0,0,1.6]],'fill="#426078" opacity=".08" stroke="#6f8ba4" stroke-width="1"');
 room+=polygon([[1,.7],[2.8,.7],[2.8,1.4],[1,1.4]],'fill="#425363" stroke="#667a8b" opacity=".75"');
 room+=polygon([[.4,4.8],[1.7,4.8],[1.7,5.4],[.4,5.4]],'fill="#425363" stroke="#667a8b" opacity=".75"');
 const fovs=`<g class="fov-layer" ${state.fov?"":"hidden"}>${polygon([[0,0],[5,1],[1,5]],'fill="#508bdc" opacity=".07" stroke="#7ba6d4" stroke-dasharray="5 5"')}${polygon([[8,0],[2,1],[7,5]],'fill="#508bdc" opacity=".07" stroke="#7ba6d4" stroke-dasharray="5 5"')}${polygon([[8,6],[3,5],[7,1]],'fill="#508bdc" opacity=".07" stroke="#7ba6d4" stroke-dasharray="5 5"')}</g>`;
 const cameras=[[0,0],[8,0],[8,6]].map((p,i)=>{const [x,y]=project(...p);return `<g transform="translate(${x} ${y})"><rect x="-10" y="-7" width="20" height="14" rx="3" fill="#304862" stroke="#8ea7c4"/><path d="M11-4l6-3v14l-6-3" fill="#7898bd"/><text x="0" y="-15" text-anchor="middle" fill="#b6cbe3" font-size="9">CAM 0${i+1}</text></g>`;}).join("");
 const markers=people.map((p,i)=>{const pos=locationOf(p),[x,y]=project(pos.x,pos.y),sel=i===state.selected;let trail="";const count=state.trailLength==="long"?8:3;for(let k=count;k>=0;k--){const [tx,ty]=project(pos.x-k*.14,pos.y+Math.sin(k*.4)*.26);trail+=`${tx},${ty} `;}return `<g class="trails-layer" ${state.trails?"":"hidden"}><polyline points="${trail}" fill="none" stroke="${p.color}" stroke-width="2" stroke-dasharray="3 4" opacity=".6"/></g><g class="person-marker ${sel?"is-selected":""}" data-person="${i}" role="button" tabindex="0" aria-label="选择 ${p.id}" transform="translate(${x},${y})"><ellipse class="marker-base" rx="${sel?18:13}" ry="${state.top?13:7}" fill="${p.color}" fill-opacity=".15" stroke="${p.color}" stroke-width="${sel?2:1}"/>${sel?`<ellipse rx="26" ry="${state.top?21:11}" fill="none" stroke="${p.color}" opacity=".3"/>`:""}${state.top?`<circle cy="-4" r="7" fill="${p.color}"/><path d="M-9 5Q0 14 9 5" fill="none" stroke="${p.color}" stroke-width="4"/>`:`<circle cy="-53" r="6" fill="${p.color}"/><path d="M0-43v22M-11-35L0-40 11-35M-11-35l-4 14M11-35l4 14M0-21L-8-3M0-21L8-3" fill="none" stroke="${p.color}" stroke-width="4" stroke-linecap="round"/>`}<rect x="-28" y="${state.top?-34:-82}" width="56" height="20" rx="4" fill="${sel?p.color:'#15273a'}" stroke="${p.color}"/><text x="0" y="${state.top?-20:-68}" text-anchor="middle" font-family="system-ui" font-size="10" font-weight="600" fill="${sel?'#0b2330':p.color}">${p.id}</text></g>`;}).join("");
 const [ox,oy]=project(0,6);let axis=`<g transform="translate(${ox-20} ${oy+40})"><path d="M0 0h32M0 0l-18-12M0 0v-28" stroke="#8aa5bf" stroke-width="1"/><text x="37" y="4" fill="#b5c8dc" font-size="9">X</text><text x="-28" y="-13" fill="#b5c8dc" font-size="9">Y</text><text x="-3" y="-34" fill="#b5c8dc" font-size="9">Z</text></g>`;
 $("scene").innerHTML=`<svg viewBox="0 0 640 420" role="img" aria-label="模拟区域三维场景，三名人员与三个相机位置">${room}${fovs}${cameras}${markers}${axis}<g fill="#7e98b3" font-size="9"><text x="22" y="27">区域 A / 演示场景</text><text x="22" y="43">3 个匿名轨迹 · 不含身份信息</text></g></svg>`;
}
function renderDetail(){if(!state.demo){renderLiveDetail();return;}
 const p=people[state.selected],pos=locationOf(p);$("person-detail").style.setProperty("--person-color",p.color);
 $("person-detail").innerHTML=`<div class="identity"><div class="identity-heading"><strong>${p.id}</strong><span class="status">${p.status}</span></div><p>${p.elapsed}<br>统一 ID 仅在当前匿名追踪会话内有效。</p></div><div class="coordinates"><h3 class="mini-title">地面位置 <span class="detail-unit">m</span></h3><div class="coord-grid"><div><span>X</span><strong>${pos.x.toFixed(2)}</strong></div><div><span>Y</span><strong>${pos.y.toFixed(2)}</strong></div><div><span>Z</span><strong>0.00</strong></div></div><p class="coord-note">参考点：脚部在地面的投影</p></div><div class="source-section"><h3 class="mini-title">观察来源 · ${p.sources.length} 路相机</h3>${p.sources.map(([c,l])=>`<div class="source-row"><span>相机 0${c} <small>本机</small></span><code>L-${String(l).padStart(3,"0")}</code></div>`).join("")}</div><div class="association-section"><h3 class="mini-title">跨相机关联证据 <span class="detail-score">${p.score?p.score+"%":"待确认"}</span></h3><div class="association-meter"><span class="score-${p.score||12}"></span></div><div class="evidence"><span>空间距离</span><span>${p.score?"0.12 m":"单路观察"}</span></div><div class="evidence"><span>时间与轨迹</span><span>${p.score?"连续 8 帧一致":"等待其他视角"}</span></div><div class="evidence"><span>外观辅助</span><span>${p.score?"相似":"暂无对照"}</span></div><p class="detail-note">${p.score?"演示评分，不代表已校准概率。关联需共同评估位置、时间与多帧证据。":"保留独立轨迹，避免仅凭外观或最近距离强行合并。"}</p></div><div class="inspector-footer"><button id="human-follow-person" aria-pressed="${state.focused}">${state.focused?"取消高亮":"高亮此人"}</button><button id="human-show-sources">定位观察画面</button></div>`;
 $("follow-person").onclick=()=>{state.focused=!state.focused;renderDetail();$("scene").classList.toggle("focused",state.focused);toast(state.focused?`已高亮 ${p.id}；场景与相机画面同步选中。`:"已取消额外高亮。");};
 $("show-sources").onclick=()=>{$("camera-list").scrollIntoView({behavior:"smooth",block:"nearest"});toast(`${p.id} 的来源：${p.sources.map(s=>"相机 "+s[0]+" / L-"+String(s[1]).padStart(3,"0")).join("、")}`);};}
function renderTable(){if(!state.demo){renderLiveTable();return;}const filtered=people.map((p,i)=>({p,i})).filter(({p})=>state.filter==="all"||state.filter===p.key);$("track-count").textContent=state.demo?filtered.length:"0";$("track-rows").innerHTML=state.demo?filtered.map(({p,i})=>{const pos=locationOf(p);return `<tr data-person="${i}" class="${state.selected===i?'selected':''}" tabindex="0" aria-label="选择 ${p.id}" aria-selected="${state.selected===i}"><td data-color="${p.id}"><i class="dot" data-color="${p.id}"></i>${p.id}</td><td>${p.sources.map(s=>`C${s[0]} <small>/ L-${String(s[1]).padStart(3,"0")}</small>`).join(" · ")}</td><td>${pos.x.toFixed(2)} / ${pos.y.toFixed(2)} / 0.00 <small>m</small></td><td><span class="table-state" data-color="${p.id}">${p.status}</span></td><td><small>演示当前帧</small></td></tr>`;}).join(""):'<tr><td colspan="5" class="empty-tracks">尚未接入实时追踪数据，不显示历史模拟人员。</td></tr>';}
function formatTime(t){return `00:${String(Math.floor(t)).padStart(2,"0")}`;}
function renderAll(){renderCameras();renderScene();renderDetail();renderTable();$("time-label").textContent=formatTime(state.time);$("timeline").value=state.time;if(!state.demo)renderLiveSummary();}
function setDataMode(demo){state.demo=demo;state.playing=demo;$("demo-mode").setAttribute("aria-pressed",demo);$("status-mode").setAttribute("aria-pressed",!demo);$("mode-notice").innerHTML=demo?'<span class="notice-dot"></span><div><strong>正在预览模拟追踪数据</strong><span>用于评审布局与交互。画面、轨迹、位置与运行指标均为演示，不代表 Orin 实时结果。</span></div>':'<span class="notice-dot"></span><div><strong>已核验记录 · 待刷新</strong><span>本页未连接设备。以下为项目核验记录；不代表实时在线状态。</span></div>';$("metric-people").textContent=demo?"3":"—";$("metric-people-note").textContent=demo?"2 人稳定关联 · 1 人单路观察":"等待真实追踪结果";$("metric-cameras").textContent=demo?"3 / 3":"— / 3";$("metric-cameras-note").textContent=demo?"演示输入 · 非设备在线状态":"未查询设备状态";$("metric-latency").textContent=demo?"128":"—";$("metric-memory").textContent=demo?"5.2":"—";$("events").hidden=!demo;root.querySelector(".scene-legend").hidden=!demo;root.querySelector(".scene-foot").hidden=!demo;root.querySelectorAll(".metrics article")[2].querySelector("p").textContent=demo?"模拟值 · 包含采集与关联":"未接入运行指标";root.querySelectorAll(".metrics article")[3].querySelector("p").textContent=demo?"模拟值 · 整机共享内存":"未接入运行指标";$("play-pause").disabled=!demo;$("timeline").disabled=!demo;$("speed").disabled=!demo;$("track-filter").disabled=!demo;$("play-pause").textContent=demo?"Ⅱ":"▶";$("play-pause").setAttribute("aria-label",demo?"暂停演示":"演示未启用");renderAll();}
root.addEventListener("click",event=>{const target=event.target.closest("[data-person]");if(target){state.selected=Number(target.dataset.person);renderAll();}});
root.addEventListener("keydown",event=>{if((event.key==="Enter"||event.key===" ")&&event.target.matches("[data-person]")){event.preventDefault();state.selected=Number(event.target.dataset.person);renderAll();}});
$("demo-mode").onclick=()=>setMode(true);$("status-mode").onclick=()=>setMode(false);
$("iso-view").onclick=()=>setView(false);$("top-view").onclick=()=>setView(true);
function setView(top){state.top=top;$("iso-view").setAttribute("aria-pressed",!top);$("top-view").setAttribute("aria-pressed",top);renderScene();}
$("reset-view").onclick=()=>{liveSceneView.resetFit();setView(false);toast("已复位为立体视角。");};
$("show-trails").onchange=e=>{state.trails=e.target.checked;renderScene();};$("show-fov").onchange=e=>{state.fov=e.target.checked;renderScene();};
$("play-pause").onclick=()=>{state.playing=!state.playing;$("play-pause").textContent=state.playing?"Ⅱ":"▶";$("play-pause").setAttribute("aria-label",state.playing?"暂停演示":"播放演示");};
$("timeline").oninput=e=>{state.time=Number(e.target.value);state.playing=false;$("play-pause").textContent="▶";$("play-pause").setAttribute("aria-label","播放演示");renderAll();};
$("speed").onclick=()=>{state.speed=state.speed===1?2:1;$("speed").textContent=state.speed+"×";};
$("track-filter").onchange=e=>{state.filter=e.target.value;renderTable();};
$("settings-open").onclick=()=>$("settings").showModal();
$("show-skeleton").onchange=e=>{state.skeleton=e.target.checked;renderCameras();};$("show-boxes").onchange=e=>{state.boxes=e.target.checked;renderCameras();};
$("trail-length").onchange=e=>{state.trailLength=e.target.value;renderScene();};
setMode(false);
document.addEventListener("multicamera:mode-change", event => {
 active = event.detail.mode === "human";
 if (active && !state.demo) refreshLive();
 if (!active) {liveController?.abort();resetCameraFrames();}
 if (!active && $("settings").open) $("settings").close();
});
if(!window.matchMedia("(prefers-reduced-motion: reduce)").matches)setInterval(()=>{if(active&&!document.hidden&&state.demo&&state.playing&&document.activeElement.tagName!=="TR"&&!document.activeElement.matches("[data-person]")){state.time=(state.time+state.speed)%61;renderAll();}},1000);


function escapeHTML(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function liveTracks(){
 if(liveData?.state!=='running'||!liveSceneStore.fresh(sceneNow()))return [];
 const elapsed=(performance.now()-liveReceivedAt)/1000;
 return (liveData.tracks||[]).filter(t=>t.observed&&Math.min(...(t.sources||[]).map(s=>s.age_s),Infinity)+elapsed<=.8);
}
function liveName(track){return track.global_id?'G-'+track.global_id.slice(0,8):track.sources.map(s=>'C'+s.camera_id.split('_').pop()+' / L-'+s.local_id).join(' · ');}
function selectedLive(){const tracks=liveTracks();return tracks.find(t=>t.track_uuid===liveSelected)||tracks[0];}
function setMode(demo){
 resetCameraFrames();
 document.dispatchEvent(new CustomEvent("multicamera:tracking-mode",{detail:{demo}}));
 setDataMode(demo);
 for(const id of ['show-skeleton','show-boxes'])$(id).disabled=!demo;
 for(const id of ['play-pause','timeline','speed','track-filter'])$(id).disabled=!demo;
 root.querySelector('.playback').hidden=!demo;
 $('show-trails').disabled=false;$('show-trails').checked=state.trails;
 $('show-fov').disabled=false;$('show-fov').checked=state.fov;
 $('settings-open').textContent=demo?'演示显示设置':'轨迹显示设置';
 $('settings').querySelector('h2').textContent=demo?'演示显示设置':'轨迹显示设置';
 $('settings').querySelector('.dialog-heading p').textContent=demo?'调整演示画面与历史长度。':'调整实时轨迹长度；骨架与检测框请使用追踪运行设置。';
 if(demo){liveController?.abort();restoreDemoLabels();root.querySelector('.metrics article>span').innerHTML='场景内人数 <small>去重</small>';root.querySelectorAll('.metrics article')[2].querySelector('span').textContent='追踪端到端延迟';}
 else{for(const id of ['show-skeleton','show-boxes'])$(id).checked=true;if(active)refreshLive();}
 root.querySelector('.settings-info').textContent=demo?'位置参考点：双脚中点投影到地面。三维姿态另行展示。':'实时骨架与检测框由追踪服务叠加。轨迹保留最近3秒或8秒的有效地面位置；本机轨迹不代表跨机去重。';
}
function restoreDemoLabels(){
 root.querySelector('.events-panel .tag').textContent='模拟事件';
 root.querySelector('.scene-panel .panel-heading p').textContent='演示区域 A · 8 × 6 m · 地面坐标系';
 root.querySelector('.column-note').textContent='同一人员使用相同颜色；画面内同时展示本机编号与演示统一编号。';
 root.querySelectorAll('.metrics article')[3].querySelector('div>span').textContent='GB';
 $('events').innerHTML='<li><time>演示</time><div><strong>P-001 由两路相机共同观察</strong><p>模拟位置与轨迹一致，保持演示统一 ID。</p></div></li><li><time>演示</time><div><strong>P-003 仅在相机 3 中观察</strong><p>保留单路轨迹，不强行合并。</p></div></li>';
}
function renderLiveSummary(){
 const fresh=liveSceneStore.fresh(sceneNow());
 const running=fresh&&liveData?.state==='running',cross=liveData?.cross_camera;
 const cameras=running?(liveData?.cameras||[]).filter(c=>c.state==='running'):[];
 root.querySelector('.metrics article>span').innerHTML='本机观测轨迹 <small>未跨机去重</small>';
 root.querySelectorAll('.metrics article')[2].querySelector('span').textContent='最近单帧处理耗时';
 $('metric-people').textContent=running&&cameras.length?liveTracks().length:'—';
 $('metric-people-note').textContent=!fresh?'正在等待新数据，当前人数未知':!cameras.length?'暂无可用输入，不能判断场景人数':cross?.state==='ready'?'已启用多帧跨机关联':'统一人数待跨机关联确认';
 $('metric-cameras').textContent=fresh&&liveData?cameras.length+' / 3':'— / 3';
 $('metric-cameras-note').textContent=!fresh?'等待状态确认，在线数量未知':running?'真实 RGB 左目输入':'等待追踪服务';
 $('metric-latency').textContent=running&&Number.isFinite(liveData.inference_ms)?Math.round(liveData.inference_ms):'—';
 root.querySelectorAll('.metrics article')[2].querySelector('p').textContent='推理、跟踪与绘制 · 非端到端延迟';
 $('metric-memory').textContent=liveData?.memory?.system_used_gib??'—';
 root.querySelectorAll('.metrics article')[3].querySelector('div>span').textContent='GiB';
 root.querySelectorAll('.metrics article')[3].querySelector('p').textContent=liveData?.memory?'整机使用 / '+liveData.memory.system_total_gib+' GiB':'未接入运行指标';
 $('mode-notice').replaceChildren();
 const text=document.createElement('div'),title=document.createElement('strong'),note=document.createElement('span');
 title.textContent=!fresh&&liveSceneStore.lastAccepted!==null?'连接延迟 · 保留最近有效场景':running&&!cameras.length?'追踪服务就绪 · 等待相机输入':running?'实时单机追踪 · '+(cross?.state==='ready'?'跨机关联可用':'跨机关联待就绪'):'人体追踪服务：'+({paused:'已暂停',loading:'加载中',failed:'异常',unavailable:'未连接'}[liveData?.state]||'连接中');
 note.textContent=!fresh&&liveSceneStore.lastAccepted!==null?'数据暂未更新；场景保留历史，人数和在线状态等待确认。':liveData?.state==='paused'?'已暂停采集与推理；保留模型以便快速恢复。恢复后重新建立轨迹。':liveData?.error||(running&&!cameras.length?'相机输入不可用，请检查通电与网络。'+(cross?.reason||''):cross?.reason)||'经核验的统一地面坐标；关联采用空间、时间与连续多帧证据。';
 if(cross?.geometry_ready&&cross?.reason)note.textContent='已完成地面标定：'+(cross.participating_camera_ids||[]).map(id=>id.replace('device_','设备')).join('、')+'；'+note.textContent;
 text.append(title,note);$('mode-notice').append(text);
 $('status-mode').textContent='实时追踪';
 $('events').hidden=false;$('events').replaceChildren();
 const li=document.createElement('li');li.textContent=cross?.state==='ready'?'跨机关联运行中；不确定匹配保留本机身份。':(cross?.reason||'正在读取服务状态');$('events').append(li);
 root.querySelector('.events-panel .tag').textContent='真实服务状态';
 root.querySelector('.scene-panel .panel-heading p').textContent=liveSceneStore.geometry?.world_frame||cross?.world_frame||'统一地面坐标系待标定';
 root.querySelector('.column-note').textContent=previewMode==='smooth'?'原图独立刷新；最新人体框/骨架异步叠加，不代表同帧测量，超过0.3秒的标注自动隐藏。需核对精度时切换逐帧核对。不同相机的相同编号不代表同一个人。':'逐帧核对：画面、人体框和骨架来自同一推理帧，预览帧率跟随追踪。';
 const controlState=fresh?liveData:{...liveData,state:'unavailable',options_revision:null,control_csrf:null};
 document.dispatchEvent(new CustomEvent('multicamera:tracking-state',{detail:controlState}));
}
function resetCameraFrames(){
 for(const frame of cameraFrames.values()){frame.controller?.abort();clearTimeout(frame.timer);}
 cameraFrames.clear();
 if(cameraDialog?.open)cameraDialog.close();
}
function ensureCameraDialog(){
 if(cameraDialog)return;
 cameraDialog=document.createElement('dialog');
 cameraDialog.className='camera-preview-dialog';
 cameraDialog.setAttribute('aria-labelledby','human-camera-preview-title');
 cameraDialog.innerHTML='<div class="camera-preview-heading"><h2 id="human-camera-preview-title">相机预览</h2><button type="button" data-camera-close aria-label="关闭放大预览">关闭 ×</button></div><p data-camera-info></p><div class="camera-preview-stage"><canvas aria-label="放大实时追踪画面" hidden></canvas><span data-camera-empty>等待新帧</span></div>';
 root.append(cameraDialog);
 cameraDialog.querySelector('[data-camera-close]').onclick=()=>cameraDialog.close();
 cameraDialog.addEventListener('close',()=>{enlargedCamera=null;});
}
function updateCameraDialog(){
 if(!cameraDialog?.open)return;
 const card=$('camera-list').querySelector(`[data-live-camera="${enlargedCamera}"]`);
 const source=card?.querySelector('canvas'),canvas=cameraDialog.querySelector('canvas');
 cameraDialog.querySelector('[data-camera-info]').textContent=card?card.querySelector('[data-camera-status]').textContent+' · '+card.querySelector('[data-live-age]').textContent:'等待新帧';
 const visible=source&&!source.hidden;
 canvas.hidden=!visible;cameraDialog.querySelector('[data-camera-empty]').hidden=visible;
 if(visible){if(canvas.width!==source.width||canvas.height!==source.height){canvas.width=source.width;canvas.height=source.height;}canvas.getContext('2d').drawImage(source,0,0);}
}
function cameraTrackingFresh(camera){
 return liveData?.state==='running'&&camera?.state==='running'&&Number.isFinite(camera.age_s)&&camera.age_s+(Number(camera.receive_to_result_ms)||0)/1000+(performance.now()-liveReceivedAt)/1000<=.3;
}
function drawCameraOverlay(canvas,id){
 const camera=liveData?.cameras?.find(c=>c.id===id);
 if(!cameraTrackingFresh(camera))return;
 const context=canvas.getContext('2d'),options=liveData.options||{};
 context.save();context.scale(canvas.width/camera.image_size[0],canvas.height/camera.image_size[1]);context.strokeStyle='#50dcb4';context.fillStyle='#50dcb4';context.lineWidth=2;context.font='18px sans-serif';
 for(const track of camera.tracks||[]){
  if(options.draw_boxes){const [x1,y1,x2,y2]=track.bbox_xyxy;context.strokeRect(x1,y1,x2-x1,y2-y1);context.fillText('L-'+track.local_id,x1,Math.max(20,y1-7));}
  const pose=track.keypoints;
  if(options.draw_skeleton&&pose)for(const [a,b] of [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]){
   if(pose[a]?.[2]>.4&&pose[b]?.[2]>.4){context.beginPath();context.moveTo(pose[a][0],pose[a][1]);context.lineTo(pose[b][0],pose[b][1]);context.stroke();}
  }
 }
 context.restore();
}
function updateCameraCard(frame,id){
 const camera=liveData?.cameras?.find(c=>c.id===id),card=frame.card;
 const visible=frame.received&&performance.now()-frame.received<1500;
 if(!visible){card.querySelector('canvas').hidden=true;card.querySelector('.empty-camera').hidden=false;}
 card.classList.toggle('is-offline',!visible);
 card.classList.toggle('live-source-selected',!!selectedLive()?.sources.some(source=>source.camera_id===id));
 card.querySelector('[data-camera-size]').textContent=visible?frame.size.join('×'):'等待画面';
 const times=frame.samples.filter(t=>performance.now()-t<5000);
 const fps=times.length>=2?(times.length-1)*1000/(times[times.length-1]-times[0]):0;
 card.querySelector('[data-camera-status]').textContent=visible?'预览 '+fps.toFixed(1)+' FPS · 追踪 '+(Number(camera?.processed_fps)||0).toFixed(1)+' FPS': '相机连接中 / 暂不可用';
 card.querySelector('[data-live-age]').textContent=liveData?.state==='paused'?'追踪暂停 · 原图继续':cameraTrackingFresh(camera)?'标注 '+camera.age_s.toFixed(1)+' s 前'+(previewMode==='smooth'?' · 异步叠加':' · 同帧核对'):'暂无新鲜追踪标注';
 card.querySelector('.empty-camera').textContent='暂无新鲜相机画面，正在重试';
}
async function paintPendingPreview(id,frame){
 if(frame.decoding)return;
 frame.decoding=true;
 try{
  while(frame.pending&&cameraFrames.get(id)===frame&&active&&!state.demo&&!document.hidden){
   const pending=frame.pending;frame.pending=null;
   if(pending.fingerprint===frame.fingerprint)continue;
   const started=performance.now();
   const bitmap=await createImageBitmap(pending.blob);
   try{
    if(cameraFrames.get(id)!==frame||frame.controller?.signal.aborted||!active||state.demo||document.hidden)return;
    const canvas=frame.card.querySelector('canvas');
    if(canvas.width!==bitmap.width||canvas.height!==bitmap.height){canvas.width=bitmap.width;canvas.height=bitmap.height;}
    canvas.getContext('2d').drawImage(bitmap,0,0);drawCameraOverlay(canvas,id);
    canvas.hidden=false;frame.card.querySelector('.empty-camera').hidden=true;
    frame.received=performance.now();frame.size=[bitmap.width,bitmap.height];frame.fingerprint=pending.fingerprint;
    frame.samples.push(frame.received);if(frame.samples.length>150)frame.samples.shift();
    canvas.dataset.drawCount=String(Number(canvas.dataset.drawCount||0)+1);
    canvas.dataset.fetchMs=pending.fetchMs;canvas.dataset.encodeMs=pending.encodeMs;
    canvas.dataset.decodeDrawMs=(performance.now()-started).toFixed(2);
    updateCameraCard(frame,id);updateCameraDialog();
   }finally{bitmap.close();}
  }
 }catch(error){frame.controller?.abort();}
 finally{frame.decoding=false;}
}
async function consumePreviewStream(id,frame){
 const controller=new AbortController();frame.controller=controller;
 let watchdog=setTimeout(()=>controller.abort(),2000),reader;
 try{
  const response=await fetch('/api/tracking/preview-stream/'+id,{cache:'no-store',signal:controller.signal});
  if(!response.ok)throw new Error('预览暂不可用');
  reader=response.body.getReader();let buffer=new Uint8Array(0);
  while(cameraFrames.get(id)===frame&&active&&!state.demo&&!document.hidden){
   const {done,value}=await reader.read();if(done)throw new Error('预览中断');
   clearTimeout(watchdog);watchdog=setTimeout(()=>controller.abort(),2000);
   if(buffer.length+value.length>4*1024*1024)throw new Error('预览帧过大');
   const joined=new Uint8Array(buffer.length+value.length);joined.set(buffer);joined.set(value,buffer.length);buffer=joined;
   while(true){
    let headerEnd=-1;
    for(let i=0;i+3<buffer.length;i++)if(buffer[i]===13&&buffer[i+1]===10&&buffer[i+2]===13&&buffer[i+3]===10){headerEnd=i+4;break;}
    if(headerEnd<0)break;
    if(headerEnd>4096)throw new Error('预览头过大');
    const header=new TextDecoder().decode(buffer.subarray(0,headerEnd));
    const length=Number(header.match(/Content-Length: (\d+)/i)?.[1]);
    if(!length||length>4*1024*1024)throw new Error('无效预览帧');
    if(buffer.length<headerEnd+length)break;
    frame.pending={blob:new Blob([buffer.slice(headerEnd,headerEnd+length)],{type:'image/jpeg'}),fingerprint:header.match(/X-Preview-Frame: ([^\r\n]+)/i)?.[1],fetchMs:header.match(/X-Preview-Fetch-Ms: ([^\r\n]+)/i)?.[1]||'0',encodeMs:header.match(/X-Preview-Encode-Ms: ([^\r\n]+)/i)?.[1]||'0'};
    buffer=buffer.slice(headerEnd+length);paintPendingPreview(id,frame);
   }
  }
 }catch(error){if(cameraFrames.get(id)===frame){updateCameraCard(frame,id);updateCameraDialog();}}
 finally{
  clearTimeout(watchdog);controller.abort();frame.pending=null;
  if(reader){try{await reader.cancel();}catch(error){}reader.releaseLock();}
  if(frame.controller===controller)frame.controller=null;
  if(cameraFrames.get(id)===frame&&active&&!state.demo&&!document.hidden)frame.timer=setTimeout(()=>refreshCameraFrame(id,frame),1000);
 }
}
async function refreshCameraFrame(id,frame){
 if(frame.controller||cameraFrames.get(id)!==frame||!active||state.demo||document.hidden)return;
 if(previewMode==='smooth'){consumePreviewStream(id,frame);return;}
 const started=performance.now(),camera=liveData?.cameras?.find(c=>c.id===id);
 const key=JSON.stringify([liveData?.session_id,camera?.epoch,camera?.render_revision,camera?.sample_id]);
 if(!cameraTrackingFresh(camera)||key===frame.drawn){
  if(!cameraTrackingFresh(camera)){frame.received=0;updateCameraCard(frame,id);updateCameraDialog();}
  frame.timer=setTimeout(()=>refreshCameraFrame(id,frame),50);return;
 }
 const controller=new AbortController();frame.controller=controller;
 const timer=setTimeout(()=>controller.abort(),1500);
 let bitmap,failed=false;
 try{
  const url='/api/tracking/frame/'+id+'?sample='+encodeURIComponent(key);
  const response=await fetch(url,{cache:'no-store',signal:controller.signal});
  if(!response.ok)throw new Error('画面暂不可用');
  const fingerprint=key;
  if(fingerprint&&fingerprint===frame.fingerprint)return;
  const beforeDecode=performance.now();
  bitmap=await createImageBitmap(await response.blob());
  if(cameraFrames.get(id)!==frame||controller.signal.aborted||!active||state.demo||document.hidden)return;
  const canvas=frame.card.querySelector('canvas');
  if(canvas.width!==bitmap.width||canvas.height!==bitmap.height){canvas.width=bitmap.width;canvas.height=bitmap.height;}
  canvas.getContext('2d').drawImage(bitmap,0,0);
  canvas.hidden=false;frame.card.querySelector('.empty-camera').hidden=true;
  frame.drawn=key;frame.fingerprint=fingerprint;frame.received=performance.now();frame.size=[bitmap.width,bitmap.height];
  frame.samples.push(frame.received);if(frame.samples.length>150)frame.samples.shift();
  canvas.dataset.drawCount=String(Number(canvas.dataset.drawCount||0)+1);
  canvas.dataset.fetchMs=(beforeDecode-started).toFixed(2);canvas.dataset.decodeDrawMs=(performance.now()-beforeDecode).toFixed(2);
  updateCameraCard(frame,id);updateCameraDialog();
 }catch(error){failed=true;if(cameraFrames.get(id)===frame){updateCameraCard(frame,id);updateCameraDialog();}}
 finally{
  bitmap?.close();clearTimeout(timer);if(frame.controller===controller)frame.controller=null;
  if(cameraFrames.get(id)===frame&&active&&!state.demo&&!document.hidden)frame.timer=setTimeout(()=>refreshCameraFrame(id,frame),failed?1000:Math.max(0,50-(performance.now()-started)));
 }
}
function renderLiveCameras(){
 if(!$('camera-list').querySelector('[data-live-camera]')){
  resetCameraFrames();
  $('camera-list').innerHTML=['device_1','device_2','device_3'].map((id,i)=>`<article data-live-camera="${id}" class="camera-card"><div class="camera-head"><strong>相机 0${i+1}</strong><span data-camera-size></span><button type="button" data-camera-expand="${id}" aria-label="放大相机 ${i+1}">放大预览 ↗</button></div><div class="camera-scene"><canvas class="live-camera-image" aria-label="相机 ${i+1} 实时人体追踪" hidden></canvas><div class="empty-camera">等待新帧</div></div><div class="camera-foot"><span data-camera-status></span><span data-live-age></span></div></article>`).join('');
 }
 for(const card of $('camera-list').querySelectorAll('[data-live-camera]')){
  const id=card.dataset.liveCamera;
  let frame=cameraFrames.get(id);
  if(!frame){
   frame={card,controller:null,timer:null,drawn:null,received:0,size:[],samples:[],fingerprint:null};cameraFrames.set(id,frame);
   refreshCameraFrame(id,frame);
  }
  updateCameraCard(frame,id);
 }
 updateCameraDialog();
}
function renderLiveScene(){
 const duration=state.trailLength==='short'?3:8;
 liveSceneView.update(liveSceneStore.snapshot(sceneNow(),duration),{top:state.top,fov:state.fov,trails:state.trails,name:liveName});
 $('show-fov').title=liveSceneStore.geometry?.views.length?'显示已标定相机的地面视野，未计遮挡':'暂无可用地面视野，请先核验相机地面标定';
}
function renderLiveTable(){
 const tracks=liveTracks();$('track-count').textContent=tracks.length;
 $('track-rows').innerHTML=tracks.length?tracks.map(t=>`<tr data-track-uuid="${escapeHTML(t.track_uuid)}" tabindex="0" class="${selectedLive()?.track_uuid===t.track_uuid?'selected':''}"><td>${escapeHTML(liveName(t))}</td><td>${t.sources.map(s=>escapeHTML(s.camera_id)+' / L-'+s.local_id).join(' · ')}</td><td>${t.world_xy_m?t.world_xy_m.map(n=>n.toFixed(2)).join(' / ')+' / 0.00 m':(liveData?.cross_camera?.geometry_ready?'未获得有效脚部地面位置':'待地面标定与时间核验')}</td><td>${t.global_id?'跨机已关联':'仅本机 ID'}</td><td>${Math.min(...t.sources.map(s=>s.age_s)).toFixed(1)} s 前</td></tr>`).join(''):'<tr><td colspan="5" class="empty-tracks">'+(!liveSceneStore.fresh(sceneNow())?'更新延迟，当前人员数量待确认；历史位置见空间总览':liveData?.state==='running'?((liveData.cameras||[]).some(c=>c.state==='running')?'当前新帧中暂无已确认人体轨迹':'相机输入不可用，等待新帧'):'等待人体追踪服务')+'</td></tr>';
}
function renderLiveDetail(){
 const t=selectedLive();
 $('person-detail').innerHTML=t?`<div class="empty-detail"><h3>${escapeHTML(liveName(t))}</h3><div class="capability"><strong>${t.global_id?'统一匿名 ID':'本机身份，尚未跨机合并'}</strong><span>${escapeHTML(t.track_uuid)}</span></div><div class="capability"><strong>观察来源</strong><span>${t.sources.map(s=>escapeHTML(s.camera_id)+' / L-'+s.local_id).join('<br>')}</span></div><div class="capability"><strong>地面位置 · m</strong><span>${t.world_xy_m?t.world_xy_m.map(n=>n.toFixed(2)).join(' / ')+' / 0.00':'未输出未经核验的空间位置'}</span></div><div class="capability"><strong>模型 / 本机跟踪</strong><span>YOLO26n-pose / ${escapeHTML(liveData?.local_tracker||'—')}</span></div><p>匿名 ID 仅用于当前追踪会话。脚部地面位置不等于完整人体三维姿态。</p></div>`:'<div class="empty-detail"><h3>暂无选中的真实轨迹</h3><p>实时画面出现已确认人体后，可在下方列表选择查看。未完成标定与时间核验时不合并跨机 ID。</p></div>';
}
async function refreshLive(){
 if(liveBusy||!active||state.demo||document.hidden)return;
 liveBusy=true;const requestEpoch=liveRequestEpoch;liveController=new AbortController();const timer=setTimeout(()=>liveController.abort(),3000);
 try{const response=await fetch('/api/tracking/state',{cache:'no-store',signal:liveController.signal});if(!response.ok)throw new Error('追踪状态读取失败');const data=await response.json();if(!active||state.demo||requestEpoch!==liveRequestEpoch)return;liveSceneStore.accept(data,sceneNow());liveData=data;liveReceivedAt=performance.now();renderAll();}
 catch(error){if(active&&!state.demo&&requestEpoch===liveRequestEpoch){liveSceneStore.fail();liveData={state:'unavailable',error:'人体追踪服务暂不可用',cameras:[],tracks:[]};renderAll();}}
 finally{clearTimeout(timer);liveBusy=false;liveController=null;if(requestEpoch!==liveRequestEpoch)setTimeout(refreshLive,0);}
}
root.addEventListener('click',event=>{const target=event.target.closest('[data-track-uuid]');if(target){liveSelected=target.dataset.trackUuid;renderAll();}});
root.addEventListener('keydown',event=>{if((event.key==='Enter'||event.key===' ')&&event.target.matches('[data-track-uuid]')){event.preventDefault();liveSelected=event.target.dataset.trackUuid;renderAll();}});
root.addEventListener('click',event=>{
 const button=event.target.closest('[data-camera-expand]');if(!button)return;
 ensureCameraDialog();enlargedCamera=button.dataset.cameraExpand;
 cameraDialog.querySelector('h2').textContent='相机 0'+enlargedCamera.split('_').pop()+' · 实时追踪';
 cameraDialog.showModal();updateCameraDialog();
});
document.addEventListener('multicamera:tracking-control-applied',event=>{
 const data=event.detail;liveRequestEpoch++;liveController?.abort();
 if(liveData){liveData={...liveData,state:data.state,options:data.options,options_revision:data.revision,session_id:data.session_id,local_tracker:data.options.tracker};if(data.session_reset){liveSceneStore.clearTracks();liveData.tracks=[];liveData.cameras=(liveData.cameras||[]).map(c=>({...c,state:'unavailable',tracks:[]}));}if(!state.demo)renderAll();}
 refreshLive();
});
document.addEventListener('multicamera:tracking-refresh',()=>{liveRequestEpoch++;liveController?.abort();refreshLive();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){liveController?.abort();resetCameraFrames();}else refreshLive();});
$('preview-mode').onchange=event=>{previewMode=event.target.value;resetCameraFrames();if(!state.demo)renderAll();};
setInterval(refreshLive,100);
setInterval(()=>{
 if(!active||state.demo||document.hidden)return;
 renderLiveScene();
 if(!liveSceneStore.fresh(sceneNow())){renderLiveSummary();renderLiveTable();renderLiveDetail();}
},100);
})();
