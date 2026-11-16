'use strict';
const main = document.querySelector('#devices');
const pause = document.querySelector('#pause');
const viewer = document.querySelector('#viewer');
const large = document.querySelector('#large');
let paused = false;
let selected = null;
let mode = "raw";
let rawRefreshMs = 750;
let pointBudget = 6000;
let lastRawTick = 0;
let lastStereoTick = 0;
try {rawRefreshMs=Number(localStorage.getItem('orin-refresh-ms'))||750;pointBudget=Number(localStorage.getItem('orin-point-budget'))||6000;}catch(e){}
if (![750,1000,1500,3000].includes(rawRefreshMs)) rawRefreshMs=750;
if (![1500,3000,6000].includes(pointBudget)) pointBudget=6000;
const stereoDevices = [];
const resolutionSelect = document.querySelector('#depth-resolution');
const qualitySelect = document.querySelector('#depth-quality');
try{const q=localStorage.getItem('orin-depth-quality');if(['fast','standard','fine'].includes(q))qualitySelect.value=q;}catch(e){}
try {const stored=localStorage.getItem('orin-depth-resolution');if(['320x180','480x270','640x360','960x540'].includes(stored))resolutionSelect.value=stored;}catch(e){}
const channels = [];
function status(channel, text, state = '') {
  channel.status.textContent = text;
  channel.status.className = `status ${state}`;
}
async function update(channel) {
  if (channel.busy) return;
  channel.busy = true;
  const controller = new AbortController();
  channel.controller = controller;
  const timeout = setTimeout(() => controller.abort(), 6000);
  try {
    const response = await fetch(channel.url, {signal: controller.signal, cache: 'no-store'});
    if (!response.ok) throw new Error('unavailable');
    const blob = await response.blob();
    if (paused || document.hidden || mode !== "raw") return;
    const url = URL.createObjectURL(blob);
    const old = channel.objectUrl;
    channel.image.src = url;
    channel.objectUrl = url;
    channel.button.classList.add('has-image');
    if (selected === channel) large.src = url;
    if (old) URL.revokeObjectURL(old);
    status(channel, `画面已更新 · ${new Date().toLocaleTimeString('zh-CN', {hour12:false})}`, 'good');
  } catch (error) {
    if (!paused && !document.hidden) {
      // Never leave a stale frame looking live when the source has disconnected.
      channel.image.removeAttribute('src');
      channel.button.classList.remove('has-image');
      channel.empty.textContent = '连接中断，正在重试';
      if (selected === channel) { large.removeAttribute('src'); viewer.close(); selected = null; }
      if (channel.objectUrl) URL.revokeObjectURL(channel.objectUrl);
      channel.objectUrl = null;
      status(channel, '画面不可用 · 自动重试中', 'bad');
    }
  } finally { clearTimeout(timeout); channel.busy = false; channel.controller = null; }
}
function tick() {
  if (paused || document.hidden) return;
  const now=Date.now();
  if (mode === 'raw' && now-lastRawTick>=rawRefreshMs) {lastRawTick=now;channels.forEach(update);}
  else if (mode==='stereo' && now-lastStereoTick>=2000) {lastStereoTick=now;stereoDevices.forEach(updateStereo);}
}
pause.addEventListener('click', () => {
  paused = !paused;
  pause.textContent = paused ? '继续预览' : '暂停预览';
  channels.forEach(c => { if (paused) { c.controller?.abort(); status(c, '已暂停 · 保留最后画面'); } });
  if (paused) stereoDevices.forEach(d => {d.controller?.abort();d.summary.textContent='已暂停 · 保留最后结果';});
  if (!paused) tick();
});
document.querySelector('#refresh').addEventListener('click', () => {
  paused = false; pause.textContent = '暂停预览'; tick();
});
document.querySelector('#close').addEventListener('click', () => viewer.close());
viewer.addEventListener('close', () => {selected = null; large.removeAttribute('src');});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    channels.forEach(c => { c.controller?.abort(); status(c, '页面已隐藏 · 暂停刷新'); });
    stereoDevices.forEach(d => d.controller?.abort());
  }
  else tick();
});
async function init() {
  try {
    const response = await fetch('/api/devices');
    if (!response.ok) throw new Error('devices');
    const data = await response.json();
    main.replaceChildren();
    for (const device of data.devices) {
      const section = document.createElement('section'); section.className = 'device';
      const info = document.createElement('div'); info.className = 'device-info';
      const title = document.createElement('h2'); title.textContent = device.name;
      const host = document.createElement('p'); host.className = 'host'; host.textContent = `IP: ${device.host}`;
      info.append(title, host); section.append(info);
      const grid = document.createElement('div'); grid.className = 'channels'; section.append(grid);
      for (const source of device.channels) {
        const card = document.createElement('article'); card.className = 'channel';
        const heading = document.createElement('h3'); heading.textContent = source.label;
        const button = document.createElement('button'); button.className = 'image-button';
        button.setAttribute('aria-label', `放大${device.name} ${source.label}`);
        const image = document.createElement('img'); image.alt = `${device.name} ${source.label}`;
        const empty = document.createElement('span'); empty.className = 'empty'; empty.textContent = '正在获取画面…';
        button.append(image, empty);
        const line = document.createElement('p'); line.className = 'status'; line.textContent = '等待连接';
        card.append(heading, button, line); grid.append(card);
        const channel = {url: source.url, image, button, empty, status: line, busy: false}; channels.push(channel);
        button.addEventListener('click', () => {
          if (!channel.objectUrl) return;
          selected = channel; large.src = channel.objectUrl;
          large.alt = image.alt; document.querySelector('#viewer-title').textContent = image.alt;
          viewer.showModal();
        });
      }
      const processing = buildStereo(device); section.append(processing.panel);
      stereoDevices.push(processing);
      main.append(section);
    }
    tick(); setInterval(tick, 250);
  } catch (error) {
    main.textContent = '设备列表加载失败，请刷新页面重试。';
  }
}

function buildStereo(device) {
  const panel=document.createElement('div');panel.className='processing';panel.hidden=true;
  const grid=document.createElement('div');grid.className='stereo-grid';panel.append(grid);
  const views={};
  for (const [key,label] of [['rectified','双目校正 · 极线对照'],['depth','FoundationStereo 深度图'],['cloud','三维点云']]) {
    const card=document.createElement('article');card.className='channel';
    const heading=document.createElement('h3');heading.textContent=label;card.append(heading);
    if (key==='cloud') {
      const canvas=document.createElement('canvas');canvas.className='cloud-canvas';card.append(canvas);
      const cloud=new CloudView(canvas);
      const reset=document.createElement('button');reset.className='cloud-reset';reset.textContent='复位视角';reset.addEventListener('click',()=>cloud.reset());card.append(reset);
      views.cloud=cloud;
    } else {
      const button=document.createElement('button');button.className='image-button';button.setAttribute('aria-label',`放大${device.name} ${label}`);
      const image=document.createElement('img');image.alt=`${device.name} ${label}`;
      const empty=document.createElement('span');empty.className='empty';empty.textContent='等待模型结果';button.append(image,empty);card.append(button);
      const channel={image,button,objectUrl:null};views[key]=channel;
      button.addEventListener('click',()=>{
        if (!channel.objectUrl) return;
        selected=channel;large.src=channel.objectUrl;large.alt=image.alt;
        document.querySelector('#viewer-title').textContent=image.alt;viewer.showModal();
      });
      if (key==='depth') {
        const legend=document.createElement('p');legend.className='status';legend.textContent='0.2–10 m · 暖色近 / 冷色远 · 黑色无效';card.append(legend);
      }
    }
    grid.append(card);
  }
  const summary=document.createElement('p');summary.className='stereo-summary';summary.setAttribute('role','status');summary.textContent='切换模式后开始处理；首次需要加载模型。';panel.append(summary);
  return {id:device.id,panel,summary,views,busy:false,controller:null};
}
function setJpeg(channel,base64) {
  const bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0));
  const url=URL.createObjectURL(new Blob([bytes],{type:'image/jpeg'}));
  const old=channel.objectUrl;channel.objectUrl=url;channel.image.src=url;channel.button.classList.add('has-image');
  if (selected===channel) large.src=url;
  if (old) URL.revokeObjectURL(old);
}
async function updateStereo(device) {
  if (device.busy) return;
  device.busy=true;
  const resolution=resolutionSelect.value;
  const quality=qualitySelect.value;
  const controller=new AbortController();device.controller=controller;
  const timer=setTimeout(()=>controller.abort(),8000);
  try {
    const response=await fetch(`/api/stereo/${device.id}?resolution=${resolution}&quality=${quality}`,{signal:controller.signal,cache:'no-store'});
    if (paused||document.hidden||mode!=='stereo'||resolution!==resolutionSelect.value||quality!==qualitySelect.value) return;
    if (response.status===202) {device.summary.textContent='FoundationStereo 模型加载 / 排队 / 推理中… 首次加载较慢。';return;}
    if (!response.ok) throw new Error(await response.text());
    const result=await response.json();
    if (paused||document.hidden||mode!=='stereo'||resolution!==resolutionSelect.value||quality!==qualitySelect.value) return;
    setJpeg(device.views.rectified,result.rectified_jpeg);setJpeg(device.views.depth,result.depth_jpeg);
    const stride=Math.max(1,Math.ceil(result.points.length/pointBudget));
    device.views.cloud.update(result.points.filter((_,i)=>i%stride===0),result.colors.filter((_,i)=>i%stride===0));
    const time=new Date(result.received_at_unix_ms).toLocaleTimeString('zh-CN',{hour12:false});
    device.summary.textContent=`${result.refreshing?'更新中 · 上帧':'接收'} ${time} · ${result.resolution.join('×')} / ${result.model_identity.iterations} 次迭代 · 推理 ${(result.inference_ms/1000).toFixed(2)} s · 有效深度 ${(result.valid_fraction*100).toFixed(1)}% · 显示 ${Math.ceil(result.points.length/stride)} / ${result.points.length} 点 · ${result.point_frame} · 接收偏差 ${result.receive_skew_ms} ms。独立快照，未硬件同步；距离精度待验收。`;
    device.summary.dataset.validPixels=String(result.valid_pixels);
    device.summary.dataset.model=result.algorithm;
  } catch(error) {
    if (!paused&&!document.hidden&&mode==='stereo') {
      for (const channel of [device.views.rectified,device.views.depth]) {
        channel.image.removeAttribute('src');channel.button.classList.remove('has-image');
        if (channel.objectUrl) URL.revokeObjectURL(channel.objectUrl);channel.objectUrl=null;
        if (selected===channel) viewer.close();
      }
      device.views.cloud.update([],[]);
      device.summary.textContent='处理失败，自动重试中：'+error.message;
    }
  } finally {clearTimeout(timer);device.busy=false;device.controller=null;}
}
for (const target of ['raw','stereo','human']) {
  document.querySelector(`#${target}-mode`).addEventListener('click',()=>{
    mode=target;
    if(location.hash!==`#${target}`)location.hash=target;
    document.querySelector('#raw-mode').setAttribute('aria-pressed',String(target==='raw'));
    document.querySelector('#stereo-mode').setAttribute('aria-pressed',String(target==='stereo'));
    document.querySelector('#human-mode').setAttribute('aria-pressed',String(target==='human'));
    document.body.dataset.previewMode=target;
    main.hidden=target==='human';
    document.querySelector('#human-workspace').hidden=target!=='human';
    document.querySelector('#preview-footer').hidden=target==='human';
    if(viewer.open)viewer.close();
    document.querySelectorAll('.channels').forEach(e=>e.hidden=target!=='raw');
    stereoDevices.forEach(d=>{d.panel.hidden=target!=='stereo';if(target!=='stereo')d.controller?.abort();else d.views.cloud.draw();});
    channels.forEach(c=>c.controller?.abort());
    document.dispatchEvent(new CustomEvent('multicamera:mode-change',{detail:{mode:target}}));
    tick();
  });
}
function restorePreviewMode(){
  const target=location.hash.slice(1);
  if(['raw','stereo','human'].includes(target)&&target!==mode)
    document.querySelector(`#${target}-mode`).click();
}
window.addEventListener('hashchange',restorePreviewMode);
restorePreviewMode();
setInterval(()=>{if(mode==='stereo')tick();},2000);
function changeDepthSettings(){
  try {localStorage.setItem('orin-depth-resolution',resolutionSelect.value);localStorage.setItem('orin-depth-quality',qualitySelect.value);}catch(e){}
  for(const device of stereoDevices){
    device.controller?.abort();
    for(const channel of [device.views.rectified,device.views.depth]){
      channel.image.removeAttribute('src');channel.button.classList.remove('has-image');
      if(channel.objectUrl)URL.revokeObjectURL(channel.objectUrl);channel.objectUrl=null;
    }
    device.views.cloud.update([],[]);device.summary.textContent='深度设置已切换，等待重新计算…';
  }
  if(viewer.open)viewer.close();lastStereoTick=0;tick();
}
resolutionSelect.addEventListener('change',changeDepthSettings);
qualitySelect.addEventListener('change',changeDepthSettings);
init();
