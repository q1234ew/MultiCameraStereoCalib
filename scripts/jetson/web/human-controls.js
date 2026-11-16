'use strict';
(() => {
 const root=document.getElementById('human-workspace');
 const el=id=>document.getElementById('human-'+id);
 const form=el('runtime-form'),dialog=el('runtime-dialog');
 let latest=null,demo=false,busy=false,draftRevision=null,mqttSignature=null;
 const names={bytetrack:'ByteTrack',ocsort:'OC-SORT',tracktrack:'TrackTrack'};
 function controlsAvailable(){return !demo&&!!latest?.options_revision&&!!latest?.control_csrf&&latest.state!=='failed';}
 function updateButtons(){
  el('run-toggle').disabled=busy||!controlsAvailable();
  el('runtime-open').disabled=busy||!controlsAvailable();
  el('run-toggle').textContent=latest?.options?(latest.options.enabled?'暂停追踪':'启动追踪'):'连接追踪服务…';
  el('runtime-apply').disabled=busy||!controlsAvailable();
  el('runtime-reload').disabled=busy;
 }
 function message(text){el('control-message').textContent=text;el('control-message').hidden=!text;}
 function hydrate(){
  if(!latest?.options)return;
  for(const [key,value] of Object.entries(latest.options)){
   const input=form.elements.namedItem(key);if(!input)continue;
   if(input.type==='checkbox')input.checked=value;else input.value=String(value);
  }
  draftRevision=latest.options_revision;
  const options=latest.options;
  el('runtime-effective').textContent=`当前生效：${names[options.tracker]||options.tracker} · 输入 ${options.imgsz} · 每路上限 ${options.camera_fps} 帧/秒。暂停保留模型，不立即释放全部内存。`;
 }
 async function post(patch,revision){
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),5000);
  try{
   const response=await fetch('/api/tracking/control',{method:'POST',headers:{'Content-Type':'application/json','X-Camera-CSRF':latest.control_csrf},body:JSON.stringify({patch,revision}),signal:controller.signal});
   const data=await response.json();
   if(response.status===409)throw new Error('设置已被其他页面更新。请点击“重新读取”后再提交。');
   if(!response.ok)throw new Error('设置未应用，请检查服务或参数后重试。');
   latest={...latest,options:data.options,options_revision:data.revision,state:data.state};
   document.dispatchEvent(new CustomEvent('multicamera:tracking-control-applied',{detail:data}));
   return data;
  }catch(error){if(error.name==='AbortError')throw new Error('提交超时，结果尚未确认。请重新读取当前设置。');if(error instanceof TypeError||error instanceof SyntaxError)throw new Error('无法确认提交结果，请重新读取当前设置。');throw error;}
  finally{clearTimeout(timeout);}
 }
 el('runtime-open').onclick=()=>{hydrate();el('runtime-message').textContent='';dialog.showModal();};
 el('runtime-close').onclick=()=>dialog.close();
 el('runtime-reload').onclick=async()=>{
  try{const response=await fetch('/api/tracking/state',{cache:'no-store'});if(!response.ok)throw new Error();const data=await response.json();if(!data.options||!data.options_revision)throw new Error();latest=data;hydrate();el('runtime-message').textContent='已读取当前生效设置，未提交的修改已替换。';updateButtons();}
  catch(_){el('runtime-message').textContent='读取失败，未替换当前输入。';}
 };
 form.onsubmit=async event=>{
  event.preventDefault();if(!form.reportValidity()||busy||!controlsAvailable())return;
  const patch={};for(const key of ['tracker','camera_fps','imgsz','new_track_conf','jpeg_quality','draw_boxes','draw_skeleton']){const input=form.elements.namedItem(key);patch[key]=input.type==='checkbox'?input.checked:input.tagName==='SELECT'&&key==='tracker'?input.value:Number(input.value);}
  busy=true;updateButtons();el('runtime-message').textContent='正在应用…';
  try{const data=await post(patch,draftRevision);hydrate();el('runtime-message').textContent=data.session_reset?'已应用；已重新建立追踪会话，等待新帧。':'已应用，后续新帧使用新的预览设置。';message('追踪运行设置已保存到 Orin。');}
  catch(error){el('runtime-message').textContent=error.message;}
  finally{busy=false;updateButtons();}
 };
 el('run-toggle').onclick=async()=>{
  if(busy||!controlsAvailable())return;busy=true;updateButtons();
  try{const data=await post({enabled:!latest.options.enabled},latest.options_revision);message(data.options.enabled?'追踪已启动，等待相机新帧。':'追踪已暂停；保留模型，恢复后重新编号。');}
  catch(error){message(error.message);}
  finally{busy=false;updateButtons();}
 };
 document.addEventListener('multicamera:tracking-state',event=>{latest=event.detail;updateButtons();if(el('mqtt-dialog').open)renderMQTT();});
 document.addEventListener('multicamera:tracking-mode',event=>{demo=event.detail.demo;updateButtons();if(demo&&dialog.open)dialog.close();});
 async function replay(){
  const target=el('replay-result');target.textContent='正在读取…';el('replay-dialog').showModal();
  try{
   const response=await fetch('/api/tracking/replay',{cache:'no-store'});if(!response.ok)throw new Error();const data=await response.json();
   if(!data.available){target.textContent=data.note||'暂无回放评测结果';return;}
   target.replaceChildren();
   const summary=document.createElement('p');summary.className='human-runtime-note';summary.textContent=`输入 ${data.input_size} · ${data.frame_count} 个采样帧 · ${data.sample_fps} 帧/秒 · 检测处理耗时中位数 ${data.detector_median_ms.toFixed(1)} ms。各跟踪器使用同一份检测缓存。`;target.append(summary);
   const scroll=document.createElement('div');scroll.className='table-scroll';const table=document.createElement('table');const head=document.createElement('tr');
   for(const name of ['跟踪器','关联中位数 ms','P95 ms','新建本机 ID','确认观测']){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
   for(const result of data.results){const tr=document.createElement('tr');for(const value of [names[result.tracker]||result.tracker,result.association_median_ms.toFixed(2),result.association_p95_ms.toFixed(2),result.local_ids_created,result.confirmed_observations]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}
   scroll.append(table);target.append(scroll);
   const note=document.createElement('p');note.className='human-runtime-note';note.textContent='该视频没有身份真值标注：新建 ID 数不等于 ID 切换次数，不能据此比较身份准确率；关联耗时也不等于完整视频帧率。';target.append(note);
  }catch(_){target.textContent='评测报告读取失败，请稍后重试。';}
 }
 el('replay-open').onclick=replay;el('replay-close').onclick=()=>el('replay-dialog').close();
 function renderMQTT(){
  const mqtt=latest?.mqtt,container=el('mqtt-topics');
  el('mqtt-state').textContent=mqtt?.enabled?`${mqtt.connected?'已连接':'未连接'} · ${mqtt.subscriber_host||mqtt.host}:${mqtt.port} · ${mqtt.authentication?'账号认证':'无账号认证'}${mqtt.last_error?' · 输出异常，请检查服务配置':''}`:'MQTT 未启用或配置尚未读取。';
  const mapping=mqtt?.topics;const signature=JSON.stringify(mapping);if(signature===mqttSignature)return;mqttSignature=signature;container.replaceChildren();if(!mapping)return;
  const rows=[['在线状态',mapping.availability],['系统状态',mapping.status],...Object.entries(mapping.cameras||{}).map(([id,topic])=>[id+' 本机轨迹',topic]),['跨机统一轨迹',mapping.global_tracks],['旧 JSON 协议兼容',mapping.legacy_report]];
  for(const [name,topic] of rows){if(!topic)continue;const label=document.createElement('label');label.className='human-topic-row';const title=document.createElement('span');title.textContent=name;const input=document.createElement('input');input.readOnly=true;input.value=topic;input.setAttribute('aria-label',name+' topic');input.onfocus=()=>input.select();label.append(title,input);container.append(label);}
 }
 el('mqtt-open').onclick=()=>{renderMQTT();el('mqtt-dialog').showModal();};el('mqtt-close').onclick=()=>el('mqtt-dialog').close();
 updateButtons();
})();
