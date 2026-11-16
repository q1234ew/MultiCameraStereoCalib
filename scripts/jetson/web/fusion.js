'use strict';
const fusionDialog=document.querySelector('#fusion-dialog');
const fusionMessage=document.querySelector('#fusion-message');
const fusionCloud=new CloudView(document.querySelector('#fusion-cloud'));
fusionCloud.worldAxes=true;fusionCloud.reset();
let fusionLastResult=null;
let fusionDraftInitialized=false;
const fusionResolution=document.querySelector('#fusion-resolution');
const fusionQuality=document.querySelector('#fusion-quality');
const fusionDeviceColors={device_1:[77,190,255],device_2:[255,172,76],device_3:[202,150,255]};
const qualityNames={fast:'快速 · 8 次迭代',standard:'标准 · 16 次迭代',fine:'精细 · 32 次迭代'};
const deviceName=id=>({device_1:'设备 1',device_2:'设备 2',device_3:'设备 3'}[id]||id);
let fusionData=null,fusionSystem=null,fusionChecking=false,fusionSubmitting=false;
let fusionSelectionInitialized=false,fusionCheckedAt=0,fusionStatusEpoch=0;
const fusionNode=id=>document.getElementById('fusion-'+id);
function previewTolerance(){return fusionNode('tolerance').value==='preview';}
function strictAdjustment(){const job=fusionData?.job;return !previewTolerance()&&(job?.state==='needs_adjustment'||job?.alignment_report?.strict_passed===false);}
function matchingCaptureCache(){
 const ids=fusionDevices().sort();
 return (fusionData?.cache?.captures||[]).find(c=>c.compatible&&c.age_s<=600&&c.frames>=Number(fusionNode('frames').value)&&c.resolution===fusionResolution.value&&c.quality===fusionQuality.value&&JSON.stringify([...c.devices].sort())===JSON.stringify(ids));
}
function renderFusionCache(){
 const cached=matchingCaptureCache(),reuse=fusionNode('reuse-cache'),busy=!!fusionData?.busy||fusionSubmitting;
 if(!cached)reuse.checked=false;
 reuse.disabled=!cached||busy;fusionNode('clear-cache').disabled=busy||!(fusionData?.cache?.captures?.length||fusionData?.cache?.preview_entries);
 fusionNode('cache-detail').textContent=cached?'可复用 '+cached.frames+' 帧/设备 · 采集于 '+new Date(cached.captured_at_unix_ms).toLocaleTimeString()+'（约'+Math.max(0,cached.age_s)+'秒前）。复用后仍执行尺度与方向检查。':'当前参数无匹配深度缓存，下次会重新采集。参数/外参变化、服务重启或超过10分钟后需重新采集。';
 const total=fusionDevices().length*Number(fusionNode('frames').value),last=fusionData?.job;
 let cost='本次 '+total+' 组深度'+(fusionNode('output-mode').value==='mesh'?'，另生成TSDF网格。':'，仅生成点云，跳过网格。');
 if(reuse.checked&&cached)cost='复用历史深度，跳过 '+total+' 组GPU推理；场景变化时请取消勾选并重新采集。';
 else if(last?.parameters?.resolution===fusionResolution.value&&last?.parameters?.quality===fusionQuality.value&&last.capture_metrics?.length){const times=last.capture_metrics.map(m=>m.inference_ms).filter(Number.isFinite);if(times.length)cost+=' 上次同档推理约 '+(times.reduce((a,b)=>a+b,0)/times.length*total/1000).toFixed(1)+' 秒/本组（不含采集与融合，实际会波动）。';}
 fusionNode('cost-note').textContent=cost;
 const info=last?.capture_cache;fusionNode('job-cache-note').hidden=!info?.hit;
 if(info?.hit)fusionNode('job-cache-note').textContent='本任务复用历史深度：'+new Date(info.captured_at_unix_ms).toLocaleString()+' · 来源任务 '+info.id.slice(0,8)+'；不是当前实时采集。';
 const timings=last?.timings_ms;fusionNode('timings').hidden=!timings;
 if(timings)fusionNode('timings').textContent=Object.entries(timings).map(([k,v])=>({capture:info?.hit?'缓存恢复':'深度采集与推理',fusion:'一致性检查与融合',total:'总耗时'}[k]||k)+' '+(v/1000).toFixed(2)+' s').join(' · ');
}
function deviceReady(id){
 const d=fusionSystem?.devices?.find(d=>d.id===id);
 return !!d?.calibration?.loaded&&['left','right'].every(role=>d.channels?.some(c=>c.role===role&&c.online&&c.size_matches));
}
function selectionReady(){
 const ids=fusionDevices();return !!fusionSystem&&Date.now()-fusionCheckedAt<30000&&ids.length>=2&&ids.every(deviceReady);
}
function calibrationMatches(){
 const ids=fusionDevices(),active=fusionData?.active_calibration;
 return ids.length>=2&&!!active&&ids.every(id=>active.devices.includes(id));
}
function initialFusionSelection(){
 if(fusionSelectionInitialized||!fusionData||!fusionSystem)return;
 const active=fusionData.active_calibration;
 const ids=active?.devices?.length>=2?active.devices:fusionSystem.devices.filter(d=>deviceReady(d.id)).map(d=>d.id);
 for(const input of document.querySelectorAll('input[name=fusion-device]'))input.checked=ids.includes(input.value);
 if(active)fusionNode('tag-size').value=active.tag_side_mm;
 fusionSelectionInitialized=true;
}
function renderFusionFlow(){
 initialFusionSelection();renderFusionCache();
 const ids=fusionDevices(),busy=!!fusionData?.busy||fusionSubmitting,checked=!!fusionSystem&&Date.now()-fusionCheckedAt<30000;
 const ready=selectionReady(),matched=calibrationMatches(),adjustment=strictAdjustment();
 fusionNode('inputs').disabled=busy;
 for(const input of document.querySelectorAll('[data-fusion-device-status]')){
  const id=input.dataset.fusionDeviceStatus,d=fusionSystem?.devices?.find(d=>d.id===id);
  const stereo=['left','right'].every(role=>d?.channels?.some(c=>c.role===role&&c.online));
  input.textContent=!checked?'待检查连接':!stereo?'双目画面不可用':!deviceReady(id)?'尺寸或单机标定不匹配':(fusionData?.active_calibration?.devices.includes(id)?'双目可用 · 外参已配置':'双目可用 · 缺少跨机外参');
 }
 const canStart=ready&&!busy&&!fusionChecking&&fusionData?.installed;
 fusionNode('calibrate-start').disabled=!canStart;
 fusionNode('start').disabled=!canStart||!matched||adjustment;
 fusionNode('next-action').disabled=!canStart;
 fusionNode('next-action').textContent=busy?'正在执行…':matched&&!adjustment?'开始采集与融合':'标定所选设备外参';
 fusionNode('use-calibrated').disabled=busy||!fusionData?.active_calibration;
 fusionNode('use-online').disabled=busy||!checked;
 fusionNode('check').disabled=busy||fusionChecking;
 let title,note;
 if(busy){title='任务正在 Orin 上执行';note='采集参数已锁定。可以关闭窗口，重新打开后继续查看进度。';}
 else if(fusionChecking||!checked){title=fusionChecking?'正在检查双目连接与单机参数…':'需要重新检查设备连接';note='仅检查图像接口、尺寸和本地标定文件，不会写入外参。';}
 else if(!fusionData){title='无法读取融合任务状态';note='请刷新任务状态后再启动，避免重复提交。';}
 else if(!fusionData.installed){title='融合运行环境未就绪';note='请先恢复 Orin 融合运行环境。';}
 else if(ids.length<2){title='先选择至少两台设备';note='参与融合的设备需有可用双目画面及匹配的单机标定。';}
 else if(!ready){title='所选设备尚未就绪';note=ids.filter(id=>!deviceReady(id)).map(deviceName).join('、')+' 的双目连接、图像尺寸或单机标定未通过检查；恢复后重新检查。';}
 else if(!matched||adjustment){title='下一步：为所选设备确认跨机外参';note=!matched?'所选组合缺少匹配外参。可选择“已有外参设备”，或保持相机与共同码静止后重新标定。':'最近一致性检查未通过，请先核验单机深度和共同码，再重新标定。';}
 else{title=previewTolerance()?'预览容差已启用，可查看叠加效果':'设备和外参配置匹配，可以开始融合';note='参与：'+ids.map(deviceName).join('、')+'。若相机或码的位置已变化，请先重新标定；配置匹配不代表位置与测量精度已验收。';}
 fusionNode('tolerance-note').textContent=previewTolerance()?'预览档：比例±10%、P90≤15cm、法向≤10°。只改变输出门限，不修正偏差；输出将标记为预览，不能用于精确测量。':'严格档：比例±5%、P90≤5cm、法向≤10°。';
 fusionNode('next-title').textContent=title;fusionNode('next-note').textContent=note;
 const calibrating=busy&&fusionData?.job?.action==='calibrate';
 const states={devices:busy||ready?'ready':'current',calibration:calibrating?'current':busy||ready&&matched&&!adjustment?'ready':ready?'current':'pending',capture:!calibrating&&(busy||ready&&matched&&!adjustment&&!fusionLastResult)?'current':'pending',result:fusionLastResult?'ready':'pending'};
 for(const step of document.querySelectorAll('[data-fusion-step]')){step.dataset.state=states[step.dataset.fusionStep];if(step.dataset.state==='current')step.setAttribute('aria-current','step');else step.removeAttribute('aria-current');}
 const job=fusionData?.job;fusionNode('progress-panel').hidden=!fusionData?.busy;
 if(job&&fusionData.busy){
  const elapsed=Math.max(0,Math.floor((Date.now()-job.created_at_unix_ms)/1000));
  fusionNode('elapsed').textContent='已运行 '+Math.floor(elapsed/60)+'分 '+elapsed%60+'秒';
  const stage=job.stage||'preparing',progress=fusionNode('progress');
  fusionNode('progress-title').textContent=({preparing:'准备任务',calibration:'采集共同码并计算外参',capture:'采集双目深度',fusion:'融合点云与生成网格'}[stage]||'任务处理中');
  if(stage==='capture'&&job.progress?.total){progress.max=job.progress.total;progress.value=job.progress.done;fusionNode('progress-note').textContent='已采集 '+job.progress.done+' / '+job.progress.total+' 组；关闭窗口后任务仍会继续。';}
  else{progress.removeAttribute('value');fusionNode('progress-note').textContent='此阶段没有可靠的完成百分比；关闭窗口后任务仍会继续。';}
 }
}
async function checkFusionConnections(){
 if(fusionChecking)return false;
 fusionChecking=true;renderFusionFlow();const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
 try{fusionSystem=await jsonRequest('/api/status',{signal:controller.signal});fusionCheckedAt=Date.now();}
 catch(error){fusionSystem=null;fusionCheckedAt=0;fusionMessage.textContent='连接检查失败，请确认 Orin 服务后重新检查。';fusionMessage.dataset.state='error';}
 finally{clearTimeout(timer);fusionChecking=false;renderFusionFlow();}
 return selectionReady();
}
function fusionFailure(job){
 if(job.state==='needs_adjustment'&&job.alignment_report?.cameras){
  const failed=Object.entries(job.alignment_report.cameras).filter(([,c])=>!c.passed);
  if(failed.length)return job.message+' '+failed.map(([id,c])=>deviceName(id)+'：法向偏差 '+Number(c.plane_normal_error_deg).toFixed(1)+'°，深度差P90 '+(Number(c.depth_error_p90_m)*100).toFixed(1)+' cm').join('；')+'。请下载尺度检查报告核验。';
 }
 if(job.message==='Bad Gateway')return '相机视频接口未响应，未生成融合结果。请重新检查连接后重试。';
 return job.message||'任务未完成，请查看任务记录。';
}
function recovery(title,reason,steps=true){
  document.querySelector('#fusion-recovery').hidden=false;
  document.querySelector('#fusion-recovery-title').textContent=title;
  document.querySelector('#fusion-recovery-reason').textContent=reason;
  document.querySelector('#fusion-recovery-steps').hidden=!steps;
  document.querySelector('#fusion-recovery-note').hidden=!steps;
}
function showFusionLayers(result){
  const root=document.querySelector('#fusion-layer-list');root.replaceChildren();
  document.querySelector('#fusion-layers').hidden=false;
  const supported=result.source_labels_available===true && result.point_device_ids?.length===result.points.length && result.devices?.length>0;
  document.querySelector('#fusion-color option[value=device]').disabled=!supported;
  if(!supported){
    fusionCloud.visibleDevices=null;
    if(fusionCloud.colorMode==='device'){fusionCloud.colorMode='rgb';document.querySelector('#fusion-color').value='rgb';}
    document.querySelector('#fusion-layer-note').textContent='此结果缺少可验证的设备来源标签，仅支持整体查看。新任务会保存来源标签；不会按点的位置猜测来源。';
    return;
  }
  fusionCloud.visibleDevices=new Set(result.devices.map(d=>d.id));
  fusionCloud.deviceColors=fusionDeviceColors;
  for(const device of result.devices){
    const label=document.createElement('label'),input=document.createElement('input');
    input.type='checkbox';input.checked=true;input.value=device.id;input.name='fusion-layer';
    const swatch=field('span','','fusion-swatch');swatch.style.background=`rgb(${(fusionDeviceColors[device.id]||[170,200,225]).join(',')})`;
    label.append(input,swatch,document.createTextNode(deviceName(device.id)),field('small',`保留 ${device.point_count.toLocaleString()} / 预览 ${device.preview_count.toLocaleString()} 点`));
    input.addEventListener('change',()=>{if(input.checked)fusionCloud.visibleDevices.add(device.id);else fusionCloud.visibleDevices.delete(device.id);fusionCloud.draw();});
    root.append(label);
  }
  document.querySelector('#fusion-layer-note').textContent='图层仅控制当前预览；采集设备在右侧设置。颜色图例用于“设备分色”；保留点是体素去重后归属该设备的点，不代表全部重叠观测。';
}
document.querySelector('#fusion-check-single').addEventListener('click',()=>{
  fusionDialog.close();document.querySelector('#stereo-mode').click();
  document.querySelector('#stereo-mode').focus();document.querySelector('#devices').scrollIntoView({block:'start'});
});
document.querySelector('#fusion-check-extrinsics').addEventListener('click',()=>{
  document.querySelector('.fusion-setup details').open=true;
  if(fusionDialog.classList.contains('fusion-expanded'))document.querySelector('#fusion-expand').click();
  const target=document.querySelector('#fusion-calibrate-start');target.scrollIntoView({block:'center'});target.focus({preventScroll:true});
});
for(const [select,key] of [[fusionResolution,'orin-fusion-resolution'],[fusionQuality,'orin-fusion-quality']])select.addEventListener('change',()=>{try{localStorage.setItem(key,select.value);}catch(e){}});
let fusionReading=false;
fusionCloud.emptyMessage='暂无有效融合点云，请先确认外参与任务状态';
function clearFusionResult(){
  fusionDialog.querySelector('.fusion-badge').textContent='测量精度待验收';fusionCloud.visibleDevices=null;fusionCloud.update([],[]);fusionDialog.classList.remove('fusion-has-result');document.querySelector('#fusion-output').hidden=true;
  document.querySelector('#fusion-layers').hidden=true;document.querySelector('#fusion-layer-list').replaceChildren();
  document.querySelector('#fusion-result-settings').textContent='当前显示结果：暂无';
  document.querySelector('#fusion-downloads').replaceChildren();fusionLastResult=null;
}
document.querySelector('#fusion-fit').addEventListener('click',()=>fusionCloud.fit());
document.querySelector('#fusion-level').addEventListener('click',()=>fusionCloud.level());
for(const [id,angle] of [['fusion-roll-left',-Math.PI/12],['fusion-roll-right',Math.PI/12]])document.getElementById(id).addEventListener('click',()=>{fusionCloud.roll+=angle;fusionCloud.schedule();});
for(const [id,factor] of [['fusion-zoom-in',1.2],['fusion-zoom-out',1/1.2]])document.getElementById(id).addEventListener('click',()=>{fusionCloud.scale(factor);fusionCloud.schedule();});
for(const button of document.querySelectorAll('[data-cloud-view]'))button.addEventListener('click',()=>fusionCloud.setView(button.dataset.cloudView));
document.querySelector('#fusion-drag').addEventListener('change',e=>{fusionCloud.mode=e.target.value;});
document.querySelector('#fusion-color').addEventListener('change',e=>{fusionCloud.colorMode=e.target.value;fusionCloud.draw();});
document.querySelector('#fusion-point-size').addEventListener('input',e=>{
  fusionCloud.pointSize=Number(e.target.value);document.querySelector('#fusion-size-value').textContent=e.target.value+' px';fusionCloud.schedule();
});
for(const [id,property] of [['fusion-axes','showAxes'],['fusion-grid','showGrid']])document.getElementById(id).addEventListener('change',e=>{fusionCloud[property]=e.target.checked;fusionCloud.draw();});
document.querySelector('#fusion-expand').addEventListener('click',e=>{
  const expanded=fusionDialog.classList.toggle('fusion-expanded');
  e.target.textContent=expanded?'恢复布局':'放大查看';e.target.setAttribute('aria-pressed',String(expanded));fusionCloud.schedule();
});
document.querySelector('#fusion-save').addEventListener('click',()=>{
  if(!fusionCloud.points.length)return;
  fusionCloud.draw();const a=document.createElement('a');
  a.download=`fusion-view-${fusionLastResult}.png`;a.href=fusionCloud.canvas.toDataURL('image/png');a.click();
});
function fusionDevices(){return [...document.querySelectorAll('input[name=fusion-device]:checked')].map(e=>e.value);}
async function fusionStatus(){
  if(fusionReading)return;fusionReading=true;const epoch=fusionStatusEpoch;
  try{
    const data=await jsonRequest('/api/fusion/status');
    if(epoch!==fusionStatusEpoch)return;
    fusionData=data;
    const adjustment=strictAdjustment();
    document.querySelector('#fusion-recovery').hidden=true;
    if(!data.busy&&(!data.active_calibration||adjustment))recovery(adjustment?'核验未通过，融合已暂停':'无有效跨机外参',adjustment?fusionFailure(data.job):'请先核验单机画面，再采集共同码建立与当前设备参数匹配的跨机外参。');
    if(data.job?.state==='failed')recovery('上次任务未完成',fusionFailure(data.job),false);
    if(!data.installed)recovery('融合环境未就绪','请先恢复 Orin 的融合运行环境；当前不能启动采集任务。',false);
    fusionMessage.dataset.state=adjustment||data.job?.state==='failed'?'error':'normal';
    document.querySelector('#fusion-calibration').textContent=data.active_calibration?`当前外参：${data.active_calibration.devices.map(deviceName).join('、')} · 共同码 ${data.active_calibration.anchor_tag_id} · 边长 ${data.active_calibration.tag_side_mm} mm`:'当前无有效公共外参，请先标定。';
    for(const button of document.querySelectorAll('#fusion-form button[type=submit]'))button.disabled=data.busy||!data.installed||(button.value==='fuse'&&(!data.active_calibration||adjustment));
    if(data.job){
      fusionMessage.textContent=(data.busy?'':data.job.state==='failed'?'上次任务：':'')+fusionFailure(data.job);
      const job=data.job;
      if(job.state==='needs_adjustment'||job.state==='failed'||job.action==='calibrate'||!data.active_calibration||(job.action==='fuse'&&job.state!=='completed')){
        clearFusionResult();
      }
      if(data.active_calibration&&job.state==='completed'&&job.action==='fuse'&&fusionLastResult!==job.id){
        const result=await jsonRequest(`/api/fusion/result/${job.id}`);
        fusionDialog.querySelector('.fusion-badge').textContent=result.preview_only?(result.strict_alignment_passed?'预览档 · 测量精度待验收':'宽容差预览 · 严格检查未通过'):'测量精度待验收';showFusionLayers(result);fusionCloud.update(result.points,result.colors,result.source_labels_available?result.point_device_ids:[]);fusionCloud.reset();fusionLastResult=job.id;fusionDialog.classList.add('fusion-has-result');document.querySelector('#fusion-output').hidden=false;
        document.querySelector('#fusion-result-settings').textContent=`当前显示结果：${job.parameters.tolerance_profile==='preview'?'预览容差 · ':''}${job.parameters.devices.map(deviceName).join("、")} · ${job.parameters.resolution.replace('x','×')} · ${qualityNames[job.parameters.quality]||job.parameters.quality} · 不随下次设置改变`;
        document.querySelector('#fusion-summary').textContent=`完整结果 ${result.point_count.toLocaleString()} 点 · 预览采样 ${result.points.length.toLocaleString()} 点 · ${result.world_frame} · 单位 m`;
        if(job.alignment_report?.cameras){const checks=Object.entries(job.alignment_report.cameras).map(([id,c])=>`${id}：码面法向偏差 ${c.plane_normal_error_deg.toFixed(1)}°，深度差 P90 ${(c.depth_error_p90_m*100).toFixed(1)} cm`);document.querySelector('#fusion-summary').textContent+='。'+checks.join('；')+'。测量精度待验收。';}
      }
      const links=document.querySelector('#fusion-downloads');links.replaceChildren();
      const artifactNames={'task.json':'任务记录','workflow.log':'运行日志','alignment.json':'尺度检查报告','calibration.json':'外参报告','report.json':'融合报告','pointcloud.ply':'点云 PLY','mesh.ply':'网格 PLY','pointcloud.html':'点云 HTML','mesh.html':'网格 HTML'};
      for(const name of job.available_artifacts||[]){
        if(!artifactNames[name])continue;
        const a=field('a',(job.parameters?.tolerance_profile==='preview'&&name.endsWith('.ply')?'预览 · ':'')+artifactNames[name]);a.href=`/api/fusion/files/${job.id}/${name}`;a.download=(job.parameters?.tolerance_profile==='preview'&&name.endsWith('.ply')?'preview_':'')+name;links.append(a);
      }
    }else{clearFusionResult();fusionMessage.textContent=data.installed?'融合环境就绪':'融合环境未安装';}
  }catch(e){if(epoch!==fusionStatusEpoch)return;fusionData=null;clearFusionResult();fusionMessage.dataset.state='error';fusionMessage.textContent='读取失败：'+e.message;recovery('无法显示当前融合结果',e.message,e.message.includes('标定')||e.message.includes('尺度'));}
  finally{fusionReading=false;renderFusionFlow();}
}
document.querySelector('#fusion-open').addEventListener('click',()=>{
  if(!fusionDraftInitialized){
    for(const [select,key,fallback] of [[fusionResolution,'orin-fusion-resolution',resolutionSelect.value],[fusionQuality,'orin-fusion-quality',qualitySelect.value]]){
      let value=fallback;try{value=localStorage.getItem(key)||fallback;}catch(e){}
      select.value=[...select.options].some(o=>o.value===value)?value:fallback;
    }
    fusionDraftInitialized=true;
  }
  fusionDialog.showModal();fusionStatus();checkFusionConnections();
});
document.querySelector('#fusion-close').addEventListener('click',()=>fusionDialog.close());
document.querySelector('#fusion-reset').addEventListener('click',()=>fusionCloud.reset());
document.querySelector('#fusion-refresh').addEventListener('click',fusionStatus);
document.querySelector('#fusion-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(!event.target.reportValidity())return;
  if(fusionSubmitting||fusionData?.busy)return;
  const action=event.submitter?.value;if(!action)return;
  if(!await checkFusionConnections()){fusionMessage.textContent='所选设备未就绪，请先处理连接检查提示。';return;}
  if(action==='fuse'&&(!calibrationMatches()||strictAdjustment())){renderFusionFlow();return;}
  fusionSubmitting=true;renderFusionFlow();
  try{
    if(!cameraCSRF)await loadSystem();
    const payload={action,devices:fusionDevices(),tag_size_mm:Number(document.querySelector('#fusion-tag-size').value),tag_id:Number(document.querySelector('#fusion-tag-id').value),samples:Number(document.querySelector('#fusion-samples').value),frames:Number(document.querySelector('#fusion-frames').value),resolution:fusionResolution.value,quality:fusionQuality.value,build_mesh:fusionNode('output-mode').value==='mesh',tolerance_profile:previewTolerance()?'preview':'strict',capture_cache_id:action==='fuse'&&fusionNode('reuse-cache').checked?matchingCaptureCache()?.id:null,max_depth_m:Number(document.querySelector('#fusion-max-depth').value)};
    const accepted=await jsonRequest('/api/fusion/jobs',{method:'POST',headers:{'Content-Type':'application/json','X-Camera-CSRF':cameraCSRF},body:JSON.stringify(payload)});
    fusionStatusEpoch++;fusionData={...fusionData,busy:true,job:accepted};
  }catch(e){fusionMessage.textContent='启动失败：'+e.message;fusionMessage.dataset.state='error';fusionSubmitting=false;await fusionStatus();fusionMessage.textContent='启动失败：'+e.message;return;}
  fusionSubmitting=false;renderFusionFlow();fusionStatus();
});
setInterval(()=>{if(fusionDialog.open)fusionStatus();},2500);

fusionNode('check').addEventListener('click',()=>{fusionStatus();checkFusionConnections();});
fusionNode('devices').addEventListener('change',()=>{fusionSelectionInitialized=true;renderFusionFlow();});
fusionNode('use-calibrated').addEventListener('click',()=>{for(const input of document.querySelectorAll('input[name=fusion-device]'))input.checked=!!fusionData?.active_calibration?.devices.includes(input.value);fusionSelectionInitialized=true;renderFusionFlow();});
fusionNode('use-online').addEventListener('click',()=>{for(const input of document.querySelectorAll('input[name=fusion-device]'))input.checked=deviceReady(input.value);fusionSelectionInitialized=true;renderFusionFlow();});
fusionNode('form').addEventListener('invalid',event=>{const details=event.target.closest('details');if(details)details.open=true;},true);
setInterval(()=>{if(fusionDialog.open&&!document.hidden&&!fusionData?.busy)checkFusionConnections();},15000);

fusionNode('next-action').addEventListener('click',()=>{const action=calibrationMatches()&&!strictAdjustment()?'start':'calibrate-start';fusionNode('form').requestSubmit(fusionNode(action));});

fusionNode('form').addEventListener('change',()=>{renderFusionCache();try{const draft={};for(const id of ['frames','samples','tag-size','tag-id','max-depth','output-mode'])draft[id]=fusionNode(id).value;localStorage.setItem('orin-fusion-draft',JSON.stringify(draft));}catch(error){}});
for(const button of document.querySelectorAll('[data-fusion-preset]'))button.addEventListener('click',()=>{const presets={quick:['320x180','fast','1','cloud'],balanced:['480x270','standard','3','cloud'],detail:['640x360','fine','3','mesh']};const [resolution,quality,frames,mode]=presets[button.dataset.fusionPreset];fusionResolution.value=resolution;fusionQuality.value=quality;fusionNode('frames').value=frames;fusionNode('output-mode').value=mode;fusionNode('reuse-cache').checked=false;fusionResolution.dispatchEvent(new Event('change',{bubbles:true}));fusionQuality.dispatchEvent(new Event('change',{bubbles:true}));fusionNode('form').dispatchEvent(new Event('change',{bubbles:true}));});
fusionNode('clear-cache').addEventListener('click',async()=>{try{if(!cameraCSRF)await loadSystem();await jsonRequest('/api/fusion/cache/clear',{method:'POST',headers:{'Content-Type':'application/json','X-Camera-CSRF':cameraCSRF}});fusionNode('reuse-cache').checked=false;await fusionStatus();}catch(error){fusionMessage.textContent='清除缓存失败：'+error.message;}});
try{const draft=JSON.parse(localStorage.getItem('orin-fusion-draft')||'{}');for(const id of ['frames','samples','tag-size','tag-id','max-depth','output-mode']){if(draft[id]!==undefined){const input=fusionNode(id),previous=input.value;input.value=draft[id];if(!input.value||!input.validity.valid)input.value=previous;}}}catch(error){}

fusionNode('tolerance').addEventListener('change',()=>{renderFusionFlow();fusionStatus();});
