'use strict';
const settingsDialog=document.querySelector('#settings-dialog');
const statusText=document.querySelector('#system-state');
const cameraMessage=document.querySelector('#camera-message');
const controlRoot=document.querySelector('#camera-controls');
const cameraApply=document.querySelector('#camera-apply');
const cameraDevice=document.querySelector('#camera-device');
const cameraRole=document.querySelector('#camera-role');
let cameraCSRF=null, loadedControls=null, cameraLoading=false;
const rgbKeys=['auto_exposure','exposure_time_absolute','gain','white_balance_automatic','white_balance_temperature','brightness','contrast','saturation','sharpness','power_line_frequency','exposure_dynamic_framerate'];
function field(tag,text,cls='') {const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e;}
async function jsonRequest(url,options={}) {
  const response=await fetch(url,{cache:'no-store',...options});
  if(!response.ok)throw new Error(await response.text());
  return response.json();
}
async function loadSystem() {
  statusText.textContent='正在读取设备状态…';
  const refresh=document.querySelector('#status-refresh');refresh.disabled=true;
  try {
    const data=await jsonRequest('/api/status');cameraCSRF=data.csrf;
    const stateNames={ready:'模型已加载',loading:'模型加载中',not_loaded:'模型未加载（按需启动）'};
    statusText.textContent=`${data.orin.hostname} · ${data.orin.machine} · Python ${data.orin.python} · OpenCV ${data.orin.opencv} · 服务运行 ${data.orin.uptime_s} s。FoundationStereo：${stateNames[data.model.state]||data.model.state}；${data.model.jobs.length} 个推理任务排队或计算中。更新 ${new Date(data.updated_at_unix_ms).toLocaleTimeString('zh-CN',{hour12:false})}`;
    const container=document.querySelector('#device-states');container.replaceChildren();
    for(const device of data.devices) {
      const card=document.createElement('article');card.className='device-state';card.append(field('h3',`设备 ${device.id.slice(-1)} · ${device.host}`));
      const lines=[['预期 MAC',device.expected_mac],['标定会话',device.source_session],['标定包',device.calibration.loaded?'哈希校验通过':'缺失或校验失败'],['基线',device.calibration.baseline_mm==null?'未知':`${device.calibration.baseline_mm} mm`],['物理镜头身份',device.physical_binding_verified?'已核实':'待核实'],['左右路核验','参见端口映射；实体镜头标签待核实'],['参数哈希',device.calibration.sha256||'不可用']];
      const dl=document.createElement('dl');for(const [label,value] of lines)dl.append(field('dt',label),field('dd',value));card.append(dl);
      for(const ch of device.channels) {
        const label={left:'RGB 左目',right:'RGB 右目',ir:'红外'}[ch.role];
        const shape=ch.size?`${ch.size.join('×')} · 标定 ${ch.calibrated_size.join('×')}`:'无画面';
        card.append(field('p',`${label} / ${device.ports[ch.role]}：${ch.online?'在线':'离线'} · ${shape}${ch.online&&!ch.size_matches?' · 尺寸不匹配':''}`,ch.online?'state-online':'state-offline'));
      }
      container.append(card);
    }
  }catch(e){statusText.textContent='状态读取失败：'+e.message;}finally{refresh.disabled=false;}
}
function currentCameraURL(){return `/api/cameras/${cameraDevice.value}/${cameraRole.value}`;}
function addInput(key,label,value,meta) {
  const wrap=field('label',label);let input;
  if(meta.menu) {
    input=document.createElement('select');for(const item of meta.menu){const option=field('option',item.label_zh||String(item.value));option.value=String(item.value);input.append(option);}
  }else {
    input=document.createElement('input');input.type='number';input.min=meta.min;input.max=meta.max;input.step=meta.step||1;
  }
  input.name=key;input.value=String(value);input.dataset.original=String(value);
  input.disabled=Boolean(meta.inactive||(meta.flags||0)&23);wrap.append(input);
  const note=meta.inactive?'自动模式下不可用':meta.menu?'':`范围 ${meta.min}–${meta.max} · 步长 ${meta.step||1}`;
  if(note)wrap.append(field('small',note));controlRoot.append(wrap);
}
async function loadCamera() {
  if(cameraLoading)return;cameraLoading=true;cameraApply.disabled=true;controlRoot.replaceChildren();loadedControls=null;cameraMessage.textContent='正在读取当前相机参数…';
  const requested=currentCameraURL();
  try {
    const data=await jsonRequest(requested);
    if(requested!==currentCameraURL())return;
    loadedControls=data;
    if(cameraRole.value==='ir') {
      addInput('colormap','红外色板',data.colormap,{menu:['inferno','turbo','jet','viridis'].map(v=>({value:v,label_zh:v}))});
      addInput('scale','红外显示倍率',data.scale,{min:1,max:4,step:1});
      addInput('overlay','温度标注',Number(data.overlay),{menu:[{value:1,label_zh:'显示'},{value:0,label_zh:'隐藏'}]});
      cameraMessage.textContent=`当前输出 ${data.out_width}×${data.out_height}；显示倍率会改变快照尺寸，几何标定为 256×192。`;
    }else {
      for(const key of rgbKeys) {
        const meta=data.controls?.[key];if(!meta)continue;
        const rendered={...meta};if(meta.kind==='bool')rendered.menu=[{value:1,label_zh:'开启'},{value:0,label_zh:'关闭'}];
        addInput(key,meta.label_zh,meta.value,rendered);
      }
      cameraMessage.textContent=`已读取 ${data.device||'相机'} 的实际参数；曝光时间使用设备原始单位。`;
    }
    cameraApply.disabled=false;
  }catch(e){cameraMessage.textContent='读取失败：'+e.message;}finally{
    cameraLoading=false;
    if(requested!==currentCameraURL())loadCamera();
  }
}
document.querySelector('#settings-open').addEventListener('click',()=>{
  document.querySelector('#settings-depth').value=resolutionSelect.value;
  document.querySelector('#settings-quality').value=qualitySelect.value;
  document.querySelector('#settings-interval').value=String(rawRefreshMs);
  document.querySelector('#settings-points').value=String(pointBudget);
  settingsDialog.showModal();loadSystem();
});
document.querySelector('#settings-close').addEventListener('click',()=>settingsDialog.close());
for(const name of ['status','camera','preview']) {
  document.querySelector(`#${name}-tab`).addEventListener('click',()=>{
    for(const other of ['status','camera','preview']) {
      document.querySelector(`#${other}-pane`).hidden=other!==name;
      document.querySelector(`#${other}-tab`).setAttribute('aria-selected',String(other===name));
    }
    if(name==='camera')loadCamera();
    if(name==='preview')document.querySelector('#settings-depth').value=resolutionSelect.value;
  });
}
document.querySelector('#status-refresh').addEventListener('click',loadSystem);
document.querySelector('#camera-read').addEventListener('click',loadCamera);
cameraDevice.addEventListener('change',loadCamera);cameraRole.addEventListener('change',loadCamera);
document.querySelector('#camera-form').addEventListener('submit',async e=>{
  e.preventDefault();if(!loadedControls||cameraLoading)return;
  const payload={};
  for(const input of controlRoot.querySelectorAll('input,select')) {
    if(input.disabled||input.value===input.dataset.original)continue;
    if(!input.checkValidity()){input.reportValidity();return;}
    payload[input.name]=input.name==='colormap'?input.value:input.name==='overlay'?input.value==='1':Number(input.value);
  }
  if(!Object.keys(payload).length){cameraMessage.textContent='参数未变化，无需写入。';return;}
  cameraApply.disabled=true;cameraMessage.textContent='正在应用并回读确认…';
  const target=currentCameraURL();cameraDevice.disabled=true;cameraRole.disabled=true;
  try {
    if(!cameraCSRF)await loadSystem();
    await jsonRequest(target,{method:'POST',headers:{'Content-Type':'application/json','X-Camera-CSRF':cameraCSRF},body:JSON.stringify(payload)});
    await loadCamera();cameraMessage.textContent='已应用，设备回读确认成功。';
  }catch(error){await loadCamera();cameraMessage.textContent='应用失败或未完整生效，已重新读取当前值：'+error.message;}
  finally{cameraDevice.disabled=false;cameraRole.disabled=false;cameraApply.disabled=!loadedControls;}
});
document.querySelector('#settings-depth').addEventListener('change',e=>{
  resolutionSelect.value=e.target.value;resolutionSelect.dispatchEvent(new Event('change'));
});
document.querySelector('#settings-interval').addEventListener('change',e=>{
  rawRefreshMs=Number(e.target.value);try{localStorage.setItem('orin-refresh-ms',String(rawRefreshMs));}catch(error){}
});
document.querySelector('#settings-points').addEventListener('change',e=>{
  pointBudget=Number(e.target.value);try{localStorage.setItem('orin-point-budget',String(pointBudget));}catch(error){}
});

document.querySelector('#settings-quality').addEventListener('change',e=>{qualitySelect.value=e.target.value;qualitySelect.dispatchEvent(new Event('change'));});
