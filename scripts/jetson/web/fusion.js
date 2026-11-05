'use strict';
const fusionDialog=document.querySelector('#fusion-dialog');
const fusionMessage=document.querySelector('#fusion-message');
const fusionCloud=new CloudView(document.querySelector('#fusion-cloud'));
fusionCloud.worldAxes=true;fusionCloud.reset();
let fusionLastResult=null;
function fusionDevices(){return [...document.querySelectorAll('input[name=fusion-device]:checked')].map(e=>e.value);}
async function fusionStatus(){
  try{
    const data=await jsonRequest('/api/fusion/status');
    document.querySelector('#fusion-calibration').textContent=data.active_calibration?`当前外参：${data.active_calibration.devices.join('、')} · 共同码 ${data.active_calibration.anchor_tag_id} · 边长 ${data.active_calibration.tag_side_mm} mm`:'当前无有效公共外参，请先标定。';
    for(const button of document.querySelectorAll('#fusion-form button[type=submit]'))button.disabled=data.busy||!data.installed;
    if(data.job){
      fusionMessage.textContent=data.job.message+(data.job.action==='fuse'?` · ${data.job.parameters.resolution} · ${data.job.parameters.quality}`:'');
      const job=data.job;
      if(job.state==='needs_adjustment'||job.state==='failed'||job.action==='calibrate'||(job.action==='fuse'&&job.state!=='completed'&&job.id!==fusionLastResult)){
        fusionCloud.update([],[]);document.querySelector('#fusion-output').hidden=true;
        fusionLastResult=null;
      }
      if(job.state==='completed'&&job.action==='fuse'&&fusionLastResult!==job.id){
        const result=await jsonRequest(`/api/fusion/result/${job.id}`);fusionCloud.update(result.points,result.colors);fusionLastResult=job.id;document.querySelector('#fusion-output').hidden=false;
        document.querySelector('#fusion-summary').textContent=`${result.point_count.toLocaleString()} 点 · ${result.world_frame} · 单位 m`;
        if(job.alignment_report?.cameras){const checks=Object.entries(job.alignment_report.cameras).map(([id,c])=>`${id}：码面法向偏差 ${c.plane_normal_error_deg.toFixed(1)}°，深度差 P90 ${(c.depth_error_p90_m*100).toFixed(1)} cm`);document.querySelector('#fusion-summary').textContent+='。'+checks.join('；')+'。测量精度待验收。';}
      }
      const links=document.querySelector('#fusion-downloads');links.replaceChildren();
      if(job.alignment_report){const a=field('a','尺度检查报告');a.href=`/api/fusion/files/${job.id}/alignment.json`;a.download='alignment.json';links.append(a);}
      if(job.state==='completed')for(const name of (job.action==='fuse'?['pointcloud.ply','mesh.ply','pointcloud.html','mesh.html']:['calibration.json'])){
        const a=field('a',name);a.href=`/api/fusion/files/${job.id}/${name}`;a.download=name;links.append(a);
      }
    }else fusionMessage.textContent=data.installed?'融合环境就绪':'融合环境未安装';
  }catch(e){fusionMessage.textContent='读取失败：'+e.message;}
}
document.querySelector('#fusion-open').addEventListener('click',()=>{fusionDialog.showModal();document.querySelector('#fusion-depth-settings').textContent=`深度设置：${resolutionSelect.value} · ${qualitySelect.value}（沿用主界面设置）`;fusionStatus();});
document.querySelector('#fusion-close').addEventListener('click',()=>fusionDialog.close());
document.querySelector('#fusion-reset').addEventListener('click',()=>fusionCloud.reset());
document.querySelector('#fusion-refresh').addEventListener('click',fusionStatus);
document.querySelector('#fusion-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(!event.target.reportValidity())return;
  for(const b of document.querySelectorAll('#fusion-form button[type=submit]'))b.disabled=true;
  try{
    if(!cameraCSRF)await loadSystem();
    const payload={action:event.submitter.value,devices:fusionDevices(),tag_size_mm:Number(document.querySelector('#fusion-tag-size').value),tag_id:Number(document.querySelector('#fusion-tag-id').value),samples:Number(document.querySelector('#fusion-samples').value),frames:Number(document.querySelector('#fusion-frames').value),resolution:resolutionSelect.value,quality:qualitySelect.value,max_depth_m:Number(document.querySelector('#fusion-max-depth').value)};
    await jsonRequest('/api/fusion/jobs',{method:'POST',headers:{'Content-Type':'application/json','X-Camera-CSRF':cameraCSRF},body:JSON.stringify(payload)});
  }catch(e){fusionMessage.textContent='启动失败：'+e.message;for(const b of document.querySelectorAll('#fusion-form button[type=submit]'))b.disabled=false;return;}
  fusionStatus();
});
setInterval(()=>{if(fusionDialog.open)fusionStatus();},2500);
