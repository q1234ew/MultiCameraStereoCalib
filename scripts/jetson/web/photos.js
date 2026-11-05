'use strict';
let photoURL=null;
const photoResolution=document.querySelector('#photo-resolution');
try{photoResolution.value=localStorage.getItem('orin-photo-resolution')||'preview';}catch(e){}
photoResolution.addEventListener('change',()=>{try{localStorage.setItem('orin-photo-resolution',photoResolution.value);}catch(e){}});
document.querySelector('#photo-capture').addEventListener('click',async()=>{
  const button=document.querySelector('#photo-capture'),message=document.querySelector('#photo-message');button.disabled=true;
  message.textContent=photoResolution.value==='native1080'?'正在切换拍照、采集并恢复视频，请稍候…':'正在截图并识别码…';
  try{
    if(!cameraCSRF)await loadSystem();
    const data=await jsonRequest('/api/photos',{method:'POST',headers:{'Content-Type':'application/json','X-Camera-CSRF':cameraCSRF},body:JSON.stringify({device:document.querySelector('#photo-device').value,role:document.querySelector('#photo-role').value,resolution:photoResolution.value})});
    const bytes=Uint8Array.from(atob(data.annotated_jpeg),c=>c.charCodeAt(0));
    if(photoURL)URL.revokeObjectURL(photoURL);photoURL=URL.createObjectURL(new Blob([bytes],{type:'image/jpeg'}));
    const image=document.querySelector('#photo-image');image.src=photoURL;image.hidden=false;
    const link=document.querySelector('#photo-download');link.href=data.original_url;link.hidden=false;
    message.textContent=`${data.native?'原生拍照':'当前预览截图'} ${data.size.join('×')} · ${data.elapsed_s} s · ${data.tag_ids.length?'识别到 AprilTag '+data.tag_ids.join('、'):'未检测到 AprilTag'}${data.native?' · 视频已恢复':''}`;
  }catch(e){message.textContent='截图失败：'+e.message;}finally{button.disabled=false;}
});
