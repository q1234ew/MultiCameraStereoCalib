'use strict';
// Offline canvas renderer. View transforms never alter the metric source coordinates.
class CloudView {
  constructor(canvas) {
    this.canvas=canvas; this.points=[]; this.colors=[];
    this.center=[0,0,0]; this.radius=1; this.yaw=0; this.pitch=0; this.roll=0; this.zoom=1;
    this.pan=[0,0]; this.pointSize=2; this.colorMode='rgb'; this.showAxes=true;
    this.showGrid=false; this.mode='rotate'; this.sources=[];this.visibleDevices=null;this.deviceColors={}; this.pending=false;
    canvas.tabIndex=0; canvas.setAttribute('role','img');
    canvas.setAttribute('aria-label','三维点云：拖动自由旋转，滚转模式摆正画面；Shift、右键或中键拖动平移；滚轮或双指缩放；方向键旋转，Q/E 滚转，F 居中，Home 复位');
    const pointers=new Map();
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointerdown',e=>{
      if(![0,1,2].includes(e.button))return;
      e.preventDefault?.();
      canvas.focus({preventScroll:true});
      pointers.set(e.pointerId,[e.clientX,e.clientY]); canvas.setPointerCapture(e.pointerId);
    });
    const release=e=>pointers.delete(e.pointerId);
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,release);
    canvas.addEventListener('pointermove',e=>{
      const previous=pointers.get(e.pointerId); if(!previous)return;
      const dx=e.clientX-previous[0],dy=e.clientY-previous[1];
      if(pointers.size===2){
        const other=[...pointers.entries()].find(([id])=>id!==e.pointerId)[1];
        const before=Math.hypot(previous[0]-other[0],previous[1]-other[1]);
        const after=Math.hypot(e.clientX-other[0],e.clientY-other[1]);
        if(before>4)this.scale(after/before);
        this.pan[0]+=dx/2; this.pan[1]+=dy/2;
      }else if(this.mode==='pan'||e.shiftKey||(e.buttons&6)){
        this.pan[0]+=dx; this.pan[1]+=dy;
      }else if(this.mode==='roll'){
        this.roll+=dx*.008;
      }else{
        this.yaw+=dx*.008; this.pitch+=dy*.008;
      }
      pointers.set(e.pointerId,[e.clientX,e.clientY]); this.schedule();
    });
    canvas.addEventListener('wheel',e=>{
      e.preventDefault(); this.scale(Math.exp(-e.deltaY*(e.deltaMode===1?.016:.001))); this.schedule();
    },{passive:false});
    canvas.addEventListener('dblclick',()=>this.fit());
    canvas.addEventListener('keydown',e=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','Home','f','F','q','Q','e','E'].includes(e.key))return;
      e.preventDefault();
      const dx=(e.key==='ArrowRight'?1:0)-(e.key==='ArrowLeft'?1:0);
      const dy=(e.key==='ArrowDown'?1:0)-(e.key==='ArrowUp'?1:0);
      if(e.shiftKey){this.pan[0]+=dx*20;this.pan[1]+=dy*20;}
      else{this.yaw+=dx*.15;this.pitch+=dy*.15;}
      if(e.key.toLowerCase()==='q')this.roll-=Math.PI/36;
      if(e.key.toLowerCase()==='e')this.roll+=Math.PI/36;
      if(e.key==='+'||e.key==='=')this.scale(1.15);
      if(e.key==='-')this.scale(1/1.15);
      if(e.key==='Home')this.reset();
      if(e.key.toLowerCase()==='f')this.fit();
      this.schedule();
    });
    new ResizeObserver(()=>this.schedule()).observe(canvas);
  }
  schedule(){
    if(this.pending)return;
    this.pending=true; requestAnimationFrame(()=>{this.pending=false;this.draw();});
  }
  scale(factor){this.zoom=Math.max(.15,Math.min(12,this.zoom*factor));}
  fit(){this.pan=[0,0];this.zoom=1;this.draw();}
  reset(){this.setView(this.worldAxes?'iso':'front');}
  setView(name){
    const views={front:[0,0],top:[0,Math.PI/2],side:[Math.PI/2,0],iso:[.6,.5]};
    [this.yaw,this.pitch]=views[name]||views.front; this.roll=0; this.fit();
  }
  level(){
    // Align projected world Z with screen up. This is a view operation, not gravity calibration.
    if(!this.worldAxes)return;
    const y=-Math.cos(this.pitch);
    // World Z maps to [0,-1,0], independent of yaw, before pitch and roll.
    // The zero horizontal component means cancelling roll is sufficient away from a pole.
    if(Math.abs(y)<1e-6){this.setView('front');return;}
    this.roll=y<0?0:Math.PI; this.schedule();
  }
  update(points,colors,sources=[]){
    const first=!this.points.length;
    this.points=[];this.colors=[];this.sources=[];
    points.forEach((p,i)=>{
      if(p.length===3 && p.every(Number.isFinite)){this.points.push(p);this.colors.push(colors[i]||[170,200,225]);this.sources.push(sources[i]);}
    });
    const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
    for(const p of this.points)for(let k=0;k<3;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}
    this.bounds=this.points.length?{low,high}:null;
    this.center=this.bounds?low.map((v,k)=>(v+high[k])/2):[0,0,0];
    this.radius=Math.max(.15,...(this.bounds?[Math.hypot(...high.map((v,k)=>(v-low[k])/2))]:[1]));
    if(first)this.fit();else this.draw();
  }
  draw(){
    const c=this.canvas,rect=c.getBoundingClientRect();if(!rect.width||!rect.height)return;
    const ratio=Math.min(devicePixelRatio||1,2),w=rect.width,h=rect.height;
    const pixelWidth=Math.round(w*ratio),pixelHeight=Math.round(h*ratio);
    if(c.width!==pixelWidth)c.width=pixelWidth;if(c.height!==pixelHeight)c.height=pixelHeight;
    const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.fillStyle='#070d16';ctx.fillRect(0,0,w,h);
    ctx.textAlign='left';ctx.font='12px system-ui';ctx.fillStyle='#b5c5d8';
    c.dataset.enabledCount=String(this.points.filter((_,i)=>!this.visibleDevices||this.visibleDevices.has(this.sources[i])).length);
    c.dataset.visibleCount='0';c.dataset.pointCount=String(this.points.length);c.dataset.yaw=String(this.yaw);
    c.dataset.pitch=String(this.pitch);c.dataset.roll=String(this.roll);
    c.dataset.zoom=String(this.zoom);c.dataset.pan=this.pan.join(',');
    if(!this.points.length){ctx.textAlign='center';ctx.fillText(this.emptyMessage||'无有效深度点',w/2,h/2);return;}
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch),cr=Math.cos(this.roll),sr=Math.sin(this.roll);
    const rotate=p=>{
      const [x,y,z]=this.worldAxes?[p[0],-p[2],p[1]]:p;
      const xx=cy*x+sy*z,zz=-sy*x+cy*z;
      const yy=cp*y-sp*zz;
      return [cr*xx-sr*yy,sr*xx+cr*yy,sp*y+cp*zz];
    };
    const distance=this.radius*3.3/this.zoom,focal=Math.min(w,h)*1.2;
    const project=p=>{
      const [x,y,z]=rotate(p.map((v,k)=>v-this.center[k])),depth=z+distance;
      return depth>.02?[w/2+this.pan[0]+focal*x/depth,h/2+this.pan[1]+focal*y/depth,depth]:null;
    };
    if(this.worldAxes&&this.showGrid){
      // World XY plane at Z=0; spacing remains in meters as the view rotates.
      const extent=Math.max(this.radius,Math.abs(this.center[0]),Math.abs(this.center[1]));
      const step=10**Math.floor(Math.log10(extent))/2,limit=Math.ceil(extent/step)*step;
      ctx.strokeStyle='#233448';ctx.lineWidth=1;
      for(let v=-limit;v<=limit+step/2;v+=step){
        for(const ends of [[[-limit,v,0],[limit,v,0]],[[v,-limit,0],[v,limit,0]]]){
          const a=project(ends[0]),b=project(ends[1]);if(!a||!b)continue;
          ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
        }
      }
      ctx.fillStyle='#b5c5d8';ctx.fillText(`XY 网格 ${Number(step.toPrecision(2))} m · Z=0`,12,22);
    }
    const projected=[];
    this.points.forEach((p,i)=>{if(this.visibleDevices&&!this.visibleDevices.has(this.sources[i]))return;const q=project(p);if(q&&q[0]>=0&&q[0]<=w&&q[1]>=0&&q[1]<=h)projected.push([...q,i]);});
    projected.sort((a,b)=>b[2]-a[2]);
    const k=2,low=this.bounds.low[k],span=this.bounds.high[k]-low||1;
    for(const [x,y,,i] of projected){
      ctx.fillStyle=this.colorMode==='height'?`hsl(${240-240*(this.points[i][k]-low)/span} 85% 60%)`:`rgb(${(this.colorMode==='device'?(this.deviceColors[this.sources[i]]||[170,200,225]):this.colors[i]).join(',')})`;
      ctx.fillRect(x-this.pointSize/2,y-this.pointSize/2,this.pointSize,this.pointSize);
    }
    c.dataset.visibleCount=String(projected.length);
    if(this.visibleDevices&&c.dataset.enabledCount==='0'){ctx.fillStyle='#b5c5d8';ctx.textAlign='center';ctx.fillText('已隐藏全部设备，请在设备图层中勾选显示',w/2,h/2);ctx.textAlign='left';}
    if(this.showAxes&&this.worldAxes){
      const origin=[50,h-60];
      for(const [axis,label,color] of [[[1,0,0],'X','#ff7070'],[[0,1,0],'Y','#62e39c'],[[0,0,1],'Z','#70b5ff']]){
        const [x,y]=rotate(axis);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(origin[0]+x*30,origin[1]+y*30);ctx.stroke();
        ctx.fillText(label,origin[0]+x*40,origin[1]+y*40);
      }
    }
    ctx.fillStyle='#b5c5d8';ctx.fillText(`视野内 ${projected.length.toLocaleString()} / ${this.points.length.toLocaleString()} 点 · ${this.zoom.toFixed(1)}×`,12,h-12);
    if(this.colorMode==='height'){
      const gradient=ctx.createLinearGradient(w-120,0,w-16,0);gradient.addColorStop(0,'hsl(240 85% 60%)');gradient.addColorStop(.5,'hsl(120 85% 60%)');gradient.addColorStop(1,'hsl(0 85% 60%)');
      ctx.fillStyle=gradient;ctx.fillRect(w-120,18,104,5);ctx.fillStyle='#b5c5d8';
      ctx.fillText(`${low.toFixed(2)} → ${this.bounds.high[k].toFixed(2)} m`,w-120,40);
    }
  }
}
