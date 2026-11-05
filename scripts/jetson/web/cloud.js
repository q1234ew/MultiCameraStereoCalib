'use strict';
// Dependency-free perspective rendering; coordinates stay in each camera's metric frame.
class CloudView {
  constructor(canvas) {
    this.canvas = canvas; this.points = []; this.colors = []; this.yaw = 0; this.pitch = 0; this.zoom = 1;
    this.center = [0,0,0]; this.radius = 1;
    canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label','三维点云，拖动或方向键旋转，滚轮或加减键缩放，Home复位');
    let drag = null;
    canvas.addEventListener('pointerdown', e => {drag = [e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointerup', () => {drag = null;});
    canvas.addEventListener('pointercancel', () => {drag = null;});
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      this.yaw += (e.clientX-drag[0])*0.008; this.pitch += (e.clientY-drag[1])*0.008;
      drag = [e.clientX,e.clientY]; this.draw();
    });
    canvas.addEventListener('wheel', e => {e.preventDefault();this.zoom = Math.max(.2,Math.min(6,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();},{passive:false});
    canvas.addEventListener('keydown', e => {
      if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','Home'].includes(e.key)) return;
      e.preventDefault();
      if (e.key==='ArrowLeft') this.yaw-=.15;
      if (e.key==='ArrowRight') this.yaw+=.15;
      if (e.key==='ArrowUp') this.pitch-=.15;
      if (e.key==='ArrowDown') this.pitch+=.15;
      if (e.key==='+'||e.key==='=') this.zoom=Math.min(6,this.zoom*1.15);
      if (e.key==='-') this.zoom=Math.max(.2,this.zoom/1.15);
      if (e.key==='Home') this.reset();
      this.draw();
    });
    new ResizeObserver(()=>this.draw()).observe(canvas);
  }
  reset() {this.yaw=this.worldAxes ? .4 : 0;this.pitch=this.worldAxes ? -.6 : 0;this.zoom=1;this.draw();}
  update(points, colors) {
    this.points=points;this.colors=colors;
    const median = a => {a.sort((x,y)=>x-y);return a[Math.floor(a.length/2)]||0;};
    this.center=[0,1,2].map(k=>median(points.map(p=>p[k])));
    const radii=points.map(p=>Math.hypot(...p.map((v,k)=>v-this.center[k]))).sort((a,b)=>a-b);
    this.radius=Math.max(.15,radii[Math.floor(radii.length*.95)]||1);
    if(this.worldAxes){this.center=[0,0,0];this.radius=Math.min(this.radius,1.5);}
    this.draw();
  }
  draw() {
    const c=this.canvas, rect=c.getBoundingClientRect();if (!rect.width) return;
    const ratio=Math.min(devicePixelRatio||1,2),w=rect.width,h=rect.height;
    c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);
    const ctx=c.getContext('2d');ctx.scale(ratio,ratio);ctx.fillStyle='#070d16';ctx.fillRect(0,0,w,h);
    ctx.font='12px system-ui';ctx.fillStyle='#b5c5d8';
    c.dataset.pointCount=String(this.points.length);
    if (!this.points.length) {ctx.fillText('无有效深度点',16,h/2);return;}
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const distance=this.radius*2.1/this.zoom,focal=Math.min(w,h)*1.2;
    const projected=[];
    for (let i=0;i<this.points.length;i++) {
      const p=this.points[i],x=p[0]-this.center[0],y=p[1]-this.center[1],z=p[2]-this.center[2];
      const xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cp*y-sp*zz,depth=sp*y+cp*zz+distance;
      if (depth<=.02) continue;
      projected.push([w/2+focal*xx/depth,h/2+focal*yy/depth,depth,i]);
    }
    projected.sort((a,b)=>b[2]-a[2]);
    for (const [x,y,z,i] of projected) {
      if (x<0||x>w||y<0||y>h) continue;
      const rgb=this.colors[i];ctx.fillStyle=`rgb(${rgb.join(',')})`;
      ctx.fillRect(x,y,2,2);
    }
    if(this.worldAxes){
      const origin=[55,h-85];
      for(const [axis,label,color] of [[[1,0,0],'X','#ff7070'],[[0,1,0],'Y','#62e39c'],[[0,0,1],'Z','#70b5ff']]){
        const [x,y,z]=axis,xx=cy*x+sy*z,zz=-sy*x+cy*z,yy=cp*y-sp*zz;
        ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(origin[0]+xx*32,origin[1]+yy*32);ctx.stroke();
        ctx.fillText(label,origin[0]+xx*40,origin[1]+yy*40);
      }
      ctx.fillStyle='#b5c5d8';ctx.fillText('世界坐标',12,h-25);
    }
    ctx.fillStyle='#b5c5d8';ctx.fillText('拖动旋转 · 滚轮缩放 · 单位 m',10,h-10);
    c.dataset.pointCount=String(this.points.length);c.dataset.yaw=String(this.yaw);
  }
}
