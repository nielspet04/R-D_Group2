import fs from 'node:fs/promises';
import {root} from './build-modern.mjs';
const data=(await Promise.all([0,6,12,18].map(i=>fs.readFile(root+`/batch-${i}.json`,'utf8').then(JSON.parse)))).flat();
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const diagrams={
4:`flowchart LR
 I["Ego inputs"] --> T["Causal<br/>transformer"]
 T --> P["Pose queries"]
 T --> E["Predicted<br/>Exo features"]
 P --> G["GLMM"]
 E --> G
 G --> H["Future poses<br/>+ validity"]
 X["Paired Exo video<br/>training only"] -. "feature targets" .-> E`,
6:`flowchart LR
 X["Paired Exo clip"] --> M["Frozen MAE"]
 M --> G["Clip average<br/>global target"]
 M --> L["Future-frame sequence<br/>local target"]
 Q["Ego-conditioned<br/>queries"] --> PG["Predicted<br/>global vector"]
 Q --> PL["Predicted<br/>local sequence"]
 G -. "MSE" .-> PG
 L -. "MSE" .-> PL`,
7:`flowchart TB
 subgraph TRAIN["Training"]
 direction LR
 I["Past Ego RGB<br/>Language<br/>Past poses"] --> T["Forecaster"] --> P["Future poses"]
 X["Paired Exo video<br/>including future frames"] -. "feature supervision" .-> T
 end
 subgraph TEST["Inference"]
 direction LR
 J["Past Ego RGB<br/>Language<br/>Past poses"] --> U["Forecaster"] --> Q["Future poses"]
 end`,
8:`flowchart LR
 P["Pose queries"] --> G["Global modulation<br/>AdaLN + cross-attention"]
 V["Predicted<br/>clip vector"] --> G
 G --> L["Local modulation<br/>alignment + gate + AdaLN"]
 F["Predicted<br/>frame sequence"] --> L
 L --> H["Pose + validity<br/>head"]`
};
let settings={};let slides=[];
for(const d of data){const content=[];let n=0;const diagram=diagrams[d.index];
 for(const e of d.el){
  if(diagram && e.y>165&&e.y<655)continue;
  const id=`s${d.index}-o${n++}`;
  const fixed=(e.y<170||e.y>=655||e.type==='box'||d.index===0);
  let step=0;if(!fixed){step=Math.max(0,Math.min(2,Math.floor((e.y-190)/150)));if(d.index===2)step=e.x<410?0:e.x<820?1:2;}
  const anim='none';settings[id]={x:e.x,y:e.y,w:e.w,h:e.h,font:e.size||24,animation:anim,step,duration:.45};
  const style=`left:${e.x}px;top:${e.y}px;width:${e.w}px;height:${e.h}px;`;
  const attrs=`data-object="${id}" data-kind="${e.type}" class="object ${!fixed?'fragment fade-up':''}" ${!fixed?`data-fragment-index="${step}"`:''} style="${style}"`;
  if(e.type==='text')content.push(`<div ${attrs} data-edit data-edit-id="${id}" data-color="${e.color}" data-weight="${e.bold?650:400}">${esc(e.v).replaceAll('\n','<br>')}</div>`);
  if(e.type==='box')content.push(`<div ${attrs}><div class="color-plane" style="background:${e.color}"></div></div>`);
  if(e.type==='arrow')content.push(`<div ${attrs}><div class="arrow" style="background:${e.color}"></div></div>`);
  if(e.type==='img')content.push(`<div ${attrs}><img src="assets/${e.name}.png" alt="${esc(e.name)} · Shi et al. (2026), CC BY 4.0" style="object-fit:${e.fit||'contain'}"></div>`);
  if(e.type==='table')content.push(`<div ${attrs}><table><colgroup>${e.widths.map(w=>`<col style="width:${w/e.w*100}%">`).join('')}</colgroup>${e.rows.map((r,j)=>`<tr>${r.map((c,k)=>`<${j?'td':'th'} data-edit data-edit-id="${id}-r${j}-c${k}">${esc(c)}</${j?'td':'th'}>`).join('')}</tr>`).join('')}</table></div>`);
  if(e.type==='chart')content.push(`<div ${attrs}><pre class="chart-source" hidden data-edit-id="${id}-chart">${esc(JSON.stringify(e))}</pre><div class="chart-render"></div></div>`);
 }
 if(diagram){const id=`s${d.index}-diagram`;let y=d.index===7?180:205,h=d.index===7?435:355;settings[id]={x:64,y,w:1152,h,animation:'none',step:0,duration:.45};content.push(`<div class="object diagram" data-object="${id}" data-kind="mermaid" style="left:64px;top:${y}px;width:1152px;height:${h}px"><pre hidden class="diagram-source" data-edit-id="${id}-source">${esc(diagram)}</pre><div class="diagram-render"></div></div>`);
 const caption=d.index===4?'DERM reconstructs Exo context. GLMM refines pose features.':d.index===6?'Ego-conditioned queries learn both targets using MSE.':d.index===7?'Future Exo frames supervise training only.':'AdaLN(x) = (1 + γ) ⊙ LN(x) + β';
 const cid=`s${d.index}-caption`;settings[cid]={x:64,y:606,w:1152,h:50,font:31,animation:'none',step:0,duration:.45};content.push(`<div class="object fragment fade-up" data-object="${cid}" data-kind="text" data-edit data-edit-id="${cid}" data-fragment-index="0" data-color="#2D5263" data-weight="600" style="left:64px;top:606px;width:1152px;height:50px">${esc(caption)}</div>`);
 await fs.writeFile(root+'/'+({4:'architecture',6:'derm',7:'inference',8:'glmm'}[d.index])+'.mmd',diagram+'\n');
 }
 slides.push(`<section data-name="${esc(d.title)}" data-seconds="${d.seconds}">${content.join('\n')}<aside class="notes" data-edit-id="notes-${d.index}"><p><strong>${esc(d.speaker)} · ${d.seconds} seconds</strong></p><p>${esc(d.notes)}</p><p>Source: ${esc(d.source)}. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.</p></aside></section>`);
}
const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exo2EgoPose · HCVISION Research Seminar</title><link rel="stylesheet" href="vendor/dist/reveal.css"><link rel="stylesheet" href="theme.css"></head><body>
<nav class="tools" aria-label="Presentation editor"><button id="edit">Edit · E</button><button id="diagram-edit">Mermaid · M</button><button id="properties">Layout & animation</button><button id="notes-edit">Notes</button><button id="save">HTML backup</button><button id="pdf">PDF view</button><span id="save-status" role="status">Connecting…</span></nav>
<div class="reveal"><div class="slides">${slides.join('\n')}</div></div>
<pre hidden id="deck-config" data-edit-id="deck-config">${esc(JSON.stringify(settings))}</pre>
<aside id="editor-panel" class="editor-panel" hidden><div class="panel-title"><strong id="panel-heading">Object settings</strong><button id="close-panel" aria-label="Close editor">×</button></div><div id="panel-body"></div></aside>
<script src="vendor/dist/reveal.js"></script><script src="vendor/plugin/notes/notes.js"></script><script src="vendor/mermaid.min.js"></script><script src="deck.js"></script></body></html>`;
await fs.writeFile(root+'/index.html',html);
await fs.writeFile(root+'/Exo2EgoPose-animated.html','<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=index.html"><title>Exo2EgoPose</title></head><body><a href="index.html">Open the editable presentation</a></body></html>');
console.log(JSON.stringify({slides:data.length,objects:Object.keys(settings).length,mermaidDiagrams:Object.keys(diagrams).length}));
