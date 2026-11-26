import fs from 'node:fs/promises';
export const root='/Users/nielspeters/Library/CloudStorage/OneDrive-KULeuven/MASTER 2/R&D project/presentation/exo2egopose';
const C={bg:'#FCFCFA',white:'#171B1E',muted:'#72787C',teal:'#2D5263',blue:'#2D5263',amber:'#72787C',panel:'#F0F2F2'};
export const meta=[
['Exo2EgoPose','Niels',35,'Open with the question: can a third-person demonstration teach a first-person model what its hands will do next? Introduce Emiel and yourself. This is Shi et al., ACM Multimedia 2026, not our own method. Preview task, mechanism, evidence and critique.','p.1'],
['What the Ego camera misses','Niels',55,'Ego cameras see what is in front of the wearer, so hands and objects may be truncated. Head motion changes the background. Exo demonstrations offer complementary context. Stable and holistic is the paper’s motivation, not a guarantee for every capture. Use the real paired images to explain this.','Fig.1, pp.1–2'],
['The forecasting task','Niels',65,'Inputs are past Ego RGB, a language instruction and past 3D hand poses. Output is ten future frames of 42 3D joints and joint validity. Coordinates are in the Ego camera system. Forecasting is different from estimating the current pose. Ten frames correspond to about 0.67 seconds at 15 fps and 0.33 seconds at 30 fps. Past pose availability is a deployment assumption.','§3.1, §4.1.3'],
['Where the paper fits','Niels',65,'USST forecasts coarse 3D hand trajectories; OCT and MADiff are other motion forecasting precedents. MCIL and HULC condition robot policies on language. GR-1 learns through human video pretraining. AR-VRM uses hand-keypoint pretraining and analogical reasoning. Exo2EgoPose combines joint-level forecasting with training-time Exo feature supervision. Baselines here are adapted, not verbatim published results.','§2; Appendix A'],
['The forecasting pipeline','Niels',70,'Walk left to right. Encoders produce tokens, the causal transformer decodes pose and Exo queries, DERM learns two Exo representations, and GLMM refines pose features. Actual Exo video supplies feature targets during training only. At test time those features are predicted from the available Ego inputs. Reconstruction refers to feature vectors, not photorealistic Exo video.','Fig.2; §3'],
['Multimodal tokens and queries','Niels',60,'CLIP text, frozen MAE and DINOv2 visual encoders, and trainable HandFormer for poses. Visual tokens are resampled from 196 to 9 per encoder. Tokens enter a 12-layer causal transformer with hidden size 384 and time embeddings. Distinct query types request pose, global Exo, future frame Exo and an auxiliary future Ego image. Avoid overloading the audience with every dimension.','§3.2–3.3; §4.1.3'],
['DERM learns two Exo targets','Niels',85,'Video-level target averages MAE CLS features across the aligned past-plus-future window. Frame-level targets are individual MAE CLS features for the future chunk. Decoded Ego-conditioned queries regress these targets with MSE. The global vector summarizes motion across the clip, while the frame sequence preserves local spatial detail. For asynchronous EgoMe videos, linear relative-time alignment is used. This alignment can be imperfect.','§3.3, Eqs.10–15'],
['Training and inference use different information','Niels',55,'Explicitly answer the potential leakage question. Future Exo frames are privileged supervision during training; they are not passed to the deployed forecaster. Test inputs remain language, past Ego RGB and past hand states. This resembles learning with privileged information. The targets may nevertheless encourage reliance on temporal alignment or dataset regularities, motivating controls. Hand off: Emiel will explain how the predicted Exo features improve pose predictions.','§3.1–3.4'],
['GLMM refines poses globally, then locally','Emiel',85,'Global modulation derives scale, shift and residual weights from one reconstructed clip vector, plus cross-attention. Local modulation cross-attends over predicted frame features, then uses a learned gate and adaptive normalization. Soft alignment accommodates imperfect temporal correspondence. The normalized pose features become (1+gamma) times LN(x) plus beta. Do not say that attention alone proves geometric understanding.','Fig.3; §3.4, Eqs.16–22'],
['The training objective','Emiel',50,'Pose coordinates use Smooth L1. Joint validity uses BCE for missing or irrelevant hands and joints. Exo targets use MSE, and an auxiliary MAE-style decoder reconstructs the third future Ego image. The pose loss weight is 3000. Raw loss scales therefore should not be compared without the weights. All image reconstruction branches serve training.','Eqs.15,23; §4.1.3'],
['Evaluation across three benchmarks','Emiel',65,'AssemblyHands is toy assembly with paired synchronized views. Ego-Exo4D is broader skilled activity with mixed manual and automatic pose annotations. EgoMe-pose is newly pseudo-labeled with InterHand and RootNet and filtered. Metrics: MPJPE is root-aligned joint position error in mm; MPJVE compares joint displacements and is reported in mm. Different fps and label protocols mean absolute errors are not directly comparable across datasets.','§4.1; Appendix B'],
['Position error falls on every benchmark','Emiel',75,'Read each within-dataset comparison. AR-VRM versus ours gives 33.39 to 25.83, 44.46 to 36.44, and 56.06 to 49.44 mm. Relative reductions are 22.6, 18.0 and 11.8 percent. GR-1 is slightly better than AR-VRM in Ego-Exo4D MPJPE at 44.29, so against the best baseline that reduction is 17.7 percent. These are test results, not the validation ablation numbers. No uncertainty intervals are reported.','Table 1; percentages calculated from Table 1'],
['What the qualitative examples show','Emiel',55,'Ask the audience to look at right-hand orientation and the grasping shape relative to GT. These are selected examples from Figure 4. Predicted 3D joints were projected to 2D and linearly scaled and aligned with stride 2. They help show hand shape, but are not independent evidence of accurate absolute 3D placement or a failure-rate estimate.','Fig.4; §4.4.1'],
['What each component adds','Emiel',70,'Use validation values only. Full model 25.37. Keeping Exo reconstruction but removing both modulation modules gives 30.11. Removing all proposed modules gives 39.70. Replacing both modulation modules by vanilla transformer layers yields 26.44. Ablation supports the architecture, but the 39.70 to 25.37 difference bundles several changes. The prose mentions a 9.59 degradation relative to the 30.11 row; the full-versus-baseline difference is 14.33. Prefer explicit table comparisons.','Table 2, AssemblyHands validation'],
['Human-to-robot transfer','Emiel',60,'A pretrained human forecasting model teaches a robot policy through intermediate-feature distillation. The CALVIN ABC-to-D setting tests an unseen scene after robot-policy training on A/B/C. Average success rises from 65.9 to 67.5 percent, a 1.6 percentage point gain; average task-chain length rises from 3.29 to 3.38. Single-task success is lower than AR-VRM: 0.859 versus 0.901. This is not direct zero-shot control of a physical robot by human hand poses.','§4.4.5; Table 4'],
['What the evidence supports','Emiel',75,'Strength: Exo is training-only, both module families have ablation support, and evaluation spans three benchmarks plus robot transfer. Limits: automatic labels and filtered easy-visible cases can bias the benchmark; reimplementations and no reported run variance limit comparison confidence; MPJPE is root-aligned so it does not establish global reaching accuracy; language necessity is not isolated; 802 GFLOPs baseline means low extra overhead does not establish real-time operation. State these as our analysis.','Our analysis based on §4, Tables 1–4 and Appendix B'],
['The next experiments','Emiel',50,'Prioritize shuffled Exo pair and time-offset controls to isolate semantic matching from generic regularization. Replace ground-truth past poses with estimated/noisy inputs to measure deployment robustness. Compare correct, shuffled and omitted instructions to test language use. Run multiple seeds, subject/action-separated splits, and hand-audited labels. These are proposed follow-up studies, not experiments reported in the paper.','Proposed experiments by the presenters'],
['Takeaways and discussion','Emiel',40,'Three messages: forecast detailed future hands from multimodal Ego inputs, learn Exo features at two temporal resolutions during training, and use those predicted features to refine poses. Evidence supports within-benchmark gains; deployment and causal benefit of paired supervision need more testing. End the talk and invite the reviewing team. Let reviewers introduce their own two anchor questions and moderate the audience board.','Summary of the paper and our assessment'],
['Backup: full benchmark results','Either',0,'Table 1 in full, excluding random forecasting only. MPJPE / MPJVE reported in mm. Clarify that all methods were reimplemented with common backbones. AR-VRM additionally receives Exo data for analogical reasoning.','Table 1; §4.2; Appendix A.6'],
['Backup: dataset construction','Either',0,'AssemblyHands keeps frames with valid root and more than 10 valid joints per hand. Ego-Exo4D uses undistorted frames, standardized joint indices and atomic-action windows from -0.5 to +1.0 seconds. For both datasets, half of original validation episodes become the processed test split. EgoMe-pose uses confidence and episode-validity filtering. No new manual annotation audit is reported.','§4.1.1; Appendix B'],
['Backup: efficiency and deployment','Either',0,'Sum Table 3: baseline 802.100 G FLOPs plus 0.676 G from new components equals 802.776 G, roughly 0.084 percent overhead. Parameters: 236.678 plus 4.435 equals 241.113 million, about 1.87 percent overhead. The paper reports no measured FPS or target-device latency. Past hand states must come from some estimator in deployment, which is not evaluated here.','Table 3; calculated sums and percentages'],
['Backup: likely questions','Either',0,'Q: Is future information leaked? A: Future Exo supplies training targets only; test inference uses past Ego inputs. Q: Why not predict Exo pixels? A: Feature reconstruction transfers context with a smaller target, but can inherit encoder bias. Q: Is language useful? A: The task is conditioned on language, but a dedicated language ablation is missing. Q: Does it solve robotics? A: CALVIN feature distillation shows a modest transfer improvement, not end-to-end dexterous real-world deployment.','§3–4; our assessment'],
['Backup: two discussion prompts','Either',0,'These prompts are optional preparation for discussion, not a replacement for the reviewing team’s own anchor questions. Ask: would correctly paired Exo still beat shuffled Exo under matched compute and multiple seeds? Ask: how would the result change under estimated past poses and manually audited, action-separated test data?','Presenters’ proposed discussion prompts'],
['References and figure credits','Either',0,'Primary source is the supplied 15-page Exo2EgoPose paper, arXiv:2607.15890v2, 5 August 2026. Venue is ACM MM 2026 as shown in the paper and seminar document. Figures 1, 2 and 4 reproduced or cropped under the paper’s CC BY 4.0 license, credited to Shi et al. Related-work descriptions come from the paper’s related work and Appendix A. Seminar requirements from HCVISION Research Seminar, updated 23 October 2026.','Primary sources and attribution']
];
export async function build({presentation,input}){
 const all=[]; const src='Shi et al., Exo2EgoPose (2026)';
 for(let i=input.start;i<input.end;i++){
  let s=presentation.slides.add({width:1280,height:720});
  s.background.fill=C.bg;const el=[];
  const text=(v,x,y,w,h,size=28,color=C.white,bold=false)=>{const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});z.text=v;z.text.style={fontFamily:'Helvetica Neue',fontSize:size,color,bold};z.text.insets=0;el.push({type:'text',v,x,y,w,h,size,color,bold});return z;};
  const box=(v,x,y,w,h,color=C.panel)=>{s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:color,line:{fill:'none',width:0}});el.push({type:'box',x,y,w,h,color});if(v)text(v,x+18,y+18,w-36,h-24,25);};
  const arrow=(x,y,w=45,color=C.teal)=>{s.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:22},fill:color,line:{fill:'none',width:0}});el.push({type:'arrow',x,y,w,h:22,color});};
  const img=async(name,x,y,w,h)=>{const b=await fs.readFile(root+'/assets/'+name+'.png');s.images.add({path:root+'/assets/'+name+'.png',alt:name+' reproduced from Exo2EgoPose, Shi et al. 2026',fit:name==='extracted-2-16'?'cover':'contain',position:{left:x,top:y,width:w,height:h}});el.push({type:'img',name,x,y,w,h,fit:name==='extracted-2-16'?'cover':'contain'});};
  const line=(y)=>box('',64,y,1152,2,'#D4DADD');
  const list=(rows,x=68,y=200,w=1100,gap=100)=>rows.forEach(([a,b],j)=>{text(a,x,y+j*gap,w,38,29,C.teal,true);text(b,x,y+j*gap+43,w,70,25,C.white);});
  const table=(rows,x,y,w,h,widths)=>{const t=s.tables.add({rows:rows.length,columns:rows[0].length,left:x,top:y,width:w,height:h,values:rows,columnWidths:widths});for(let r=0;r<rows.length;r++)for(let c=0;c<rows[0].length;c++){let cell=t.getCell(r,c);cell.fill=r===0?'#EDF0F1':C.bg;cell.text.style={fontSize:23,color:r===0?C.muted:C.white,bold:r===0||r===rows.length-1};}el.push({type:'table',rows,x,y,w,h,widths});};
  if(i>0)text(meta[i][0],64,57,1150,110,meta[i][0].length>44?44:50,C.white,true);
  text(i<18?String(i+1).padStart(2,'0'):'B'+(i-17),1168,668,65,28,18,C.muted);
  if(i>0)text((i>=18?'Backup · ':'')+meta[i][4],64,670,1060,25,15,C.muted);
  switch(i){
  case 0:

   await img('extracted-2-16',660,0,620,720);
   text('Exo2\nEgoPose',64,145,590,245,108,C.white,true);
   text('Forecasting 3D hand poses\nwith guidance from another view',68,436,550,110,32,C.muted);
   text('Niels Peters & Emiel Digneffe',68,601,570,38,25,C.white,true);
   text('HCVISION · Shi et al., ACM MM 2026',68,650,570,30,20,C.muted);
  break;
  case 1:

   await img('extracted-2-0',64,205,540,330);await img('extracted-2-19',676,205,540,330);
   text('Ego',64,550,540,45,34,C.white,true);text('Exo',676,550,540,45,34,C.white,true);
   text('Close to the action. Partial and moving.',64,603,540,60,25,C.muted);
   text('A wider view of the same activity.',676,603,540,60,25,C.muted);
  break;
  case 2:

   text('Observe',64,220,340,46,31,C.muted);text('Condition',472,220,340,46,31,C.muted);text('Forecast',880,220,340,46,31,C.muted);
   text('Past Ego RGB\nand 3D poses',64,300,350,125,46,C.white,true);
   text('A language\ninstruction',472,300,350,125,46,C.white,true);
   text('10 future\nhand poses',880,300,350,125,46,C.teal,true);
   arrow(413,340,35);arrow(821,340,35);
   box('',64,492,1152,1,'#D9DDDF');
   text('42 joints',64,535,360,63,48,C.white,true);text('3D + validity',472,535,360,63,48,C.white,true);text('0.33–0.67 s',880,535,360,63,48,C.white,true);
   text('Two hands, 21 joints each',64,610,360,32,22,C.muted);text('Ego camera coordinates',472,610,360,32,22,C.muted);text('10 frames at 30 / 15 fps',880,610,360,32,22,C.muted);
  break;
  case 3:
   table([['Research line','Representative methods','Remaining gap'],['Hand motion forecasting','OCT · USST · MADiff','Mostly coarse motion'],['Language-conditioned control','MCIL · HULC','Robot data and embodiment'],['Human-to-robot pretraining','GR-1 · AR-VRM','Limited Ego context'],['Exo2EgoPose','Joint-level multimodal forecasting','Exo supervision at two levels']],68,202,1140,350,[320,420,400]);
   text('Contribution: learn Exo context, then use it to refine Ego pose features',68,590,1120,60,30,C.teal,true);break;
  case 4:

   text('Learn the context',64,198,1140,45,30,C.muted);
   box('Multimodal\nEgo inputs',64,278,240,118);arrow(323,325,40);box('Causal\ntransformer',384,278,240,118);arrow(644,325,40);box('Predicted\nExo features',704,278,240,118);arrow(963,325,40);box('GLMM +\npose head',1024,278,192,118);
   box('',704,460,512,1,'#D9DDDF');text('Training supervision',704,490,500,39,25,C.muted);text('Paired Exo video supplies\nglobal and local feature targets',704,545,500,88,31,C.white,true);
   text('Predict first.\nRefine second.',64,481,570,150,50,C.white,true);
  break;
  case 5:
   list([['Encode each modality','CLIP text · MAE + DINOv2 images · HandFormer poses'],['Fuse through causal attention','12 transformer layers · hidden size 384 · time embeddings'],['Decode distinct query types','Future poses · global Exo · frame-level Exo · auxiliary Ego image']],68,205,1120,137);break;
  case 6:

   text('One clip, two levels of supervision',64,194,1130,50,32,C.muted);
   box('Paired Exo clip',64,297,275,120);arrow(359,346,45);box('Frozen MAE',424,297,245,120);arrow(689,346,45);
   text('Global',780,250,430,44,36,C.white,true);text('Average features across the clip',780,309,430,62,26,C.muted);
   text('Local',780,420,430,44,36,C.white,true);text('Keep the future-frame sequence',780,479,430,62,26,C.muted);
   text('Ego-conditioned queries learn to predict both targets',64,600,1130,54,34,C.teal,true);
  break;
  case 7:

   text('Training',64,214,520,58,44,C.white,true);text('Inference',704,214,510,58,44,C.white,true);
   box('',640,214,1,375,'#D9DDDF');
   text('Past Ego RGB\nLanguage\nPast hand poses',64,313,510,160,35,C.white);
   text('Past Ego RGB\nLanguage\nPast hand poses',704,313,510,160,35,C.white);
   text('+ paired Exo feature targets',64,525,520,59,29,C.teal,true);
   text('Exo features predicted internally',704,525,510,80,29,C.teal,true);
   text('Future Exo frames supervise training only',64,625,1130,37,27,C.muted);
  break;
  case 8:

   text('Global',64,213,510,60,46,C.white,true);text('Local',704,213,510,60,46,C.white,true);
   text('Reconstructed clip vector',64,294,510,45,29,C.muted);text('Reconstructed frame sequence',704,294,510,45,29,C.muted);
   box('Adaptive normalization\n+ cross-attention',64,381,510,115);arrow(603,426,66);box('Soft alignment\n+ gated refinement',704,381,510,115);
   text('AdaLN(x) = (1 + γ) ⊙ LN(x) + β',64,567,1130,63,44,C.teal,true);
  break;
  case 9:
   text('L = 3000 Lpose + Lvalid + LV + 0.1 LF + LE',68,213,1140,95,42,C.teal,true);
   table([['Target','Loss','Role'],['Future 3D joints','Smooth L1','Forecast pose coordinates'],['Joint validity','Binary cross-entropy','Handle missing / inactive joints'],['Global + local Exo features','Mean squared error','Transfer cross-view context'],['Third future Ego image','Reconstruction loss','Auxiliary visual learning']],68,338,1140,283,[400,310,430]);break;
  case 10:
   table([['Benchmark','Train / val / test episodes','Pose labels'],['AssemblyHands','6,120 / 355 / 355','Existing 3D annotations'],['Ego-Exo4D','10,869 / 1,540 / 1,540','Manual + automatic'],['EgoMe-pose','6,067 / 1,344 / 2,648','Automatic + filtered']],68,199,1140,262,[350,430,360]);
   text('MPJPE: root-aligned position error',68,507,1100,48,30,C.teal,true);
   text('MPJVE: joint displacement error between frames',68,570,1100,48,30,C.blue,true);
   text('Lower is better · compare methods within each dataset',68,625,1100,32,22,C.muted);break;
  case 11:

   const categories=['AssemblyHands','Ego-Exo4D','EgoMe-pose'];
   const series=[{name:'AR-VRM',values:[33.39,44.46,56.06],fill:'#B7C3C9'},{name:'Exo2EgoPose',values:[25.83,36.44,49.44],fill:C.teal}];
   s.charts.add('bar',{position:{left:64,top:205,width:1152,height:355},categories,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:160},hasLegend:true,legend:{position:'top',textStyle:{fontSize:21,fill:C.white}},xAxis:{textStyle:{fontSize:22,fill:C.white},majorGridlines:null},yAxis:{visible:false,minimumScale:0,maximumScale:65,majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:22,fill:C.white}},chartFill:C.bg,plotAreaFill:C.bg});
   el.push({type:'chart',x:64,y:205,w:1152,h:355,categories,series});
   text('22.6%',94,579,320,57,43,C.teal,true);text('18.0%',494,579,320,57,43,C.teal,true);text('11.8%',894,579,320,57,43,C.teal,true);
   text('MPJPE reduction vs AR-VRM · test sets · errors in mm',64,639,1140,30,21,C.muted);
  break;
  case 12:

   await img('zipper-results',64,211,550,350);await img('watch-results',666,211,550,350);
   text('Pulling a zipper',64,580,550,45,33,C.white,true);text('Picking up a watch',666,580,550,45,33,C.white,true);
   text('Projected 3D poses, scaled and aligned for visualization',64,635,1130,28,21,C.muted);
  break;
  case 13:

   text('AssemblyHands validation · MPJPE (mm)',64,197,1130,40,27,C.muted);
   const ablationCategories=['No Exo modules','Reconstruction only','Full model'];const ablationSeries=[{name:'MPJPE',values:[39.70,30.11,25.37],fill:C.teal}];
   s.charts.add('bar',{position:{left:64,top:268,width:1152,height:300},categories:ablationCategories,series:ablationSeries,barOptions:{direction:'column',gapWidth:170},hasLegend:false,xAxis:{textStyle:{fontSize:23,fill:C.white},majorGridlines:null},yAxis:{visible:false,minimumScale:0,majorGridlines:null},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:25,fill:C.white}},chartFill:C.bg,plotAreaFill:C.bg});el.push({type:'chart',x:64,y:268,w:1152,h:300,categories:ablationCategories,series:ablationSeries});
   text('−4.74 mm',64,596,400,58,44,C.teal,true);text('Adding GLMM to Exo reconstruction',473,610,740,48,29,C.muted);
  break;
  case 14:

   text('Human forecasting teacher → robot policy distillation',64,197,1130,55,31,C.muted);
   text('65.9%',64,324,495,110,92,C.muted,true);arrow(555,363,94);text('67.5%',719,324,495,110,92,C.teal,true);
   text('AR-VRM',64,459,495,48,30,C.muted);text('Exo2EgoPose',719,459,495,48,30,C.white,true);
   box('',64,549,1152,1,'#D9DDDF');text('+1.6 percentage points',64,586,730,62,43,C.white,true);text('CALVIN ABC → D\nAverage success rate',864,588,350,69,24,C.muted);
  break;
  case 15:

   text('Evidence',64,211,510,55,42,C.white,true);text('Open questions',704,211,510,55,42,C.white,true);
   text('Exo used only during training\n\nGains across three benchmarks\n\nAblations and robot transfer',64,306,510,290,30,C.white);
   text('Bias from automatic labels\n\nNo reported run variance\n\nDependence on past pose inputs\n\nGlobal placement after root alignment',704,306,510,327,29,C.muted);
  break;
  case 16:
   list([['Test whether pairing matters','Correct Exo pairs vs shuffled pairs and temporal offsets'],['Test realistic inputs','Estimated past poses, missing hands and perturbed instructions'],['Test robustness of the result','Multiple seeds, action-separated splits and audited 3D labels']],68,204,1120,139);break;
  case 17:
   text('01',68,216,90,62,43,C.teal,true);text('Forecast detailed future hand poses',190,216,1010,70,37);
   text('02',68,335,90,62,43,C.teal,true);text('Learn Exo context at two temporal levels',190,335,1010,70,37);
   text('03',68,454,90,62,43,C.teal,true);text('Refine Ego pose features globally, then locally',190,454,1010,84,37);
   text('Discussion',68,597,1100,53,39,C.blue,true);break;
  case 18:
   table([['Method','AssemblyHands','Ego-Exo4D','EgoMe-pose'],['USST','52.58 / 19.75','63.07 / 22.99','71.93 / 73.67'],['GCBC','45.38 / 12.13','49.97 / 21.86','63.43 / 71.45'],['MCIL','42.58 / 11.05','47.17 / 17.53','62.81 / 70.48'],['HULC','41.88 / 9.75','46.73 / 17.34','60.81 / 69.83'],['GR-1','40.03 / 8.69','44.29 / 16.44','59.10 / 69.03'],['AR-VRM','33.39 / 7.86','44.46 / 16.33','56.06 / 67.66'],['Exo2EgoPose','25.83 / 6.35','36.44 / 16.06','49.44 / 61.03']],68,185,1140,420,[300,280,280,280]);
   text('Each cell: MPJPE / MPJVE (mm), lower is better',68,625,1130,34,23,C.muted);break;
  case 19:
   list([['AssemblyHands','Filter missing joints · world-to-Ego coordinates · action labels'],['Ego-Exo4D','Undistort · standardize joints · atomic-action windows'],['EgoMe-pose','Hand detection · InterHand + RootNet · confidence filtering']],68,201,1120,130);
   text('Processed AssemblyHands / Ego-Exo4D test sets use half the original val episodes',68,606,1130,55,24,C.amber);break;
  case 20:
   text('802.776 GFLOPs',68,222,1100,80,66,C.teal,true);text('Calculated total inference compute from Table 3',68,313,1100,43,28,C.muted);
   text('+0.084% compute · +1.87% parameters',68,411,1120,65,42,C.blue,true);
   text('Small added overhead does not establish real-time deployment',68,532,1130,82,33,C.white);break;
  case 21:
   list([['Does it require future Exo video at test time?','No. Future Exo features are training targets.'],['Is language necessary?','A dedicated instruction ablation is not reported.'],['Does it directly control a robot from hand poses?','No. Transfer uses intermediate-feature distillation.']],68,198,1120,141);break;
  case 22:
   text('Would correctly paired Exo still help against shuffled Exo,\nwith matched compute and multiple seeds?',68,222,1135,130,37,C.teal,true);
   text('How much remains with estimated past poses\nand manually audited, action-separated test data?',68,427,1135,130,37,C.blue,true);break;
  case 23:
   text('Shi, Qiu, Wang, Li & Li (2026)\nExo2EgoPose: Leveraging Exocentric Demonstrations for\nVision-Language guided Egocentric 3D Hand Pose Forecasting',68,205,1140,141,30,C.white,true);
   text('arXiv:2607.15890v2 · ACM Multimedia 2026\nhttps://arxiv.org/abs/2607.15890\nhttps://github.com/ZhaofengSHI/Exo2EgoPose',68,383,1135,125,26,C.blue);
   text('Images adapted from Figs. 1, 2 and 4 · © authors · CC BY 4.0\nSeminar requirements: HCVISION Research Seminar, updated 23 Oct 2026',68,551,1130,83,23,C.muted);break;
  }
  s.speakerNotes.text=`${meta[i][1]} · ${meta[i][2]} seconds\n\n${meta[i][3]}\n\nSOURCE: ${src}, ${meta[i][4]}. Supplied paper: arXiv:2607.15890v2. Figures © authors, CC BY 4.0.\n${[4,6,7,8].includes(i)?'Diagram source: diagrams.mmd. Native slide objects mirror the Mermaid flow.':''}`;
  all.push({index:i,title:meta[i][0],speaker:meta[i][1],seconds:meta[i][2],notes:meta[i][3],source:meta[i][4],el});
 }
 await fs.writeFile(root+`/batch-${input.start}.json`,JSON.stringify(all,null,2));
 return {slides:presentation.slides.items.length,range:[input.start,input.end]};
}
