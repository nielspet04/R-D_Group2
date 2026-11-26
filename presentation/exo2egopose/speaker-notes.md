# Exo2EgoPose rehearsal notes

Main talk: 18.583333333333332 minutes. Slides 19–24 are backup only. Niels presents slides 1–8; Emiel presents slides 9–18.

Browser controls: arrow keys to navigate, F fullscreen, N speaker notes, B backup, H home. Test fullscreen on the presentation laptop. Submit the PDF to Toledo before the session. Attendance is mandatory. The reviewing team leads the 20–25 minute discussion. The seminar email specifies live Poll Everywhere Q&A.

## 1. Exo2EgoPose

Niels · 35 seconds

Open with the question: can a third-person demonstration teach a first-person model what its hands will do next? Introduce Emiel and yourself. This is Shi et al., ACM Multimedia 2026, not our own method. Preview task, mechanism, evidence and critique.

Source: p.1

## 2. What the Ego camera misses

Niels · 55 seconds

Ego cameras see what is in front of the wearer, so hands and objects may be truncated. Head motion changes the background. Exo demonstrations offer complementary context. Stable and holistic is the paper’s motivation, not a guarantee for every capture. Use the real paired images to explain this.

Source: Fig.1, pp.1–2

## 3. The forecasting task

Niels · 65 seconds

Inputs are past Ego RGB, a language instruction and past 3D hand poses. Output is ten future frames of 42 3D joints and joint validity. Coordinates are in the Ego camera system. Forecasting is different from estimating the current pose. Ten frames correspond to about 0.67 seconds at 15 fps and 0.33 seconds at 30 fps. Past pose availability is a deployment assumption.

Source: §3.1, §4.1.3

## 4. Where the paper fits

Niels · 65 seconds

USST forecasts coarse 3D hand trajectories; OCT and MADiff are other motion forecasting precedents. MCIL and HULC condition robot policies on language. GR-1 learns through human video pretraining. AR-VRM uses hand-keypoint pretraining and analogical reasoning. Exo2EgoPose combines joint-level forecasting with training-time Exo feature supervision. Baselines here are adapted, not verbatim published results.

Source: §2; Appendix A

## 5. The forecasting pipeline

Niels · 70 seconds

Walk left to right. Encoders produce tokens, the causal transformer decodes pose and Exo queries, DERM learns two Exo representations, and GLMM refines pose features. Actual Exo video supplies feature targets during training only. At test time those features are predicted from the available Ego inputs. Reconstruction refers to feature vectors, not photorealistic Exo video.

Source: Fig.2; §3

## 6. Multimodal tokens and queries

Niels · 60 seconds

CLIP text, frozen MAE and DINOv2 visual encoders, and trainable HandFormer for poses. Visual tokens are resampled from 196 to 9 per encoder. Tokens enter a 12-layer causal transformer with hidden size 384 and time embeddings. Distinct query types request pose, global Exo, future frame Exo and an auxiliary future Ego image. Avoid overloading the audience with every dimension.

Source: §3.2–3.3; §4.1.3

## 7. DERM learns two Exo targets

Niels · 85 seconds

Video-level target averages MAE CLS features across the aligned past-plus-future window. Frame-level targets are individual MAE CLS features for the future chunk. Decoded Ego-conditioned queries regress these targets with MSE. The global vector summarizes motion across the clip, while the frame sequence preserves local spatial detail. For asynchronous EgoMe videos, linear relative-time alignment is used. This alignment can be imperfect.

Source: §3.3, Eqs.10–15

## 8. Training and inference use different information

Niels · 55 seconds

Explicitly answer the potential leakage question. Future Exo frames are privileged supervision during training; they are not passed to the deployed forecaster. Test inputs remain language, past Ego RGB and past hand states. This resembles learning with privileged information. The targets may nevertheless encourage reliance on temporal alignment or dataset regularities, motivating controls. Hand off: Emiel will explain how the predicted Exo features improve pose predictions.

Source: §3.1–3.4

## 9. GLMM refines poses globally, then locally

Emiel · 85 seconds

Global modulation derives scale, shift and residual weights from one reconstructed clip vector, plus cross-attention. Local modulation cross-attends over predicted frame features, then uses a learned gate and adaptive normalization. Soft alignment accommodates imperfect temporal correspondence. The normalized pose features become (1+gamma) times LN(x) plus beta. Do not say that attention alone proves geometric understanding.

Source: Fig.3; §3.4, Eqs.16–22

## 10. The training objective

Emiel · 50 seconds

Pose coordinates use Smooth L1. Joint validity uses BCE for missing or irrelevant hands and joints. Exo targets use MSE, and an auxiliary MAE-style decoder reconstructs the third future Ego image. The pose loss weight is 3000. Raw loss scales therefore should not be compared without the weights. All image reconstruction branches serve training.

Source: Eqs.15,23; §4.1.3

## 11. Evaluation across three benchmarks

Emiel · 65 seconds

AssemblyHands is toy assembly with paired synchronized views. Ego-Exo4D is broader skilled activity with mixed manual and automatic pose annotations. EgoMe-pose is newly pseudo-labeled with InterHand and RootNet and filtered. Metrics: MPJPE is root-aligned joint position error in mm; MPJVE compares joint displacements and is reported in mm. Different fps and label protocols mean absolute errors are not directly comparable across datasets.

Source: §4.1; Appendix B

## 12. Position error falls on every benchmark

Emiel · 75 seconds

Read each within-dataset comparison. AR-VRM versus ours gives 33.39 to 25.83, 44.46 to 36.44, and 56.06 to 49.44 mm. Relative reductions are 22.6, 18.0 and 11.8 percent. GR-1 is slightly better than AR-VRM in Ego-Exo4D MPJPE at 44.29, so against the best baseline that reduction is 17.7 percent. These are test results, not the validation ablation numbers. No uncertainty intervals are reported.

Source: Table 1; percentages calculated from Table 1

## 13. What the qualitative examples show

Emiel · 55 seconds

Ask the audience to look at right-hand orientation and the grasping shape relative to GT. These are selected examples from Figure 4. Predicted 3D joints were projected to 2D and linearly scaled and aligned with stride 2. They help show hand shape, but are not independent evidence of accurate absolute 3D placement or a failure-rate estimate.

Source: Fig.4; §4.4.1

## 14. What each component adds

Emiel · 70 seconds

Use validation values only. Full model 25.37. Keeping Exo reconstruction but removing both modulation modules gives 30.11. Removing all proposed modules gives 39.70. Replacing both modulation modules by vanilla transformer layers yields 26.44. Ablation supports the architecture, but the 39.70 to 25.37 difference bundles several changes. The prose mentions a 9.59 degradation relative to the 30.11 row; the full-versus-baseline difference is 14.33. Prefer explicit table comparisons.

Source: Table 2, AssemblyHands validation

## 15. Human-to-robot transfer

Emiel · 60 seconds

A pretrained human forecasting model teaches a robot policy through intermediate-feature distillation. The CALVIN ABC-to-D setting tests an unseen scene after robot-policy training on A/B/C. Average success rises from 65.9 to 67.5 percent, a 1.6 percentage point gain; average task-chain length rises from 3.29 to 3.38. Single-task success is lower than AR-VRM: 0.859 versus 0.901. This is not direct zero-shot control of a physical robot by human hand poses.

Source: §4.4.5; Table 4

## 16. What the evidence supports

Emiel · 75 seconds

Strength: Exo is training-only, both module families have ablation support, and evaluation spans three benchmarks plus robot transfer. Limits: automatic labels and filtered easy-visible cases can bias the benchmark; reimplementations and no reported run variance limit comparison confidence; MPJPE is root-aligned so it does not establish global reaching accuracy; language necessity is not isolated; 802 GFLOPs baseline means low extra overhead does not establish real-time operation. State these as our analysis.

Source: Our analysis based on §4, Tables 1–4 and Appendix B

## 17. The next experiments

Emiel · 50 seconds

Prioritize shuffled Exo pair and time-offset controls to isolate semantic matching from generic regularization. Replace ground-truth past poses with estimated/noisy inputs to measure deployment robustness. Compare correct, shuffled and omitted instructions to test language use. Run multiple seeds, subject/action-separated splits, and hand-audited labels. These are proposed follow-up studies, not experiments reported in the paper.

Source: Proposed experiments by the presenters

## 18. Takeaways and discussion

Emiel · 40 seconds

Three messages: forecast detailed future hands from multimodal Ego inputs, learn Exo features at two temporal resolutions during training, and use those predicted features to refine poses. Evidence supports within-benchmark gains; deployment and causal benefit of paired supervision need more testing. End the talk and invite the reviewing team. Let reviewers introduce their own two anchor questions and moderate the audience board.

Source: Summary of the paper and our assessment

## 19. Backup: full benchmark results

Either · 0 seconds

Table 1 in full, excluding random forecasting only. MPJPE / MPJVE reported in mm. Clarify that all methods were reimplemented with common backbones. AR-VRM additionally receives Exo data for analogical reasoning.

Source: Table 1; §4.2; Appendix A.6

## 20. Backup: dataset construction

Either · 0 seconds

AssemblyHands keeps frames with valid root and more than 10 valid joints per hand. Ego-Exo4D uses undistorted frames, standardized joint indices and atomic-action windows from -0.5 to +1.0 seconds. For both datasets, half of original validation episodes become the processed test split. EgoMe-pose uses confidence and episode-validity filtering. No new manual annotation audit is reported.

Source: §4.1.1; Appendix B

## 21. Backup: efficiency and deployment

Either · 0 seconds

Sum Table 3: baseline 802.100 G FLOPs plus 0.676 G from new components equals 802.776 G, roughly 0.084 percent overhead. Parameters: 236.678 plus 4.435 equals 241.113 million, about 1.87 percent overhead. The paper reports no measured FPS or target-device latency. Past hand states must come from some estimator in deployment, which is not evaluated here.

Source: Table 3; calculated sums and percentages

## 22. Backup: likely questions

Either · 0 seconds

Q: Is future information leaked? A: Future Exo supplies training targets only; test inference uses past Ego inputs. Q: Why not predict Exo pixels? A: Feature reconstruction transfers context with a smaller target, but can inherit encoder bias. Q: Is language useful? A: The task is conditioned on language, but a dedicated language ablation is missing. Q: Does it solve robotics? A: CALVIN feature distillation shows a modest transfer improvement, not end-to-end dexterous real-world deployment.

Source: §3–4; our assessment

## 23. Backup: original method diagram

Either · 0 seconds

Original architecture from Figure 2. Use this slide to answer questions about separate query types, the frozen Exo MAE targets, causal tokenization, the pose-query path into GLMM, and auxiliary Ego-image reconstruction. The two discussion prompts are retained in the separate speaking notes.

Source: Fig.2; §3

## 24. References and figure credits

Either · 0 seconds

Primary source is the supplied 15-page Exo2EgoPose paper, arXiv:2607.15890v2, 5 August 2026. Venue is ACM MM 2026 as shown in the paper and seminar document. Figures 1, 2 and 4 reproduced or cropped under the paper’s CC BY 4.0 license, credited to Shi et al. Related-work descriptions come from the paper’s related work and Appendix A. Seminar requirements from HCVISION Research Seminar, updated 23 October 2026.

Source: Primary sources and attribution
