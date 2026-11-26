# Exo2EgoPose speaking script

Niels: slides 1–8. Emiel: slides 9–18. Slides 19–24 contain spoken answers for Q&A. Timing labels are rehearsal targets.

## 1. Exo2EgoPose

Niels · 35 seconds

Hi everyone. I'm Niels, and together with Emiel, I'll present Exo2EgoPose, a paper by Shi and colleagues from ACM Multimedia 2026.

The main question is whether a third-person demonstration can help a first-person model predict what someone's hands will do next.

I'll start with the problem and the main idea behind the method. Emiel will then explain how the model refines its predictions, go through the results, and discuss the strengths and limitations.

Source: p.1. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 2. What the Ego camera misses

Niels · 55 seconds

The problem starts with the first-person camera. It moves with the person's head, and it only sees part of the scene. A hand or an object can disappear from the image, even though the action is still happening.

That makes it difficult to understand the full interaction and predict the next movement.

A third-person camera can show more of the person, the objects, and the overall action. That's the extra information the authors want to use.

The idea is to learn from both views during training, so the model can make better predictions when it only has the first-person inputs.

Source: Fig.1, pp.1–2. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 3. The forecasting task

Niels · 65 seconds

The model gets three inputs: a short sequence of first-person images, a language instruction, and the previous 3D hand poses.

For example, the instruction could be “pull up the zipper with your right hand.” The model then predicts the hand poses for the next ten frames.

There are 21 joints per hand, so 42 joints in total. For each joint, it predicts three coordinates and whether the joint is valid.

The forecast covers about two-thirds of a second at 15 frames per second, or one-third of a second at 30 frames per second.

So the task is to predict future hand movements. It also assumes that the previous hand poses are available as an input.

Source: §3.1, §4.1.3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 4. Where the paper fits

Niels · 65 seconds

There are a few research directions that lead to this paper.

Methods such as USST predict future hand trajectories. They mainly describe the overall movement, rather than the detailed position of every finger joint.

Methods such as MCIL and HULC use language to guide robot actions. GR-1 and AR-VRM also learn from human data to help with robot manipulation.

Exo2EgoPose brings these ideas together. It predicts detailed future hand poses from several inputs, and it uses a third-person demonstration as extra supervision during training.

For the experiments, the authors adapt the comparison methods to this task and use common backbones. The numbers are therefore results from those adapted versions.

Source: §2; Appendix A. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 5. The forecasting pipeline

Niels · 70 seconds

This diagram shows the full pipeline.

First, the model encodes the first-person images, the instruction, and the previous hand poses. These features go into a causal transformer.

The transformer produces pose queries and predicted Exo features. The pose queries are the features that will be used to predict the future hand poses.

DERM learns the Exo features at two levels: one representation for the clip, and a sequence of representations for the future frames.

GLMM then uses those predicted features to refine the pose queries before the final prediction.

The third-person video provides the training targets for this process. The model reconstructs feature vectors, rather than generating an actual third-person video.

Source: Fig.2; §3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 6. Multimodal tokens and queries

Niels · 60 seconds

The model uses a separate encoder for each type of input.

The instruction goes through a CLIP text encoder. The images go through MAE and DINOv2, and the previous hand poses go through HandFormer.

The visual features are reduced from 196 tokens to nine per encoder. This makes the sequence smaller before the features are combined.

The combined tokens go into a 12-layer causal transformer. Time embeddings tell the model which observations belong to which moment.

There are also different query types for the future poses, the global Exo representation, the local Exo representations, and an auxiliary future Ego image. These queries let the same transformer produce the different outputs needed during training.

Source: §3.2–3.3; §4.1.3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 7. DERM learns two Exo targets

Niels · 85 seconds

DERM learns two kinds of information from the third-person video.

For the global target, the authors take MAE features from the aligned clip and average them. This gives one vector covering the observed part and the future prediction window.

For the local targets, they keep a separate feature vector for each frame in the future window. This preserves more of the frame-by-frame detail.

The transformer predicts both targets from the first-person inputs. Mean squared error measures the difference between the predicted features and the features extracted from the real third-person video.

For EgoMe, the two views are not synchronized. The authors align them using their relative position within the activity. That gives an approximate match, but it can still introduce alignment errors.

Source: §3.3, Eqs.10–15. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 8. Training and inference use different information

Niels · 55 seconds

This distinction between training and inference is important.

During training, the model can use features from the paired third-person video as targets. Those targets can include future frames.

During inference, the model only receives the previous first-person images, the instruction, and the previous hand poses. It has to predict the Exo features internally.

So it doesn't need a third-person camera or future video at test time. The extra view teaches the model during training.

A remaining question is how much this depends on the correct pairing and timing of the videos. We'll come back to that in our discussion.

Emiel will now explain how the predicted Exo features are used to improve the hand poses.

Source: §3.1–3.4. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 9. GLMM refines poses globally, then locally

Emiel · 85 seconds

Once the model has predicted the Exo features, GLMM uses them in two steps.

First comes global modulation. The predicted clip vector changes the scale and shift of the normalized pose features. Cross-attention also combines the pose queries with this global information.

Then comes local modulation. The pose features attend to the predicted frame-level features, and a learned gate controls how much of that information is added.

This attention provides a soft alignment between the pose queries and the frame features. That is useful when their timing does not match perfectly.

The formula at the bottom is adaptive layer normalization. Gamma changes the scale, and beta changes the shift. The aim is to adjust the pose features using the context learned from the Exo view.

Source: Fig.3; §3.4, Eqs.16–22. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 10. The training objective

Emiel · 50 seconds

The model is trained with several losses.

Smooth L1 measures the error in the predicted joint coordinates. Binary cross-entropy handles joint validity, for example when a hand is missing or not involved in the action.

The global and local Exo features use mean squared error. There is also an auxiliary task that reconstructs the third future first-person image.

The pose loss has a weight of 3,000, while the other terms have different weights. These weights balance the different training objectives.

Together, the losses train the model to predict the future poses and learn useful visual context.

Source: Eqs.15,23; §4.1.3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 11. Evaluation across three benchmarks

Emiel · 65 seconds

The authors evaluate the method on three benchmarks.

AssemblyHands contains toy assembly activities with synchronized first-person and third-person videos.

Ego-Exo4D covers a wider range of skilled activities and includes both manual and automatic pose annotations.

EgoMe-pose is constructed for this paper. The authors estimate the hand poses automatically with InterHand and RootNet, and then filter the annotations.

They report two metrics. MPJPE measures joint position error after root alignment. MPJVE measures the error in joint displacement between frames. Lower is better for both.

The datasets have different frame rates and annotation procedures, so we compare methods within each dataset rather than comparing the absolute errors across datasets.

Source: §4.1; Appendix B. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 12. Position error falls on every benchmark

Emiel · 75 seconds

Exo2EgoPose has the lowest position error on all three benchmarks.

Compared with AR-VRM, the error on AssemblyHands goes from 33.39 to 25.83 millimeters. On Ego-Exo4D, it goes from 44.46 to 36.44. On EgoMe-pose, it goes from 56.06 to 49.44.

That gives the reductions shown below: 22.6 percent, 18 percent, and 11.8 percent.

There is one detail in the comparison: GR-1 is slightly better than AR-VRM on Ego-Exo4D. Against that strongest position-error baseline, the reduction is 17.7 percent.

So the improvement is consistent, but its size differs across datasets. The paper doesn't report confidence intervals, so we can't judge the variation across training runs.

Source: Table 1; percentages calculated from Table 1. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 13. What the qualitative examples show

Emiel · 55 seconds

These examples show what the predictions look like.

On the left, the person is pulling up a zipper with the right hand. On the right, the person is picking up a watch with both hands.

Compared with the other methods, Exo2EgoPose produces hand shapes that look closer to the ground truth in these selected examples. We can compare the orientation of the hand and the shape of the fingers during the action.

The authors project the 3D joints onto the images and scale and align them for visualization. That makes the hand shapes easier to compare, but these images alone don't show how accurate the absolute 3D placement is.

Source: Fig.4; §4.4.1. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 14. What each component adds

Emiel · 70 seconds

The ablation study tests what the different components contribute. These numbers are from the AssemblyHands validation set.

Without the proposed Exo modules, the position error is 39.70 millimeters. Adding the Exo reconstruction reduces it to 30.11.

With both reconstruction and GLMM, the error goes down to 25.37. So adding the designed modulation gives another improvement of 4.74 millimeters.

The authors also replace the modulation modules with ordinary transformer layers. That version reaches 26.44 millimeters, which is slightly worse than the full model.

These results support both reconstruction and modulation. The overall difference between the baseline and the full model combines several changes, so we shouldn't attribute that entire gain to one component.

Source: Table 2, AssemblyHands validation. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 15. Human-to-robot transfer

Emiel · 60 seconds

The authors also test whether the learned representation helps a robot policy.

They use the pretrained hand-forecasting model as a teacher and transfer information to the robot policy through intermediate features.

The experiment uses CALVIN. The robot policy is trained on scenes A, B, and C, and evaluated on the unseen scene D.

The average success rate increases from 65.9 to 67.5 percent. That's an improvement of 1.6 percentage points. The average number of tasks completed in a row also increases, from 3.29 to 3.38.

The result is more mixed for individual tasks: single-task success is lower than AR-VRM. So this supports a modest transfer benefit, rather than a general improvement on every measure.

Source: §4.4.5; Table 4. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 16. What the evidence supports

Emiel · 75 seconds

For us, the main strength is that the model benefits from Exo supervision without requiring an Exo video during inference. The gains also appear on three benchmarks, and the ablations support the main components.

There are still several limitations.

Automatic pose labels and confidence filtering can bias which examples are included. The paper also doesn't report variation across training runs.

The model assumes that previous hand poses are available. We don't know how much the results would change if those poses came from a noisy estimator.

Finally, root-aligned position error doesn't tell us whether the hand is reaching the correct absolute position. These are the main gaps we see between the reported benchmark results and a practical system.

Source: Our analysis based on §4, Tables 1–4 and Appendix B. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 17. The next experiments

Emiel · 50 seconds

We would start with three follow-up experiments.

First, compare correctly paired Exo videos with shuffled pairs and videos shifted in time. That would test whether the gain comes from the matching demonstration, rather than just an extra training objective.

Second, use estimated past poses instead of clean pose inputs. We would also test missing hands and incorrect or missing instructions.

Third, repeat the experiments with multiple seeds, action-separated splits, and a manually checked set of pose labels.

These are experiments we would propose. They would help show how reliable the improvement is and which information the model actually uses.

Source: Proposed experiments by the presenters. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 18. Takeaways and discussion

Emiel · 40 seconds

To conclude, there are three points we want to leave you with.

The task is to predict detailed future hand poses from first-person images, language, and previous poses.

The main idea is to learn third-person context at two levels during training, and then use those predicted features to refine the hand poses.

The results show consistent improvements within the evaluated benchmarks. For us, the next step is to test how much those gains depend on paired supervision and reliable pose inputs.

Thank you for listening. We're ready for the discussion.

Source: Summary of the paper and our assessment. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 19. Backup: full benchmark results

Either · 0 seconds

This table contains the full position and motion errors for the comparison methods. Each cell shows MPJPE followed by MPJVE, and lower is better for both.

The authors reimplement the methods for this forecasting task with common backbones. They also extend AR-VRM with Exo data for analogical reasoning.

Exo2EgoPose has the lowest reported error for both metrics on all three benchmarks. The size of the improvement varies, especially for motion error on Ego-Exo4D.

Source: Table 1; §4.2; Appendix A.6. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 20. Backup: dataset construction

Either · 0 seconds

The datasets go through different processing steps before evaluation.

For AssemblyHands, the authors keep annotations with a valid root and more than ten valid joints per hand. They also transform the poses into Ego camera coordinates and connect them to action labels.

For Ego-Exo4D, they undistort the images, standardize the joint indices, and match the pose sequences to atomic-action descriptions.

For both datasets, half of the original validation episodes become the processed test set.

EgoMe-pose uses automatic pose estimation followed by confidence and episode-validity filtering. That helps remove unreliable annotations, but it can also favor examples where the hands are easier to detect.

Source: §4.1.1; Appendix B. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 21. Backup: efficiency and deployment

Either · 0 seconds

The full model uses about 802.8 billion floating-point operations for inference. Most of that comes from the baseline architecture.

The proposed modules add about 0.084 percent to the reported compute and about 1.87 percent to the parameter count.

So the extra modules are relatively small. However, the paper doesn't report measured inference speed or latency on a target device. We therefore can't conclude that the full system runs in real time.

A practical system would also need to obtain the previous hand poses. The effect of using an imperfect pose estimator is another open question.

Source: Table 3; calculated sums and percentages. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 22. Backup: likely questions

Either · 0 seconds

Future Exo information is used as a training target. It is not an input during inference, so the model doesn't need access to future video when it makes a prediction.

The reason for reconstructing Exo features is to learn useful context without generating all the video pixels. A limitation is that the target features can inherit biases from the visual encoder.

Language is part of the model's input, but the paper doesn't isolate its contribution with a dedicated ablation. We therefore don't know how much the model depends on the instruction.

For robotics, the result comes from feature distillation on CALVIN. It doesn't demonstrate direct real-world robot control from the predicted human hand poses.

Source: §3–4; our assessment. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 23. Backup: original method diagram

Either · 0 seconds

This is the original architecture diagram from the paper.

The inputs are encoded at the top left and passed into the causal transformer. The different queries produce future pose features, global and local Exo features, and an auxiliary Ego-image reconstruction.

At the bottom, the paired Exo video goes through a frozen MAE encoder to provide the training targets. At the top right, the predicted Exo representations guide the pose queries through global and local modulation.

The actual Exo video is used for supervision during training. The model predicts the Exo representations from the available first-person inputs during inference.

Source: Fig.2; §3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 24. References and figure credits

Either · 0 seconds

Our main source is the Exo2EgoPose paper by Shi and colleagues, presented at ACM Multimedia 2026. The paper and code links are shown here.

The research figures are reproduced or cropped from the paper and credited to the authors under the Creative Commons Attribution license.

The related-work descriptions and experimental results also come from that paper. Our critical assessment and proposed follow-up experiments are our own analysis.

Source: Primary sources and attribution. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

