# Exo2EgoPose speaking script

Niels: slides 1–8. Emiel: slides 9–19. Slides 20–25 are backups. Planned speaking time: 19 minutes 00 seconds. Slide 14: Next shows the zipper zoom; Next again returns to the overview on slide 15.

## 1. Exo2EgoPose

Niels · 30 seconds

Hi everyone. I'm Niels, and together with Emiel, I'll present Exo2EgoPose, a paper by Shi and colleagues from ACM Multimedia 2026.

The main question is whether a third-person demonstration can help a first-person model predict what someone's hands will do next.

I'll start with the problem and the main idea behind the method. Emiel will then explain how the model refines its predictions, go through the results, and discuss the strengths and limitations.

Source: p.1. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 2. What the Ego camera misses

Niels · 50 seconds

The problem starts with the first-person camera. It moves with the person's head, and it only sees part of the scene. A hand or an object can disappear from the image, even though the action is still happening.

That makes it difficult to understand the full interaction and predict the next movement.

A third-person camera can show more of the person, the objects, and the overall action. That's the extra information the authors want to use.

The idea is to learn from both views during training, so the model can make better predictions when it only has the first-person inputs.

To understand what this extra view should help with, we first need to define exactly what the model predicts.

Source: Fig.1, pp.1–2. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 3. Predicting the next ten hand poses

Niels · 65 seconds

This slide separates what the model already knows from what it has to predict.

On the left, there are three inputs: the previous first-person images, the previous 3D hand poses, and a language instruction. All three go into the same model.

For example, the instruction could be “pull up the zipper with your right hand.” The model then predicts how the hands will move over the next ten frames, shown on the right.

For every future frame, it predicts the 3D position of 42 joints, so 21 per hand, and whether each joint is valid.

Ten frames is a short forecast: about two-thirds of a second at 15 frames per second, or one-third of a second at 30 frames per second.

So we're using what we know now to predict the next hand movements.

Now that we know the inputs and outputs, we can compare this task with earlier work and see what Exo2EgoPose adds.

Source: §3.1, §4.1.3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 4. Where the paper fits

Niels · 50 seconds

There are a few research directions that lead to this paper.

Methods such as USST predict future hand trajectories. They mainly describe the overall movement, rather than the detailed position of every finger joint.

Methods such as MCIL and HULC use language to guide robot actions. GR-1 and AR-VRM also learn from human data to help with robot manipulation.

Exo2EgoPose brings these ideas together. It predicts detailed future hand poses from several inputs, and it uses a third-person demonstration as extra supervision during training.

The extra idea here is to learn third-person context during training. We can now open the forecasting model and see where that information enters.

Source: §2; Appendix A. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 5. The model: two feature paths, one pose forecast

Niels · 65 seconds

We can now open the forecasting model from slide three and follow the information through it.

The three Ego inputs first go through their encoders and a shared causal transformer. After that, there are two feature paths.

The upper path contains the pose features. These carry the information for predicting the future hand poses.

The lower path predicts Exo context. This is the part called DERM: the Dual-level Exocentric Reconstruction Module. During training, paired third-person video teaches this branch what features to predict.

Both paths meet in GLMM, the Global-to-Local Modulation Module. It uses the predicted Exo context to refine the pose features before the final hand-pose prediction.

We'll now follow this same map: first the encoders and transformer, then the DERM branch, and then the refinement step.

Source: Fig.2; §3. Shi et al. (2026), Exo2EgoPose. Simplified overview. Figures CC BY 4.0.

## 6. The shared backbone: encoders and transformer

Niels · 65 seconds

We're now looking at the highlighted backbone in the same model map.

Each input has its own encoder. The images use MAE and DINOv2 for pixel and semantic features, with 196 visual tokens reduced to nine per encoder. The instruction uses CLIP, and the past poses use HandFormer.

These tokens go into a shared 12-layer causal transformer with hidden size 384. Time embeddings identify the frame order, and the causal mask prevents access to future input observations.

The transformer also processes learned query slots. Pose-query features enter the upper path. Global and frame-level Exo-query features enter the lower DERM path.

The Ego-image queries serve a separate training objective: reconstructing the image at time t plus three.

So the backbone gives us the features for both paths. Next we'll follow the lower one and see how DERM learns the Exo context.

Source: §3.2–3.3; §4.1.3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 7. DERM: training the Exo-context branch

Niels · 85 seconds

DERM teaches the first-person model to predict useful information from a third-person view.

During training, we have both videos. The real third-person video goes through a frozen MAE encoder. This turns the frames into lists of numbers describing their visual content. Those numbers give us the targets.

Separately, the model uses the first-person video, instruction, and past hand poses to predict what those Exo features should be.

We then compare the predicted features with the features from the real Exo video. MSE measures the difference, and training adjusts the model to make that difference smaller.

This happens at two levels. One target is a single summary of the clip, made by averaging its frame features. The other target keeps a separate feature vector for each future frame. That's why it's called dual-level.

The model is learning to predict these feature vectors, rather than generating another video. Its predicted Exo features then go to GLMM to help refine the hand-pose features. The real Exo video is only needed during training.

Source: §3.3, Eqs.10–15. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 8. Training and inference use different information

Niels · 55 seconds

This distinction between training and inference is important.

During training, the model can use features from the paired third-person video as targets. Those targets can include future frames.

During inference, the model only receives the previous first-person images, the instruction, and the previous hand poses. It has to predict the Exo features internally.

So it doesn't need a third-person camera or future video at test time. The extra view teaches the model during training.

A remaining question is how much this depends on the correct pairing and timing of the videos. We'll come back to that in our discussion.

On our model map, both the pose path and the Exo-context path remain active during inference. Only the real Exo video and its training targets are removed. Emiel will now explain GLMM, where those two paths meet.

Source: §3.1–3.4. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 9. GLMM: where the two feature paths meet

Emiel · 70 seconds

We're now at the point where the two feature paths meet. GLMM stands for Global-to-Local Modulation Module.

It receives the pose features from the upper path and the predicted Exo features from DERM below. Its job is to improve the pose features before the final prediction.

First, the global Exo vector conditions the pose features through adaptive normalization and cross-attention. This brings in information summarized over the clip.

Next, the frame-level Exo sequence provides more local guidance. Cross-attention allows soft alignment, and a learned gate controls how much information is added.

The formula at the bottom is adaptive layer normalization. Gamma changes the scale and beta changes the shift of the normalized features.

The refined pose features then go to the final head, which predicts the future 3D hand poses and joint validity.

Source: Fig.3; §3.4, Eqs.16–22. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 10. GLMM: overall context, then frame details

Emiel

The title gives the order: GLMM: overall context, then frame details. We follow the arrows from left to right.

We start with Pose queries: these are the features to refine before predicting the future joints.

The Predicted clip vector is a whole-movement summary. It feeds into 1. Global modulation, whose role is to Add overall context. The box lists Adaptive layer normalization + cross-attention. AdaLN adjusts the pose features using that context. Cross-attention lets the pose queries take in relevant information from the clip vector.

Next, the Predicted frame sequence provides details over time. It feeds into 2. Local modulation, whose role is to Add frame-level detail. The box lists attention + gate + AdaLN. Attention connects each pose query to relevant frame features, even when their timing differs. The gate controls how much information is added. AdaLN adjusts the pose features using this local context.

The formula below explains AdaLN: Normalize the features, then adjust their scale (γ) and shift (β). LN means layer normalization. Gamma controls the scale adjustment, beta controls the shift, and the circled dot means element-by-element multiplication. The context determines these adjustments.

Finally, the box on the right says Predict 3D joints + joint validity with the pose and validity head: output the future joint positions and predict whether each joint is valid.

## 11. The training objective

Emiel

The model is trained with several losses.

Smooth L1 measures the error in the predicted joint coordinates. Binary cross-entropy handles joint validity, for example when a hand is missing or not involved in the action.

The global and local Exo features use mean squared error. There is also an auxiliary task that reconstructs the third future first-person image.

All these losses are added into one total training loss, which the model tries to minimize.

The pose loss has a weight of 3,000, while the other terms have different weights. These weights balance the different training objectives.

Together, the losses train the model to predict the future poses and learn useful visual context.

## 12. Evaluation across three benchmarks

Emiel

The authors evaluate the method on three benchmarks.

AssemblyHands contains toy assembly activities with synchronized first-person and third-person videos.

Ego-Exo4D covers a wider range of skilled activities and includes both manual and automatic pose annotations.

EgoMe-pose is constructed for this paper. The authors estimate the hand poses automatically with InterHand and RootNet, and then filter the annotations.

They report two metrics. MPJPE measures joint position error after root alignment. MPJVE measures the error in joint displacement between frames, so the velocity error between both frames. Lower is better for both.

The datasets have different frame rates and annotation procedures, so we compare methods within each dataset rather than comparing the absolute errors across datasets.

## 13. Exo2EgoPose has lower position error

Emiel · 85 seconds

We now compare two methods on the three benchmarks from the previous slide.

AR-VRM is the comparison method. It is adapted to combine Ego pose features and Exo features through analogical reasoning. It is an existing method that the authors adapt for this task.

Exo2EgoPose is the method from this paper. DERM predicts Exo features. GLMM uses them to refine pose features. Both methods use Exo information in this comparison, but they incorporate it differently.

We measure MPJPE: root-aligned joint position error. All values are in millimeters, and lower is better. This measures the average distance between predicted and reference joints after aligning the root joint.

Read the table row by row. Each row is a benchmark. The middle column is AR-VRM, and the highlighted right column is Exo2EgoPose.

On AssemblyHands, AR-VRM has an error of 33.39 millimeters and Exo2EgoPose has 25.83. On Ego-Exo4D, the values are 44.46 and 36.44. On EgoMe-pose, they are 56.06 and 49.44.

Exo2EgoPose has lower error on all three test sets. Compare methods within each benchmark, because the datasets have different activities and annotation procedures.

This table compares two methods. Full results include GR-1 and other baselines. AR-VRM is not the strongest position-error baseline on every dataset: GR-1 is slightly better on Ego-Exo4D. Exo2EgoPose still has the lowest MPJPE in Table 1 on all three benchmarks.

Source: Table 1; §4.2; Appendix A.6. Shi et al. (2026), Exo2EgoPose.

## 14. What the qualitative examples show

Emiel · 35 seconds

These examples show what the predictions look like. We start with Pulling a zipper on the left.

Press Next once to show the animated close-up. It compares AR-VRM, Exo2EgoPose (Ours), and Ground truth (GT) at the same frame and zoom. Compare the hand orientation and finger shape with GT.

Exo2EgoPose looks closer to the ground truth in this selected example. These are projected, scaled and aligned poses, so the picture does not establish accurate absolute 3D placement.

Press Next again to return to the overview on the following slide and discuss the other pictures.

Source: Fig.4; §4.4.1. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 15. Qualitative examples: overview again

Emiel · 20 seconds

We are back at the overview. The full sequences show how the predicted hand shapes change over time.

Now look at Picking up a watch on the right. This action uses both hands. Compare the orientation of the hands and the finger shapes across the model rows and the ground truth.

These selected examples illustrate the predictions; the quantitative results provide the broader accuracy comparison.

Source: Fig.4; §4.4.1. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 16. What do reconstruction and modulation add?

Emiel · 90 seconds

What do reconstruction and modulation add? We compare the same base model with different Exo modules. These are three versions of Exo2EgoPose.

The results are from AssemblyHands validation. MPJPE means root-aligned position error, in millimeters. Lower is better. Read the table from top to bottom.

First, No Exo modules. Predict poses from Ego inputs: past first-person images, the instruction, and past hand poses. There is no DERM reconstruction and no GLMM modulation. The error is 39.70 millimeters.

Second, Reconstruction only. DERM learns clip-level and frame-level Exo features, using the paired third-person video as training targets. This helps the shared model learn movement context. There is no GLMM: no explicit Exo-guided pose modulation. The error is 30.11 millimeters.

Third, Full model. DERM + GLMM: use the predicted Exo features to refine pose features globally, then locally. DERM supplies the movement summary and frame details. GLMM uses them to adjust the pose features before the final prediction. The error is 25.37 millimeters.

The bottom lines follow the same steps. Adding DERM: 9.59 millimeters less error, from 39.70 to 30.11. Adding GLMM: 4.74 millimeters less error, from 30.11 to 25.37.

Learning Exo context helps; using it for pose modulation improves the result further.

Source: Table 2, AssemblyHands validation, selected ablations. Shi et al. (2026), Exo2EgoPose.

## 17. Why evaluate human-to-robot transfer on CALVIN?

Niels · 45 seconds

Lower hand-pose error tells us how well the model forecasts human movement. The authors also want to know whether its learned features are useful for robot control.

They test this in CALVIN, a simulated robot benchmark with language instructions, such as opening a drawer or moving a block. The robot has to complete a sequence of five instructions.

In this experiment, the robot policy trains on scenes A, B and C, and is tested on the unseen scene D.

The human forecasting model acts as a teacher for a robot policy through feature distillation. This tests whether knowledge learned from human actions can help a robot perform manipulation tasks.

Mees et al., CALVIN (2022) · Official repository and illustration (MIT)

Sources: https://github.com/mees/calvin ; https://github.com/mees/calvin/blob/main/media/teaser.png ; Exo2EgoPose §4.4.5. CALVIN background from the benchmark authors; motivation and transfer procedure from Exo2EgoPose.

## 18. CALVIN: completing tasks in sequence

Niels · 55 seconds

We can compare the robot results using one measure: how many tasks it completes in a row.

Each CALVIN test sequence contains five language instructions. We count how many the robot completes before its first failure.

With AR-VRM, the robot completes an average of 3.29 tasks out of five. With Exo2EgoPose, that increases to 3.38 tasks out of five.

So the gain is 0.09 tasks per sequence. It's a small improvement, but it suggests that the features learned from human hand forecasting can help the robot complete longer task sequences.

The improvement is not uniform: first-task performance is lower. So we should describe this as a modest transfer benefit.

Source: Shi et al., Exo2EgoPose, §4.4.5, Table 4, Avg.L. (ABC → D). Values: AR-VRM 3.29, Exo2EgoPose 3.38. Difference calculated from the displayed values: 0.09 tasks. The paper also reports Avg.R. 65.9% and 67.5%; those percentages are omitted from this slide to keep one interpretable unit. First-task success is 0.901 versus 0.859.

## 19. Takeaways, evidence and discussion

Emiel · 100 seconds

To conclude, our message is: learn from Exo demonstrations during training; use Ego inputs at inference. A real third-person video is not required when making the forecast.

There are three takeaways. First, forecast detailed future hand poses from Ego images, language and past poses. Second, learn Exo context at two levels: DERM predicts clip-level and frame-level features. Third, refine globally, then locally: GLMM uses those features to refine pose features before the joint predictions.

On the right are the evidence and limitations. The evidence shows gains across three benchmarks. Ablations support DERM and GLMM: reconstruction helps, and adding modulation improves the result further. CALVIN results also show human-to-robot transfer in the evaluated robot tasks.

The limitations are equally important. Automatic labels may introduce bias, and there is no reported variation across training runs. Reliance on past pose inputs means that performance with noisy estimated poses remains uncertain. Root alignment hides absolute placement error: The hand shape can be correct while the whole hand is in the wrong place.

Our discussion question is: How reliable are forecasts with noisy poses and uncertain labels? How well does Exo2EgoPose work when its input hand poses or training labels are inaccurate?



## 20. Backup: full benchmark results

Either · 0 seconds

This table contains the full position and motion errors for the comparison methods. Each cell shows MPJPE followed by MPJVE, and lower is better for both.

The authors reimplement the methods for this forecasting task with common backbones. They also extend AR-VRM with Exo data for analogical reasoning.

Exo2EgoPose has the lowest reported error for both metrics on all three benchmarks. The size of the improvement varies, especially for motion error on Ego-Exo4D.

Source: Table 1; §4.2; Appendix A.6. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 21. Backup: dataset construction

Either · 0 seconds

The datasets go through different processing steps before evaluation.

For AssemblyHands, the authors keep annotations with a valid root and more than ten valid joints per hand. They also transform the poses into Ego camera coordinates and connect them to action labels.

For Ego-Exo4D, they undistort the images, standardize the joint indices, and match the pose sequences to atomic-action descriptions.

For both datasets, half of the original validation episodes become the processed test set.

EgoMe-pose uses automatic pose estimation followed by confidence and episode-validity filtering. That helps remove unreliable annotations, but it can also favor examples where the hands are easier to detect.

Source: §4.1.1; Appendix B. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 22. Backup: efficiency and deployment

Either · 0 seconds

The full model uses about 802.8 billion floating-point operations for inference. Most of that comes from the baseline architecture.

The proposed modules add about 0.084 percent to the reported compute and about 1.87 percent to the parameter count.

So the extra modules are relatively small. However, the paper doesn't report measured inference speed or latency on a target device. We therefore can't conclude that the full system runs in real time.

A practical system would also need to obtain the previous hand poses. The effect of using an imperfect pose estimator is another open question.

Source: Table 3; calculated sums and percentages. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 23. Backup: likely questions

Either · 0 seconds

Future Exo information is used as a training target. It is not an input during inference, so the model doesn't need access to future video when it makes a prediction.

The reason for reconstructing Exo features is to learn useful context without generating all the video pixels. A limitation is that the target features can inherit biases from the visual encoder.

Language is part of the model's input, but the paper doesn't isolate its contribution with a dedicated ablation. We therefore don't know how much the model depends on the instruction.

For robotics, the result comes from feature distillation on CALVIN. It doesn't demonstrate direct real-world robot control from the predicted human hand poses.

Source: §3–4; our assessment. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 24. Backup: original method diagram

Either · 0 seconds

This is the original architecture diagram from the paper.

The inputs are encoded at the top left and passed into the causal transformer. The different queries produce future pose features, global and local Exo features, and an auxiliary Ego-image reconstruction.

At the bottom, the paired Exo video goes through a frozen MAE encoder to provide the training targets. At the top right, the predicted Exo representations guide the pose queries through global and local modulation.

The actual Exo video is used for supervision during training. The model predicts the Exo representations from the available first-person inputs during inference.

Source: Fig.2; §3. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

## 25. References and figure credits

Either · 0 seconds

Our main source is the Exo2EgoPose paper by Shi and colleagues, presented at ACM Multimedia 2026. The paper and code links are shown here.

The research figures are reproduced or cropped from the paper and credited to the authors under the Creative Commons Attribution license.

The related-work descriptions and experimental results also come from that paper. Our critical assessment and proposed follow-up experiments are our own analysis.

Source: Primary sources and attribution. Shi et al. (2026), Exo2EgoPose. Figures CC BY 4.0.

CALVIN background and illustration: Mees et al. (2022), official MIT-licensed repository: https://github.com/mees/calvin .

