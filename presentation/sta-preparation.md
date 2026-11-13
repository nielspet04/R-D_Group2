# Niels & Emiel — STA presentation preparation

**For Niels’s latest simple script, use [my-speaking-notes.md](my-speaking-notes.md). It replaces the suggested spoken text below and includes the queued extraction and video problems.**

The live presentation is a PDF. Keep this guide on paper or another device; browser speaker notes and videos are not available on the presenting laptop. The full AR/STA talk targets 9:30. The AR section remains for the AR teammate to complete.

## Your main message

**I turn the raw videos into reusable features while enforcing what the anticipation model is allowed to see. Emiel uses these same features to compare the models.**

Lead with the purpose of your work. The audience does not need a tour of the code or a chronological account of cluster setup.

## Speaking split

| Part | Speaker | Time |
|---|---|---:|
| Slide 6: participant split and metric | Niels | 0:55 |
| Slide 7: legal context, extraction and handoff | Niels | 1:05 |
| Slide 8: models and training | Emiel | 1:05 |
| Slide 9: baseline scores | Emiel | 0:40 |
| Slide 9: extraction evidence | Niels | 0:35 |
| Slide 10: feature completion | Niels | ~0:10 |
| Slide 10: model experiments | Emiel | ~0:15 |

Your allocated time is approximately 2:45 including next steps. Emiel has approximately 2:00. The host and AR presenter cover the other sections. Existing notes inside the HTML have been updated to match.

## Your spoken explanation

### Slide 6 — why the data setup matters (55 seconds)

“For anticipation, we first needed a reliable way to prepare and evaluate the input. We split the data by participant: 26 people for training and six for validation. This means the validation actions come from people who were excluded from training, rather than mixing actions from the same people across both sets.

“We reuse this split for all three STA models. Because some actions are much more frequent than others, we use class-mean Top-5 recall: calculate recall per class and average the classes equally. That prevents the most common labels from dominating the metric.”

Point to the participant counts, not every event count. Call this the project split; do not call it the official benchmark split.

### Slide 7 — your feature pipeline (65 seconds)

“For an action starting at time t, I take five seconds of context ending one second before the action. For an action at 20 seconds, that means video from 14 up to, but not including, 19 seconds. Nothing from the gap or the action itself is allowed into the input.

“I divide the context into five one-second segments and sample 16 frames from each. A frozen pretrained R3D-18 converts each valid segment into a 512-dimensional feature vector. If a segment is unavailable near the beginning of a video, I mark it invalid using a mask.

“I save and validate these features so we do not need to decode the videos again during each training run. Emiel receives five feature vectors and the validity mask for each event.”

**Handoff:** “Emiel will explain how the two models use exactly this same input.”

Point left to right along the timeline. Say “five vectors of 512 numbers” if “five by 512 features” feels abstract. You do not need to list storage types, batch sizes, file names, or scheduler settings.

### Slide 9 — your measured evidence (35 seconds, after Emiel)

“On the extraction side, the real H100 smoke run processed 64 events in 48 seconds with no video errors. We checked the saved features against the annotations and loaded them back successfully. The job also passed 58 tests before the timed extraction.

“This confirms that the extraction pipeline works on that sample. It does not yet tell us how well the trained models predict. The next step is to finish and verify the full feature sets.”

If a completed full-extraction log is available, replace the final sentence with its verified status. Do not infer completion from a submitted job. The 48 seconds is extraction time only. The later plan reports 69 tests in another development snapshot; do not change the historical smoke-run count to 69.

### Slide 10 — your next action (10 seconds)

“My next step is to finish training and validation feature extraction and verify that every event is represented correctly.”

Then Emiel states the model-training experiments. Avoid both people repeating the whole pipeline.

## Understand the terms you say

**Feature:** a numerical representation of a video segment. The encoder replaces raw pixels with 512 numbers that capture learned visual patterns. It is not yet a verb or noun prediction.

**R3D-18:** a pretrained 3D convolutional video network. Unlike an image-only encoder, its convolutions process space and time. Our extraction removes its final classification layer and uses its 512-dimensional representation.

**Frozen:** the encoder weights are not updated in our initial experiments. We use evaluation and inference mode for extraction. Emiel trains the downstream pooling/GRU prediction models using the cached outputs.

**Mask:** five Boolean flags saying which segment features represent usable video. Padding is stored as zeros, but the explicit mask is needed so models can distinguish missing context from actual data.

**Cached:** features are computed once and stored. This avoids repeating video decoding and encoder work for every epoch and every model comparison. It does not mean the full dataset has already been extracted.

**Shard:** one file containing many events. The planned default is 2,048 events per file. Completed shards can be validated and skipped when extraction resumes.

**Leakage:** using information that would not be available when making the prediction. Seeing the one-second gap or target action would make anticipation results misleading.

## Questions you should be ready for

**Why do feature extraction separately?**

“Video decoding and the backbone are repeated work. Caching lets both downstream models use the same visual input without rerunning that work every epoch.”

**Why R3D-18?**

“It supplies a practical pretrained video representation that models space and time. Freezing it gives us a manageable starting point for comparing downstream models.” Do not claim it is the optimal backbone or has been benchmarked against alternatives in this project.

**Why five seconds and 16 frames?**

“They are the fixed initial design choices: five one-second segments, each represented by 16 sampled frames. We have not yet shown that these are optimal.” The one-second anticipation gap is part of our task protocol; distinguish it from the chosen context length.

**How do you prevent future-frame leakage?**

“The sampler enforces that every selected frame timestamp is strictly before action_start minus one second, and we test the boundaries.” If pressed: frame indices range from ceil(segment_start × FPS) up to, but excluding, ceil(segment_stop × FPS).

**What happens at the beginning of a recording?**

“Unavailable segments are zero-padded and marked invalid. For example, an action at four seconds has the mask [False, False, True, True, True]. If no usable context exists, the event is retained.” Emiel can add: those events do not contribute to training loss and receive training-frequency predictions during shared validation. This is implemented; hidden-test submission generation is still future work.

**Why not drop difficult or broken events?**

“We preserve the event identities and make missing context or video errors explicit. That keeps evaluation coverage honest rather than silently evaluating only easy-to-read events.”

**How do you know the handoff to Emiel is correct?**

“We agreed on the shard schema, tensor shapes, labels, masks and IDs. Shards are validated, and the dataset loader exposes the same interface to both models.” The per-event feature shape is [5,512]; a training batch is [B,5,512] with mask [B,5].

**Does this demonstrate good prediction performance?**

“No. Extraction validation establishes that the input pipeline works. Predictive quality needs full training and validation, compared against the frequency baseline.”

**Does the GRU isolate the value of temporal order?**

“Both models use the same visual features, but they also differ in model capacity. The initial comparison tests temporal modelling; a shuffled-order ablation would help isolate the contribution of order.” Let Emiel lead this answer.

## Three short rehearsal passes

1. Explain the timeline and handoff without notes in 65 seconds. Make sure a listener can repeat what the model sees and cannot see.
2. Rehearse with Emiel: your handoff on slide 7, his model explanation and baseline scores, then your 35-second extraction evidence on slide 9.
3. Rehearse the complete PDF with the AR teammate and all handoffs. Stop by 9:30. If late, cut implementation detail rather than rushing or dropping next steps.

Your cue card can be just:

**People split → five seconds → one-second gap → 16 frames → frozen encoder → five vectors + mask → cache → Emiel.**

For results: **64 events / 48 seconds / zero errors / correct input pipeline, not trained-model performance.**

## Evidence used

- `../docs/plan_STA.md`: ownership, extraction design, model/trainer integration and future experiments.
- `../docs/milestones_1_to_10_presentation_notes.md`: measured baseline and historical H100 extraction evidence.
- `../src/sta_training.py`: confirms no-context fallback in shared validation.
- `../results/tables/sta_model_0_frequency.json`: exact measured baseline values.

The plan's model-development status and the extraction notes describe different work streams and snapshots. This guide does not treat planned steps or correctness fixtures as completed real-data experiments.
