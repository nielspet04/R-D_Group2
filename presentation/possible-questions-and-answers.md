# Possible questions and short answers

Use these as short spoken answers. The section title shows who should lead. Current extraction status is based on Niels’s update on 13 November 2026.

## Niels - data and feature extraction

**What is a feature?**

It is a list of numbers that represents a short part of the video. Our encoder gives 512 numbers per segment. These are inputs for Emiel’s models, not action predictions yet.

**Why extract and save features first?**

Reading videos and running the encoder takes time. Saving the features means we do that work once. Both models can then use the same saved input during training.

**Why use R3D-18, and why keep it fixed?**

It is a pretrained video network that uses both space and time. Keeping it fixed gives us a practical starting point. We have not shown that it is the best possible encoder.

**Why five seconds of context and a one-second gap?**

The one-second gap is part of our anticipation setup: we must predict before the action starts. Five seconds of context is our initial design choice. We have not yet compared different context lengths.

**How do you stop the model from seeing the future?**

Every sampled frame must be earlier than one second before the action. For an action at 20 seconds, we use frames from 14 seconds up to, but not including, 19 seconds. We test these limits.

**What happens when some video context is missing?**

We store zeros for missing segments and mark them with a mask. The models only use the valid parts. If no usable context exists, shared validation uses the training-label frequencies instead.

**What caused the video errors, and how did you fix them?**

Some early local tests used videos that were not on my laptop. I changed the test data. I also found one corrupted video and downloaded it again. I do not know why that file was corrupted.

**What have you checked, and what is still waiting?**

The small GPU test processed 64 events in 48 seconds with no video errors. That only checks a small sample. Training-set feature extraction is now queued. Once it finishes, I will check the files and pass the features to Emiel.

## Emiel - models and evaluation

**What is the difference between the three models?**

M0 only uses label frequencies from training. M1 averages the valid video features. M2 uses a GRU to read those features in time order. The two video models use the same features from Niels.

**What does the baseline tell you?**

It shows what we can predict without looking at video. Its class-mean Top-5 recall is 6.94 percent for verbs, 2.87 percent for nouns and about 0.27 percent for full actions. These are the reference scores for our video models.

**Why is the full-action score much lower?**

The verb and noun must both be correct as a pair. There are also more action pairs than individual verbs or nouns. A frequency-only model has limited coverage of those pairs.

**Why use class-mean Top-5 recall?**

We check whether the correct label is among the top five predictions, calculate recall per class, and average across classes present in validation. This gives rare and common classes the same weight.

**Why split by person instead of by action?**

We want to check whether the models work on people excluded from training. Splitting individual actions could put actions from the same person in both sets. We use one fixed participant split for all STA models.

**How will you train and choose the models?**

First, I will check that they can learn from a small feature sample. Then I will train on the full training features and choose checkpoints using validation Action MT5R. Both models use the same split and evaluator.

**What if the GRU is worse? Would a better GRU prove that order helps?**

A worse score is still a useful result, and we would report it. A better score alone would not prove that order is the reason, because the models also differ in capacity. Shuffling feature order would be a useful extra test.

## Team and Zoe - project questions

**Why is this useful for human-computer interaction?**

An assistant needs to understand what someone is doing. Predicting the next action could help it offer instructions at the right time. That is our motivation; we have not built the assistant itself.

**How did you divide the work?**

Zoe worked on AR. Niels prepares the STA videos and saved features. Emiel builds, trains and evaluates the STA models. For STA, we agreed on one feature format so extraction and model development could happen in parallel.

**What is the difference between AR and STA?**

AR sees the action itself and predicts its verb and noun. STA only sees earlier context, ending one second before the action starts, and predicts the upcoming action.

**Can you compare your scores with published benchmark results?**

Not directly. Our STA scores use our own participant split. Published results may use different splits and settings. Our first comparison is between our models under the same setup.

**Why do you not have full trained STA results yet?**

We have measured the frequency baseline and checked the feature pipeline on a small real sample. The two video models and training code are implemented, but full model training still needs the extracted features. Training-feature extraction is queued.

**What are the next steps?**

Niels finishes extraction and checks the saved files. Emiel then checks learning on a small sample, trains both models and compares them with the baseline. Zoe explains the next AR experiment using her actual method and results.

## Zoe: add these AR-specific answers before presenting

- Which AR architecture did you use, and why?
- What are the measured AR results, on which split and metric?
- What is the main AR failure case and next experiment?

The actual AR method and results are not available in this workspace. Zoe should fill these answers from her own work.

## If you do not know an answer

“We have not tested that yet, so I cannot give a measured answer.”

“Emiel can explain the model details.” / “Niels can explain the extraction.” / “Zoe can explain the AR part.”

Project sources: docs/plan_STA.md; docs/sta_feature_shard_contract.md; docs/milestones_1_to_10_presentation_notes.md; results/tables/sta_model_0_frequency.json. Current queue status and corrupted-file repair: Niels’s updates.
