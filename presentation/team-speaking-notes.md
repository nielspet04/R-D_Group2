# Who speaks when

Speaker names and times are cues, not spoken text. Everything below each cue is the speaking script. Brackets mark AR content that Zoe still needs to fill.

**Team split:** Zoe — AR. Niels — STA video data and feature extraction. Emiel — STA models, training and evaluation.

## Slide 1 — Opening

**Zoe · 0:00–0:40**

Our project asks two questions: what is someone doing, and what will they do next?

I worked on action recognition. Niels prepares the video features for anticipation, and Emiel builds and trains the anticipation models.

Recognition looks at the action itself. Anticipation only looks at what happens before it.

## Slide 2 — Motivation

**Zoe · 0:40–1:20**

Imagine a cooking assistant that understands what you are doing. Predicting the next action could help it give the next instruction at the right time.

For AR, we ask how well we can recognize the current action. For STA, we ask whether video helps predict the next action, and whether time order helps further.

Emiel will briefly explain the related work.

## Slide 3 — Related work

**Emiel · 1:20–2:10**

EPIC-KITCHENS gives us first-person videos with action labels.

For STA, we use a pretrained video network to extract features. Earlier work, such as RU-LSTM, uses the order of events to predict what comes next.

We start with a simpler comparison: averaging the features or using a GRU.

Zoe will now explain action recognition.

## Slide 4 — AR method

**Zoe · 2:10–3:15**

For action recognition, the model sees the action itself and predicts a verb and a noun.

Our model is [AR model name]. We use [input and training setup].

## Slide 5 — AR evidence

**Zoe · 3:15–4:20**

[AR baseline score, current model score and main finding.]

Niels will now explain how we prepare the videos for anticipation.

## Slide 6 — STA protocol

**Niels · 4:20–5:15**

I save one record for each action. It contains five feature vectors, each with 512 numbers.

There are also five true-or-false values called a mask. They show which video parts are available. I save the verb, noun and event ID alongside the features.

These records are stored in .pt files called shards. Emiel reads them with our shared loader. Agreeing on this format let us work in parallel.

We use 26 people for training and six different people for validation. This checks how the models work on people they have not seen during training.

The next slide shows how I create these features.

## Slide 7 — STA timing

**Niels · 5:15–6:20**

We use five seconds of video and stop one second before the action starts.

For an action at 20 seconds, we use video from 14 seconds up to, but not including, 19 seconds.

I split this into five one-second parts and take 16 frames from each.

A pretrained network called R3D-18 turns each part into 512 numbers. I do not train this network again.

I mark missing parts with a mask and save the features so we only do this work once.

Emiel will explain how his models use these features.

## Slide 8 — STA models

**Emiel · 6:20–7:25**

We compare three models. The first only uses how often labels appear in the training data.

The second averages the valid video features. The third uses a GRU to process them in time order.

My part is building the models, training them, and checking their scores. Both video models use the same features from Niels.

The code is ready and tested. We still need to train the video models on the full feature set and compare the results.

## Slide 9 — STA evidence

**Emiel · 7:25–7:55**

Our first model scores 6.94 percent for verbs, 2.87 percent for nouns, and about 0.27 percent for full actions. These are our starting scores.

Niels will explain the completed validation extraction.

**Niels · 7:55–8:40**

The validation feature extraction is now finished. The validation set has 7,450 actions, saved in four files called shards.

Saving in separate files means we can reuse completed parts if a later extraction job stops.

I had some video errors during development. I fixed the local test inputs and downloaded one corrupted video again.

Training-set feature extraction is now queued. Once that is ready, Emiel can train the video models.

## Slide 10 — Next steps

**Zoe · 8:40–8:50**

For AR, I will test frame count, fine-tuning and weighted loss against the sixteen-frame baseline.

**Niels · 8:50–9:00**

For STA, I will test different frame sampling and context lengths to see which video input gives better results.

**Emiel · 9:00–9:20**

I will first train pooling and GRU, then tune the learning rate, dropout and loss. We will look at mistakes to decide whether to try stronger features or fine-tuning. We will keep changes that improve our validation score.

**Zoe · 9:20–9:30**

Our goal is to get the strongest results we can, and understand which changes help. Thank you.

## Slide 12 — Metric details

**Emiel · Q&A**

Top-1 accuracy checks whether the first prediction is correct.

Class-mean Top-5 recall checks whether the correct label is in the top five, then averages the score across classes.

## Slide 13 — Edge cases

**Niels · Q&A**

If an action happens near the start of a video, some context is missing. The mask tells the model which parts are real.

**Emiel · Q&A**

If there is no usable video, we keep the action and use training-label frequencies during validation. We do not use that example to train the video models.
