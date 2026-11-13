# Niels — speaking notes

Names and times are cues. Read the paragraphs below them.

## Slide 6 — Niels · 4:20–5:15

I save one record for each action. It contains five feature vectors, each with 512 numbers.

There are also five true-or-false values called a mask. They show which video parts are available. I save the verb, noun and event ID alongside the features.

These records are stored in .pt files called shards. Emiel reads them with our shared loader. Agreeing on this format let us work in parallel.

We use 26 people for training and six different people for validation. This checks how the models work on people they have not seen during training.

The next slide shows how I create these features.

## Slide 7 — Niels · 5:15–6:20

We use five seconds of video and stop one second before the action starts.

For an action at 20 seconds, we use video from 14 seconds up to, but not including, 19 seconds.

I split this into five one-second parts and take 16 frames from each.

A pretrained network called R3D-18 turns each part into 512 numbers. I do not train this network again.

I mark missing parts with a mask and save the features so we only do this work once.

Emiel will explain how his models use these features.

## Slide 9 — Niels · 7:55–8:40

The validation feature extraction is now finished. The validation set has 7,450 actions, saved in four files called shards.

Saving in separate files means we can reuse completed parts if a later extraction job stops.

I had some video errors during development. I fixed the local test inputs and downloaded one corrupted video again.

Training-set feature extraction is now queued. Once that is ready, Emiel can train the video models.

## Slide 10 — Niels · 8:50–9:00

For STA, I will test different frame sampling and context lengths to see which video input gives better results.

## Slide 13 — Niels · Q&A

If an action happens near the start of a video, some context is missing. The mask tells the model which parts are real.
