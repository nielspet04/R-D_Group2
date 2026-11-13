# Niels — speaking notes

Names and times are cues. Read the paragraphs below them.

## Slide 6 — Niels · 4:20–5:15

I prepare the video data for Emiel’s models.

We agreed on one saved feature format. This let me build the extraction while Emiel worked on the models using test features.

We split the data by person: 26 people for training and six for validation. This checks how the models work on people they have not seen during training.

All STA models use the same split. Our main score gives each action class the same weight.

## Slide 7 — Niels · 5:15–6:20

We use five seconds of video and stop one second before the action starts.

For an action at 20 seconds, we use video from 14 seconds up to, but not including, 19 seconds.

I split this into five one-second parts and take 16 frames from each.

A pretrained network called R3D-18 turns each part into 512 numbers. I do not train this network again.

I mark missing parts with a mask and save the features so we only do this work once.

Emiel will explain how his models use these features.

## Slide 9 — Niels · 7:55–8:40

I had some video errors during development. The first local tests used videos that were not on my laptop, so I changed the test data.

I also found one corrupted video and had to download it again.

The small GPU test processed 64 actions in 48 seconds, with no video errors.

Feature extraction for the training set is now queued. This prepares the features; model training comes after that.

## Slide 10 — Niels · 8:50–9:00

Once extraction finishes, I will check the saved features, masks and action IDs. Then I will pass the features to Emiel.

## Slide 13 — Niels · Q&A

If an action happens near the start of a video, some context is missing. The mask tells the model which parts are real.
