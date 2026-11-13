# Team rehearsal plan

Niels owns STA data and extraction; Emiel owns STA models, training and evaluation. Assign HOST and AR presenter to the remaining team members. All members must speak and participate in Q&A. If there are more than three members, divide continuous sections between speakers rather than alternating every slide.

| Slide | Owner | Duration | Cumulative end | Handoff / goal |
|---|---|---:|---:|---|
| 1. Opening | Host | 0:40 | 0:40 | Static context frame and annotated action; no video dependency |
| 2. Motivation | Host | 0:40 | 1:20 | Motivate the HCI scenario |
| 3. Related work | Host | 0:50 | 2:10 | “First, recognizing the current action.” |
| 4. AR method | AR presenter | 1:05 | 3:15 | Actual model and design choice |
| 5. AR evidence | AR presenter | 1:05 | 4:20 | “Anticipation removes the action itself from the input.” |
| 6. STA protocol | Niels | 0:55 | 5:15 | Participant generalization and metric |
| 7. STA timing | Niels | 1:05 | 6:20 | One-second gap, no leakage |
| 8. STA models | Emiel | 1:05 | 7:25 | Prior vs pooling vs temporal order |
| 9. STA evidence | Emiel 0:30 → Niels 0:45 | 1:15 | 8:40 | Separate quality scores from extraction validation |
| 10. Next steps | AR + Niels + Emiel / host | 0:50 | 9:30 | Concrete next experiments; close |

There are 30 seconds of headroom before the strict 10:00 limit. Durations include media and handoffs. If late, shorten the related-work explanation and extraction discussion; keep both task contributions and next steps.

## Before rehearsal

- Finish AR placeholders, add actual architecture reference, and update AR notes.
- Check the latest STA job status against logs before changing any claims.
- Rehearse in the actual PDF viewer; the presenting laptop will only have the PDF.
- Keep cue cards / rehearsal notes on paper or a separate device.
- Do not depend on video, speaker mode, or animations during delivery.
- Assign one slide operator; agree on verbal or physical advance cues.
- Put the fresh PDF locally on the presenting machine as the main presentation file.

## Prepare for Q&A

Each member should understand the high-level AR/STA distinction and the shared motivation. Task leads handle technical specifics, while others can add evidence.

Likely STA questions: Why split by participant? Why class-mean Top-5? Why frozen R3D-18? Why a GRU? How is future leakage prevented? What if context is missing? What do tests establish versus actual predictive results?

Likely AR questions: Why this architecture? Which inputs and pretraining? What is the validation protocol? Are results comparable with a baseline? What is the main failure mode?

Honest limits: source notes have no full M1/M2 trained results; the H100 run verifies only a small extraction sample; the local split is not the official benchmark split; the application is a motivation, not a built assistant.

Niels has 2:45 across slides 6, 7 and the extraction portion of 9, plus about 0:10 on next steps. Emiel has 1:35 across slides 8 and the baseline portion of 9, plus about 0:15 on next steps. Transition on slide 9 is Emiel → Niels; agree who advances the PDF.

Use `my-speaking-notes.md` for Niels’s latest simple, word-for-word script. It includes the corrupted video, the local video errors and the queued training-feature extraction.
