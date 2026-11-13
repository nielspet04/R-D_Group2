# Team rehearsal plan

Replace HOST / AR / STA with names. All members must speak and participate in Q&A. If there are more than three members, divide continuous sections between speakers rather than alternating every slide.

| Slide | Owner | Duration | Cumulative end | Handoff / goal |
|---|---|---:|---:|---|
| 1. Opening | Host | 0:40 | 0:40 | Five-second clip, one guess, ground-truth reveal |
| 2. Motivation | Host | 0:40 | 1:20 | Motivate the HCI scenario |
| 3. Related work | Host | 0:50 | 2:10 | “First, recognizing the current action.” |
| 4. AR method | AR presenter | 1:05 | 3:15 | Actual model and design choice |
| 5. AR evidence | AR presenter | 1:05 | 4:20 | “Anticipation removes the action itself from the input.” |
| 6. STA protocol | STA presenter | 0:55 | 5:15 | Participant generalization and metric |
| 7. STA timing | STA presenter | 1:05 | 6:20 | One-second gap, no leakage |
| 8. STA models | STA presenter | 1:05 | 7:25 | Prior vs pooling vs temporal order |
| 9. STA evidence | STA presenter | 1:15 | 8:40 | Separate quality scores from extraction validation |
| 10. Next steps | AR + STA / host | 0:50 | 9:30 | Concrete next experiments; close |

There are 30 seconds of headroom before the strict 10:00 limit. Durations include media and handoffs. If late, shorten the related-work explanation and extraction discussion; keep both task contributions and next steps.

## Before rehearsal

- Finish AR placeholders, add actual architecture reference, and update AR notes.
- Check the latest STA job status against logs before changing any claims.
- Check speaker notes in the second display; test S and allow popups.
- Test both videos and fullscreen on the presenting machine.
- Assign one slide operator; agree on verbal or physical advance cues.
- Put the fresh PDF locally on the presenting machine as a fallback.

## Prepare for Q&A

Each member should understand the high-level AR/STA distinction and the shared motivation. Task leads handle technical specifics, while others can add evidence.

Likely STA questions: Why split by participant? Why class-mean Top-5? Why frozen R3D-18? Why a GRU? How is future leakage prevented? What if context is missing? What do tests establish versus actual predictive results?

Likely AR questions: Why this architecture? Which inputs and pretraining? What is the validation protocol? Are results comparable with a baseline? What is the main failure mode?

Honest limits: source notes have no full M1/M2 trained results; the H100 run verifies only a small extraction sample; the local split is not the official benchmark split; the application is a motivation, not a built assistant.
