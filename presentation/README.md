# HCVISION mid-term — Reveal.js

A 10-slide main talk, timed to **9:30**, plus three backup slides. The AR method, results, next experiment and architecture reference must be completed by the AR teammate. The PDF is a **draft**, not submission-ready until these edits are made.

## Present locally (no internet needed)

On macOS, double-click `Start.command`. Alternatively, in a terminal in this folder:

```sh
python3 serve.py
```

The deck opens at http://localhost:8765/ with autosave. Keep the terminal running; use `serve.py`, not a plain `python3 -m http.server`, for saving. Use the actual presenting laptop for rehearsal, including its projector display configuration.

- Left/right arrows or Space: advance (some slides have one reveal step).
- **S**: speaker window with notes, next slide, and timer. Allow the popup. Use an extended display for private notes.
- **F**: fullscreen. **Esc**: overview. **?**: shortcuts.
- Click the video play button on slides 1 and 4. Both clips are local and muted. These are optional for browser rehearsal. The presenting laptop uses the PDF, which shows still frames; the talk must not depend on playback.
- Hover the top-right toolbar to access editing and PDF tools. Controls are excluded from PDF.
- The main talk ends on slide 10. References and technical material follow for Q&A.

Directly opening index.html can display the deck, but autosave requires the local `serve.py` server. Nothing needs to be uploaded to slides.com.

## AR teammate: edit without writing code

1. Open the local deck and go to slide 4.
2. Press **E** (or click **Edit text**). Editable text gets a dotted outline.
3. Replace the AR method and results on slides 4 and 5. Update the AR next step on slide 10. Update the related-work slide and references for the architecture actually used.
4. Press **Esc** or **Finish editing** to restore slide keyboard controls.
5. Pause briefly after typing. The toolbar shows **Saving…**, then **Saved**. Changes are written directly to `index.html`; there is no downloaded file to move. **⌘S / Ctrl+S** or **Finish editing** also saves immediately.
6. Once it says **Saved**, refreshing retains the edits. Export a fresh PDF afterward. **Download HTML backup** is an optional manual copy.

The editor supports text changes. For images, table rows, layout, or speaker notes, edit `index.html`. Each slide is a `<section>` and its notes are inside `<aside class="notes">`. Styling lives in `theme.css`; behavior in `deck.js`. Text changes must fit the existing layout: shorten text instead of reducing fonts.

### AR inputs required

- Names / roles of the team: introduce verbally, or add a compact line on the opening.
- Actual AR architecture, input sampling, pretraining and training setup.
- AR validation protocol and chosen metrics (do not assume the STA split).
- Measured baseline/model results, clearly attributed, with comparable settings.
- One supported finding or failure example and the next controlled experiment.
- Scientific references for AR and any borrowed figure or result.

Unknown AR results remain bracketed placeholders. Zoe fills those fields before final submission.

## PDF for Toledo

The supplied `midterm-draft.pdf` reflects the initial draft only. After edits:

1. Click **PDF view** (or open http://localhost:8765/?print-pdf).
2. In Chrome or Edge, Print → Save as PDF.
3. Landscape; background graphics on; margins none; browser headers/footers off. The deck defines its own 16:9 page size.
4. Check all 13 pages. Fragments appear fully revealed and videos are represented by still frames.
5. Use a normal slide PDF without speaker notes. References and backup slides can remain after the main talk.

Deadline from course notice: **13 November 2026, 15:00 CET**. The 10-minute limit applies to the live presentation; backup slides are for Q&A.

## Scientific status and evidence

STA content is grounded in `../docs/plan_STA.md` (updated model/trainer status), `../docs/milestones_1_to_10_presentation_notes.md` and `../results/tables/sta_model_0_frequency.json`, as recorded on 12 November 2026. No `docs/presentation.md` was present during creation.

- M0 scores are measured on the project participant split, not the official benchmark.
- M1 and M2 are implemented and tested, but the source notes do not document full training or comparative performance.
- The real H100 smoke extraction covered 64 events in 48 seconds with zero video errors; 58 tests passed separately. Do not extrapolate full dataset throughput from this timing.
- Full validation extraction was submitted and waiting at the source snapshot. Update that status only after verifying a completed log.
- The updated STA plan and `src/sta_training.py` confirm training/validation fallback integration: no-context events are excluded from loss and use training-frequency scores during validation. Hidden-test submission generation remains ahead.
- Footage P01_03_5: “take soy milk”, annotated action 13.57–14.56 seconds. Opening context is 7.57–12.57 seconds, ending one second before the action. The revealed answer is ground truth, not a model output.
- The five-second observation-window diagram is a schematic; features use 16 sampled frames per valid one-second segment.

## Team rehearsal

See `rehearsal.md` for the timing and handoff plan, and `sta-preparation.md` for Niels’s script and Q&A. Rehearse twice using the PDF, all speakers and handoffs. Bring speaking cues separately; the presenting laptop has only the PDF. Notes are speaking cues, not a script to read verbatim. Keep everyone involved in Q&A.

## Files

- `index.html`: editable content and speaker notes
- `theme.css`, `deck.js`: styling and controls
- `assets/`: local video clips and PDF stills
- `vendor/`: Reveal.js 5.2.1 (MIT license included)
- `midterm-draft.pdf`: draft submission artifact
- `rehearsal.md`: team timings

The full folder is portable. Do not share only the HTML. The project footage retains its dataset terms and attribution.

## Latest revision — 13 November 2026

The motivation now states AR and STA research questions. The evidence slide includes video-input lessons and the queued training-set feature extraction reported by Niels. The next-step table assigns Niels and Emiel concrete actions and success checks. See `ar-handoff.md` for the remaining AR fields; the deck remains a draft until those are filled.

## Team speaking guide

Zoe handles AR; Niels handles STA features; Emiel handles STA models. Each slide’s notes now identify the speaker and planned time. `team-speaking-notes.md` contains all scripts and handoffs; `my-speaking-notes.md` contains only Niels’s parts. Keep them separately for PDF delivery.

## Automatic saving

- Autosave runs about 0.7 seconds after you stop typing. Look for **Saved** before closing or refreshing.
- Each changed save keeps the previous HTML in `.backups/`. These local backups are excluded from the portable ZIP.
- If the file changed in another tab or editor, autosave stops rather than overwriting it. Use **Download HTML backup** to preserve your current edits before reloading.
- If you see a save error or “Autosave unavailable”, keep the page open and download a backup. Start the server with `python3 serve.py`; a basic static server cannot accept saves.
- Browser slide edits save to `index.html`. Speaking-guide Markdown files and the PDF are separate. The PDF is updated only when you export again.
- The old browser copy was preserved as `.backups/browser-before-autosave.html` when autosave was installed.
