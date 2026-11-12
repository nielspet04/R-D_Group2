# HCVISION mid-term — Reveal.js

A 10-slide main talk, timed to **9:30**, plus three backup slides. The AR method, results, next experiment and architecture reference must be completed by the AR teammate. The PDF is a **draft**, not submission-ready until these edits are made.

## Present locally (no internet needed)

On macOS, double-click `Start.command`. Alternatively, in a terminal in this folder:

```sh
python3 serve.py
```

The deck opens at http://localhost:8765/. Keep the terminal running. Use the actual presenting laptop for rehearsal, including its projector display configuration.

- Left/right arrows or Space: advance (some slides have one reveal step).
- **S**: speaker window with notes, next slide, and timer. Allow the popup. Use an extended display for private notes.
- **F**: fullscreen. **Esc**: overview. **?**: shortcuts.
- Click the video play button on slides 1 and 4. Both clips are local and muted. Do not rely on autoplay.
- Hover the top-right toolbar to access editing and PDF tools. Controls are excluded from PDF.
- The main talk ends on slide 10. References and technical material follow for Q&A.

Directly opening index.html can display the deck, but the local server is recommended, especially for speaker view. Nothing needs to be uploaded to slides.com.

## AR teammate: edit without writing code

1. Open the local deck and go to slide 4.
2. Press **E** (or click **Edit text**). Editable text gets a dotted outline.
3. Replace the AR method and results on slides 4 and 5. Update the AR next step on slide 10. Update the related-work slide and references for the architecture actually used.
4. Press **Esc** or **Finish editing** to restore slide keyboard controls.
5. Click **Save edited HTML**. Replace this folder's `index.html` with the downloaded file, keeping the filename `index.html`. The file still needs this folder's assets, vendor files, theme.css, and deck.js.
6. Refresh and verify the edits. Unsaved browser edits are lost on reload. Export a fresh PDF afterward.

The editor supports text changes. For images, table rows, layout, or speaker notes, edit `index.html`. Each slide is a `<section>` and its notes are inside `<aside class="notes">`. Styling lives in `theme.css`; behavior in `deck.js`. Text changes must fit the existing layout: shorten text instead of reducing fonts.

### AR inputs required

- Names / roles of the team: introduce verbally, or add a compact line on the opening.
- Actual AR architecture, input sampling, pretraining and training setup.
- AR validation protocol and chosen metrics (do not assume the STA split).
- Measured baseline/model results, clearly attributed, with comparable settings.
- One supported finding or failure example and the next controlled experiment.
- Scientific references for AR and any borrowed figure or result.

Unknown AR results are written as pending, never fabricated. Replace every “AR teammate”, “Add…”, and “Pending teammate input” before final submission.

## PDF for Toledo

The supplied `midterm-draft.pdf` reflects the initial draft only. After edits:

1. Click **PDF view** (or open http://localhost:8765/?print-pdf).
2. In Chrome or Edge, Print → Save as PDF.
3. Landscape; background graphics on; margins none; browser headers/footers off. The deck defines its own 16:9 page size.
4. Check all 13 pages. Fragments appear fully revealed and videos are represented by still frames.
5. Use a normal slide PDF without speaker notes. References and backup slides can remain after the main talk.

Deadline from course notice: **13 November 2026, 15:00 CET**. The 10-minute limit applies to the live presentation; backup slides are for Q&A.

## Scientific status and evidence

STA content is grounded in `../docs/milestones_1_to_10_presentation_notes.md` and `../results/tables/sta_model_0_frequency.json`, as recorded on 12 November 2026. No `docs/presentation.md` was present during creation.

- M0 scores are measured on the project participant split, not the official benchmark.
- M1 and M2 are implemented and tested, but the source notes do not document full training or comparative performance.
- The real H100 smoke extraction covered 64 events in 48 seconds with zero video errors; 58 tests passed separately. Do not extrapolate full dataset throughput from this timing.
- Full validation extraction was submitted and waiting at the source snapshot. Update that status only after verifying a completed log.
- Frequency-prior fallback for no-context events is planned; full trainer/evaluator integration remains ahead.
- Footage P01_03_5: “take soy milk”, annotated action 13.57–14.56 seconds. Opening context is 7.57–12.57 seconds, ending one second before the action. The revealed answer is ground truth, not a model output.
- The five-second observation-window diagram is a schematic; features use 16 sampled frames per valid one-second segment.

## Team rehearsal

See `rehearsal.md` for the timing and handoff plan. Rehearse twice with videos, reveal clicks, all speakers and handoffs. Notes are speaking cues, not a script to read verbatim. Keep everyone involved in Q&A.

## Files

- `index.html`: editable content and speaker notes
- `theme.css`, `deck.js`: styling and controls
- `assets/`: local video clips and PDF stills
- `vendor/`: Reveal.js 5.2.1 (MIT license included)
- `midterm-draft.pdf`: draft submission artifact
- `rehearsal.md`: team timings

The full folder is portable. Do not share only the HTML. The project footage retains its dataset terms and attribution.
