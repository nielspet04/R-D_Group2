# Exo2EgoPose research seminar

This is the editable Reveal.js version, following the previous mid-term presentation’s local workflow. It has 18 main slides, about 18½ minutes of speaking notes, and 6 backup slides. Niels presents slides 1–8 and Emiel presents slides 9–18.

## Open and edit

Double-click `Start.command` on macOS. It starts the local editor at http://localhost:8766/. Keep its terminal open while editing. Alternatively run `python3 serve.py` in this folder. If the server is already running, just open that address.

The toolbar appears at the top right when hovered.

- **E / Edit:** edit text and table cells directly. Finish editing or pause briefly, then wait for **Saved**. Changes are written to `index.html` and survive a refresh.
- **M / Mermaid:** on a diagram slide (5, 7, 8, 9), edit the actual Mermaid code and choose **Apply & save**. You can download its `.mmd` source. Diagram labels, nodes and connections remain editable.
- **Layout & animation:** select an object by clicking it. Change its position, size, font size, entrance effect, duration and build order. Objects sharing a build step appear together. Choose **none** to keep an object visible throughout the slide. Enable **Drag objects** to move items. Choose **Present this slide** to preview the builds.
- **Charts:** select the chart on slide 12 or 14, then choose **Edit values** in Layout & animation. Edit its categories, series names and numeric arrays.
- **Pictures:** select an image in Layout & animation, then use **Replace image**. The replacement is embedded in the saved configuration. Use images under 6 MB.
- **Notes:** edit the current slide’s speaker notes, then choose **Apply & save**.
- **HTML backup:** download a copy of the current editable HTML. Keep it with `theme.css`, `deck.js`, `assets/` and `vendor/`.

Plain file opening works for viewing. Autosave requires the supplied local server. On a save conflict, download an HTML backup before refreshing. Previous saved HTML versions are kept in `.backups/`.

## Present

- **Right / Space:** advance exactly one slide. Every slide is fully visible immediately.
- **Left:** go back.
- **S:** speaker window with notes, next slide and timer. Allow the popup.
- **F:** fullscreen.
- **Esc:** close an editor panel, finish editing, or show the slide overview.
- **?:** Reveal.js shortcuts.

Staged reveals and Mermaid node animations are disabled for presenting without a clicker. Each slide appears fully at once. The subtle transition between slides remains. All runtime libraries, pictures and diagrams are local, so the deck works without internet.

## Toledo PDF

`Exo2EgoPose-Toledo.pdf` is the current static submission version. After any edit, export it again:

1. Choose **PDF view** in the toolbar.
2. Print to PDF in Chrome or Edge, with background graphics, no browser headers/footers, and no margins.
3. Check all 24 pages. All build steps must be visible. Notes and editing controls are excluded.

The main talk ends at slide 18. Backup slides support Q&A and do not extend the planned main talk. Both presenters should rehearse their own timing. The reviewing team leads the subsequent discussion.

## Sources and licenses

Content and figures: Shi et al., *Exo2EgoPose*, arXiv:2607.15890v2 / ACM Multimedia 2026. Original figures are credited under CC BY 4.0. Sources also appear in the slide footers, notes and references slide.

Reveal.js 5.2.1 and Mermaid 10.9.3 are bundled locally under their MIT licenses. The separate `.mmd` files are the initial diagram sources; after browser edits, the authoritative updated source is embedded in `index.html` and can be downloaded from the Mermaid editor. Charts, tables, notes, layout and animation configuration are also saved in that HTML.
