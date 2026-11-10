# EPIC-KITCHENS-100 action understanding

R&D Experience project for the 2026-2027 HCVISION course. The project covers
two video-understanding tasks on EPIC-KITCHENS-100:

- Action Recognition (AR): predict the verb and noun of a trimmed action.
- Short-Term Action Anticipation (STA): predict the next action using video up
  to one second before the action starts.

## Project structure

```text
.
|-- data/
|   |-- raw/               # Original EPIC-KITCHENS data; never edit in place
|   `-- processed/         # Reproducible derived data and metadata
|-- docs/
|   `-- reference/         # Course brief, dataset guide, and lecture slides
|-- notebooks/             # Exploration, prototyping, and visualisation
|-- results/
|   |-- figures/           # Generated plots and figures
|   `-- tables/            # Generated result tables
|-- scripts/               # Standalone entry points and pipelines
|-- src/                   # Reusable, testable Python code
|-- .gitignore
|-- README.md
`-- requirements.txt
```

Keep raw data immutable. Put reusable logic in `src/`, then import it from
notebooks or scripts. Generated outputs belong in `results/` and should be
recreatable from committed code and configuration.

## Environment setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

## Download the videos

Authenticate with your KU Leuven R-number and download the downscaled videos:

```bash
python scripts/download_kitchen_mango.py
```

The downloader stores the videos under
`data/raw/EPIC-KITCHENS/videos_640x360/`. Existing files are skipped. Store the
provided annotation CSV files under `data/raw/EPIC-KITCHENS/annotations/`.

## Working conventions

- Name notebooks in execution order, for example
  `01-initial-data-exploration.ipynb`.
- Refactor stable notebook logic into functions in `src/`.
- Add a script in `scripts/` for each reproducible training or evaluation
  pipeline.
- Commit after each logical change with a descriptive message.
- Never commit raw videos, processed datasets, model checkpoints, credentials,
  or generated results.

## STA Model 0 frequency baseline

The first STA baseline predicts the training-set verb and noun frequencies for
every validation event. It does not read video. Its purpose is to verify the
participant split, score-vector format, valid verb-noun action mapping, and
class-mean Top-5 recall before training a video model.

Create the participant split if it is not already present:

```bash
python -m src.data_processing
```

Run the hand-checkable metric tests and evaluate Model 0:

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_sta_model_0.py
```

The generated metric file is written to
`results/tables/sta_model_0_frequency.json`. Generated results are ignored by
Git and should be reproducible from the committed code.

