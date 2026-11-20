# STA2 submission candidate from all allowed training data

The development split remains unchanged: 59,767 actions from 26 participants
for training and 7,450 actions from six different participants for validation.
The supplied results currently select epoch 5 at Action MT5R 0.0877. This is
the best observed result, not evidence that development has converged.

Final retraining uses all 67,217 supplied labelled actions from 32 participants,
with an empty validation table. It never reads hidden-test labels. There is no
final-run validation score or early stopping. A new head, action vocabulary,
optimizer and training priors are constructed; the backbone remains frozen.
The selected best.pt supplies the head settings, observed-only/future mode,
data settings, seed, learning-rate schedule and number of epochs. The original
schedule horizon is retained: selecting epoch 5 of a 20-epoch schedule does not
compress its cosine decay into five epochs. The larger dataset means more
optimizer updates per epoch than development.

## Transfer this local change to VSC

These changes are local; they have not been committed or pushed. From the Mac
project directory, transfer the changed runtime files using your normal VSC
SSH destination (replace `YOUR_VSC_LOGIN` with your account and login host, or
a configured alias). This overwrites only the listed source paths; retain any
independent VSC edits to those same files before transferring.

```bash
rsync -avR \
  scripts/train_sta2.py scripts/prepare_sta2_final.py scripts/evaluate_sta2.py \
  src/sta2/training.py src/sta2/evaluation.py \
  slurm/sta2_final_h100.slurm docs/sta2_final_training.md \
  YOUR_VSC_LOGIN:/data/leuven/397/vsc39706/R-D_Group2/
```

## Submit a new final-training job

On VSC, first list the existing development checkpoints:

```bash
cd "$VSC_DATA/R-D_Group2"
find results/sta2 -path '*/checkpoints/best.pt' -print
```

Set the path below to the best.pt from the run whose validation results you
selected (replace DEVELOPMENT_JOBID). Do not use a smoke run or last.pt.
The wrapper reads the epoch count directly from that checkpoint; if its best
epoch is 5, it runs five epochs. If development is still running, select a
stable checkpoint after finishing model selection.

```bash
cd "$VSC_DATA/R-D_Group2"
mkdir -p logs
bash -n slurm/sta2_final_h100.slurm
unset STA2_FINAL_RESUME_RUN
export STA2_SELECTION_CHECKPOINT="results/sta2/full-h100-DEVELOPMENT_JOBID/checkpoints/best.pt"
test -s "$STA2_SELECTION_CHECKPOINT"
sbatch slurm/sta2_final_h100.slurm
```

This requests the same two-H100, BF16 setup as development. New output is
`results/sta2/final-h100-NEW_JOBID`; the log is
`logs/sta2-final-h100-NEW_JOBID.out`. The selection checkpoint is read for its
settings and epoch, not used to initialize or resume the final head. Existing
development CSVs and checkpoints remain untouched. Frozen final inputs are
saved under the new run's `inputs/` directory.

Check the `training_setup` log or `run-manifest-NEW_JOBID.json` for:

- `train_events: 67217`, `validation_events: 0`.
- 32 training participants, no validation participants.
- `configuration.final_training.epochs` matching the selected development epoch.

```bash
squeue -u vsc39706
tail -f logs/sta2-final-h100-NEW_JOBID.out
```

The final checkpoint is `checkpoints/last.pt`. No best.pt is created because
there is no validation. If walltime stops the run before its selected epoch,
resume it; independent evaluation/export rejects incomplete final training.

```bash
export STA2_FINAL_RESUME_RUN="results/sta2/final-h100-NEW_JOBID"
sbatch slurm/sta2_final_h100.slurm
unset STA2_FINAL_RESUME_RUN
```

## Hidden-test inference and export

Use the existing `scripts/evaluate_sta2.py --split test --events PATH_TO_COURSE_TEST_CSV`
workflow on a GPU allocation. Pass the completed final run's `checkpoints/last.pt`
as `--checkpoint`, `inputs/train.csv` as `--train-split`, and
`inputs/validation.csv` as `--validation-split`. Keep
`data/raw/EPIC_100_train.csv` as `--annotations`. Use these same paths for
`scripts/export_sta2.py`; do not switch back to the development split files.
The export verifies checkpoint/data hashes, reconstructs the full-training
action vocabulary and priors, and checks all hidden-test IDs.

Package the resulting `submission_sta.pt` with AR `submission.pt` at the ZIP
root for the course Codabench. A final-training checkpoint alone is not a
submission, and retraining does not upload anything.

## Comparison with research

Report 8.77% (or the eventual selected development score) under our participant
split, and the final course Codabench score separately. JFAA's published 27.95%
uses the official challenge test protocol and an ensemble. A direct benchmark
comparison requires the same evaluation set, metric, observation boundary,
allowed training data and comparable model/ensemble setup. The course restricts
development to the supplied training annotations, so do not substitute the
public official validation labels or retrieve hidden-test labels to reproduce
that benchmark. No split conversion can turn the internal result into the
paper's result.
