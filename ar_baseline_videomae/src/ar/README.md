# AR baseline (EPIC-KITCHENS-100): VideoMAE + verb/noun heads

Run everything from the repo root, in the `rd_lab` environment.

## 0. One-time setup (login node, has internet)
```bash
conda activate rd_lab
pip install transformers av
export HF_HOME=$VSC_DATA/hf_cache
python -c "from transformers import VideoMAEForVideoClassification as M; M.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')"
```

## 1. Smoke test (tiny random model, 16 clips, CPU is fine)
```bash
python -m src.ar.train --model tiny-random --overfit 16 --epochs 20 --batch_size 4 --accum 1 \
  --lr 1e-3 --workers 2 --warmup_epochs 0 --label_smoothing 0 \
  --train_csv data/processed/EPIC_100_train_participant_split.csv \
  --val_csv   data/processed/EPIC_100_validation_participant_split.csv \
  --video_root $VSC_DATA/epic-kitchens/videos_640x360 \
  --out_dir $VSC_SCRATCH/runs/smoke
```
Loss must go down and training accuracy up. Any `WARNING: decode failed` line = a path/video problem.

## 2. Overfit test with the real model (GPU job)
Same command with the default model: drop `--model tiny-random`, use `--overfit 32 --epochs 30`.

## 3. Full baseline
```bash
sbatch scripts/train_ar.sbatch
```

## 4. Controlled experiments (one change at a time, new RUN name each)
```bash
RUN=frozen   sbatch scripts/train_ar.sbatch --freeze_backbone
RUN=f8       sbatch scripts/train_ar.sbatch --frames 8
RUN=noaug    sbatch scripts/train_ar.sbatch --aug none
```

## 5. Evaluate / export from a checkpoint
```bash
python -m src.ar.train --eval_only --export --val_csv <csv> --video_root <root> --out_dir <run_dir>
```
`--export_json` additionally writes an EPIC-style json; check it against the agreed evaluation format.
