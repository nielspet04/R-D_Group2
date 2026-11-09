"""EPIC-KITCHENS-100 trimmed-segment dataset: decodes clips straight from the mp4 files."""
from __future__ import annotations

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import find_video, sample_times

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def decode_frames(path: str, times: np.ndarray) -> np.ndarray:
    """Return uint8 (T, H, W, 3): for every target time the first decoded frame at/after it."""
    out = [None] * len(times)
    with av.open(path) as c:
        stream = c.streams.video[0]
        c.seek(int(max(times[0], 0.0) * 1_000_000), any_frame=False, backward=True)
        j = 0
        for frame in c.decode(stream):
            t = frame.time
            if t is None or t < times[j]:
                continue
            arr = frame.to_ndarray(format="rgb24")
            while j < len(times) and t >= times[j]:
                out[j] = arr
                j += 1
            if j == len(times):
                break
    got = [o for o in out if o is not None]
    if not got:
        raise RuntimeError(f"no frames decoded from {path} at t={times[0]:.2f}s")
    last = got[-1]  # clip ends after last frame of the video: repeat last frame
    return np.stack([o if o is not None else last for o in out])


class ClipDataset(Dataset):
    def __init__(self, df, video_root, frames=16, train=False, crop=224, aug="basic",
                 seed=1, skip_missing=False):
        self.T, self.train, self.crop, self.aug, self.seed = frames, train, crop, aug, seed
        self.epoch = 0
        paths, missing = {}, []
        for vid in df["video_id"].unique():
            try:
                paths[vid] = find_video(video_root, vid)
            except FileNotFoundError:
                missing.append(vid)
        if missing:
            msg = f"{len(missing)} videos not found under {video_root}, e.g. {missing[:5]}"
            if not skip_missing:
                raise FileNotFoundError(msg + " (use --skip_missing to drop those clips)")
            print("WARNING:", msg, flush=True)
            df = df[~df["video_id"].isin(missing)].reset_index(drop=True)
        self.df = df
        self.vid = df["video_id"].tolist()
        self.paths = paths
        self.start = df["start_s"].to_numpy()
        self.stop = df["stop_s"].to_numpy()
        self.verb = df["verb_class"].to_numpy()
        self.noun = df["noun_class"].to_numpy()

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.df)

    def _transform(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        _, _, H, W = x.shape
        aug = self.train and self.aug == "basic"
        base = round(self.crop * 256 / 224)
        short = int(rng.integers(base, round(base * 320 / 256) + 1)) if aug else base
        s = short / min(H, W)
        nh, nw = round(H * s), round(W * s)
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        cs = self.crop
        if aug:
            top, left = int(rng.integers(0, nh - cs + 1)), int(rng.integers(0, nw - cs + 1))
        else:
            top, left = (nh - cs) // 2, (nw - cs) // 2
        x = x[:, :, top:top + cs, left:left + cs]
        if aug and rng.random() < 0.5:
            x = x.flip(-1)
        return (x - MEAN) / STD

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])  # reproducible per (seed, epoch, clip)
        times = sample_times(self.start[i], self.stop[i], self.T, self.train, rng)
        try:
            frames = decode_frames(self.paths[self.vid[i]], times)
        except Exception as e:  # keep a long run alive, but be loud
            print(f"WARNING: decode failed for clip {i} ({self.vid[i]}): {e}", flush=True)
            frames = np.zeros((self.T, 360, 640, 3), dtype=np.uint8)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        x = self._transform(x, rng)
        return x, int(self.verb[i]), int(self.noun[i]), i
