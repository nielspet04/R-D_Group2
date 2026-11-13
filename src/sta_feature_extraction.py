# video sampling and frozen r3d-18 feature extraction for STA model training

from __future__ import annotations
import math
from pathlib import Path
import numpy as np


def sta_segment_boundaries(
        action_start: float,
        context_seconds: float= 5.0,
        gap_seconds: float= 1.0,
        segment_seconds: int= 5,
) -> tuple[list[tuple[float, float]], list[bool]]:
    # return clamped segment boundaries and their validity mask.
    # the oberservation window is [action_start - context_seconds, action_start - gap_seconds]

    if not math.isfinite(action_start):
        raise ValueError(f"action_start must be finite, got {action_start}")
    if context_seconds <= 0:
        raise ValueError(f"context_seconds must be positive, got {context_seconds}")
    if gap_seconds < 0:
        raise ValueError(f"gap_seconds must be non-negative, got {gap_seconds}")
    if segment_seconds <= 0:
        raise ValueError(f"segment_seconds must be positive, got {segment_seconds}")

    observation_end = action_start - gap_seconds
    segment_duration = context_seconds / segment_seconds
    

    boundaries: list[tuple[float, float]] = []
    valid_mask: list[bool] = []
    for index in range(segment_seconds):
        segment_start = (
            observation_end
            - (segment_seconds - index) * segment_duration
        )
        segment_end = (
            observation_end
            - (segment_seconds - index - 1) * segment_duration
        )
        start = max(0.0, segment_start)
        stop = max(0.0, segment_end)
        valid = stop > start
        if valid:
            boundaries.append((start, stop))
        else:
            boundaries.append((0.0, 0.0))

        valid_mask.append(valid)
    return boundaries, valid_mask

def sample_segment_indices(
        fps: float,
        segment_start: float,
        segment_stop: float,
        frame_count: int= 16,
        total_video_frames: int | None= None,
         
) -> np.ndarray:
    # select frame indices from half-open interval [segment_start, segment_stop) at the given fps.
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not math.isfinite(segment_start) or segment_start < 0:
        raise ValueError(f"segment_start must be non-negative, got {segment_start}")
    if not math.isfinite(segment_stop) or segment_stop < 0:
        raise ValueError(f"segment_stop must be non-negative, got {segment_stop}")
    if segment_stop < segment_start:
        raise ValueError(f"segment_stop must be >= segment_start, got {segment_stop} < {segment_start}")
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")    

        # Include neighboring candidates, then enforce timestamp bounds directly.
    first_index = max(
        0,
        int(math.floor(segment_start * fps)) - 1,
    )
    stop_index = int(math.ceil(segment_stop * fps)) + 1

    if total_video_frames is not None:
        if total_video_frames <= 0:
            raise ValueError(f"total_video_frames must be positive, got {total_video_frames}")
        stop_index = min(stop_index, total_video_frames)

    if stop_index <= first_index:
        raise RuntimeError(f"no frames available in segment [{segment_start}, {segment_stop}) at {fps} fps")

    available = np.arange(first_index, stop_index, dtype=np.int64)
    frame_times = available.astype(np.float64) / fps
    available = available[
        (frame_times >= segment_start)
        & (frame_times < segment_stop)
    ]

    if available.size == 0:
        raise RuntimeError(
            f"no frames available in segment "
            f"[{segment_start}, {segment_stop}) at {fps} fps"
        )
    positions = np.linspace(0, len(available) - 1, num=frame_count, dtype=np.int64,)
    selected = available[positions]
    if np.any(selected.astype(np.float64) / fps >= segment_stop):
        raise RuntimeError(f"selected frame indices exceed segment stop: {selected} at {fps} fps for segment [{segment_start}, {segment_stop})")
    return selected

def sample_segment_frames(
        video_path: Path,
        segment_start: float,
        segment_stop: float,
        frame_count: int= 16,

) -> tuple[np.ndarray, np.ndarray, float]:
    # read uniformly selected RGB frames from the one video segment. 
    # returns (frames, frame_indices, fps)
    import cv2
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video file {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_segment_indices(
            fps=fps,
            segment_start=segment_start,
            segment_stop=segment_stop,
            frame_count=frame_count,
            total_video_frames=total_frames,
        )
        frames: list[np.ndarray] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames), indices, fps
    finally:
        capture.release()


def prepare_event_frames(
        video_path: Path,
        action_start: float,
) -> tuple[list[np.ndarray], list[int], list[bool], str]:
    """Read an event with one capture and one seek, retaining only sampled RGB frames.

    Each segment keeps its original ordered indices, including repeats. Captures
    belong to this call, so concurrent event workers never share decoder state.
    """
    boundaries, mask_values = sta_segment_boundaries(action_start)
    positions = [position for position, valid in enumerate(mask_values) if valid]
    if not positions:
        return [], [], mask_values, "no_context"

    import cv2

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video file {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_indices = []
        for position in positions:
            start, stop = boundaries[position]
            indices = sample_segment_indices(
                fps=fps,
                segment_start=start,
                segment_stop=stop,
                frame_count=16,
                total_video_frames=total_frames,
            )
            cutoff = action_start - 1.0
            if np.any(indices.astype(np.float64) / fps >= cutoff):
                raise RuntimeError(
                    f"selected frame indices exceed cutoff {cutoff}: {indices}"
                )
            segment_indices.append(indices)

        requested = np.unique(np.concatenate(segment_indices))
        first, last = int(requested[0]), int(requested[-1])
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError(f"failed to seek to frame {first} in {video_path}")
        requested_set = set(requested.tolist())
        frames_by_index = {}
        for frame_index in range(first, last + 1):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
            if frame_index in requested_set:
                frames_by_index[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames = [
            np.stack([frames_by_index[int(index)] for index in indices])
            for indices in segment_indices
        ]
        status = "full" if len(positions) == 5 else "partial"
        return frames, positions, mask_values, status
    finally:
        capture.release()


def build_r3d18_feature_extractor(device:str):
    # creating the frozen r3d-18 model for feature extraction
    import torch.nn as nn
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = R3D_18_Weights.DEFAULT
    encoder = r3d_18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, weights.transforms()

def encode_segment_batch(
        frame_batches: list[np.ndarray],
        encoder,
        preprocessing,
        device: str,
):
    # encode a batch of video segments into r3d-18 features
    import torch
    processed = []
    for frames in frame_batches:
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        tensor = preprocessing(tensor)
        processed.append(tensor)
    batch = torch.stack(processed).to(device)

    with torch.inference_mode():
        features = encoder(batch)
    if tuple(features.shape) != (len(frame_batches), 512):
        raise RuntimeError(f"unexpected feature shape {features.shape}, expected ({len(frame_batches)}, 512)"
        )

    return features.cpu()

def extract_event_features(
        video_path: Path,
        action_start: float,
        encoder,
        preprocessing,
        device: str,
):
    # extract r3d-18 features from the observation window of a single event
    import torch
    try:
        valid_frames, valid_positions, mask_values, status = prepare_event_frames(
            video_path, action_start
        )

        features = torch.zeros((5,512), dtype=torch.float32)
        valid_mask = torch.tensor(mask_values, dtype=torch.bool)

        if valid_frames:
            encoded = encode_segment_batch(
                valid_frames,
                encoder,
                preprocessing,
                device,
            )
            features[valid_positions] = encoded
        return features, valid_mask, status
    except (RuntimeError, OSError):
        return (
            torch.zeros((5, 512), dtype=torch.float32),
            torch.zeros(5, dtype=torch.bool),
            "video_error",
        )


def extract_event_batch(
        video_paths: list[Path],
        action_starts: list[float],
        encoder,
        preprocessing,
        device: str,
        workers: int= 0,
):
    # extract r3d-18 features from the observation windows of a batch of events
    from concurrent.futures import ThreadPoolExecutor
    import torch

    if len(video_paths) != len(action_starts):
        raise ValueError(f"video_paths and action_starts must have the same length, got {len(video_paths)} and {len(action_starts)}")
    if not video_paths:
        raise ValueError("video_paths must not be empty")
    if workers < 0:
        raise ValueError(f"workers must be non-negative, got {workers}")
    def prepare_event(arguments):
        video_path, action_start = arguments
        # Validate the event time before applying the existing video-error policy.
        sta_segment_boundaries(action_start)
        try:
            return prepare_event_frames(video_path, action_start)
        except (RuntimeError, OSError, ValueError):
            return [], [], [False] * 5, "video_error"
    arguments = list(zip(video_paths, action_starts))
    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared_events = list(executor.map(prepare_event, arguments))
    else:
        prepared_events = [prepare_event(item) for item in arguments]
    event_count = len(prepared_events)
    features = torch.zeros((event_count, 5, 512), dtype=torch.float32)
    valid_masks = torch.zeros((event_count, 5), dtype=torch.bool)
    flattened_frames: list[np.ndarray] = []
    flattened_positions: list[tuple[int, int]] = []
    statuses: list[str] = []

    for event_index, prepared in enumerate(prepared_events):
        event_frames, segment_positions, mask_values, status = prepared
        valid_masks[event_index] = torch.tensor(mask_values, dtype=torch.bool)
        statuses.append(status)
        for frames, segment_position in zip(event_frames, segment_positions):
            flattened_frames.append(frames)
            flattened_positions.append((event_index, segment_position))

    if flattened_frames:
        encoded = encode_segment_batch(
            frame_batches = flattened_frames,
            encoder=encoder,
            preprocessing=preprocessing,
            device=device,
        )
        for encoded_feature, position in zip(encoded, flattened_positions):
            event_index, segment_index = position
            features[event_index, segment_index] = encoded_feature
    return features, valid_masks, statuses
 