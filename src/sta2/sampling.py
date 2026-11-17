"""Pure timestamp sampling, independent of video decoding and PyTorch."""

from dataclasses import dataclass
import math

import numpy as np


class UnavailableContextError(RuntimeError):
    """The recording does not contain the requested observation interval."""


@dataclass(frozen=True)
class SamplingResult:
    indices: np.ndarray
    timestamps: np.ndarray
    status: str


def sample_frame_indices(
    cutoff_seconds: float,
    source_timestamps: np.ndarray,
) -> SamplingResult:
    cutoff = float(cutoff_seconds)
    times = np.asarray(source_timestamps, dtype=np.float64)

    if not math.isfinite(cutoff):
        raise ValueError("Cutoff must be finite")

    if times.ndim != 1:
        raise ValueError("Source timestamps must be one-dimensional")

    if not np.isfinite(times).all():
        raise ValueError("Source timestamps contain nonfinite values")

    if times.size:
        if times[0] < 0:
            raise ValueError("Negative source timestamp")

        if np.any(np.diff(times) <= 0):
            raise ValueError(
                "Source timestamps must be strictly increasing"
            )

    if not times.size or not np.any(times < cutoff):
        return SamplingResult(
            indices=np.full(32, -1, dtype=np.int64),
            timestamps=np.full(32, np.nan, dtype=np.float64),
            status="no_context",
        )

    desired = (
        cutoff
        - 4.0
        + np.arange(32, dtype=np.float64) / 8.0
    )

    # Do not hide a truncated recording by repeating its final frame.
    if desired[-1] > times[-1]:
        raise UnavailableContextError(
            "Requested context extends beyond the recording"
        )

    # Rightmost source frame at or before each desired timestamp.
    indices = np.searchsorted(
        times,
        desired,
        side="right",
    ) - 1

    left_padding = desired < 0

    if np.any((indices < 0) & ~left_padding):
        raise ValueError(
            "Source timeline has an unexplained initial gap"
        )

    # Negative desired timestamps repeat the first legal source frame.
    indices[left_padding] = 0

    selected_times = times[indices]

    if not np.all(selected_times < cutoff):
        raise ValueError("Selected frame reaches the anticipation cutoff")

    if np.any(np.diff(indices) < 0):
        raise ValueError("Selected indices are not ordered")

    return SamplingResult(
        indices=indices.astype(np.int64),
        timestamps=selected_times,
        status="partial" if left_padding.any() else "full",
    )