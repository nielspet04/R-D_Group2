"""Decode only requested legal frames, verified by presentation timestamp."""

from pathlib import Path

import numpy as np


class VideoReadError(RuntimeError):
    pass


def decode_requested_frames(
    path,
    pts,
    time_base,
    indices,
    cutoff_seconds,
):
    import av

    requested_pts = {
        int(value)
        for value in pts[np.unique(indices)]
    }

    first_pts = min(requested_pts)
    last_pts = max(requested_pts)

    frames_by_pts = {}

    try:
        with av.open(str(Path(path))) as container:
            if not container.streams.video:
                raise VideoReadError("No video stream")

            stream = container.streams.video[0]

            if stream.time_base != time_base:
                raise VideoReadError(
                    "Decoder time base differs from metadata"
                )

            container.seek(
                first_pts,
                stream=stream,
                backward=True,
                any_frame=False,
            )

            for frame in container.decode(stream):
                if frame.pts is None:
                    raise VideoReadError(
                        "Decoded frame has no timestamp"
                    )

                current_pts = int(frame.pts)

                if current_pts > last_pts:
                    break

                if current_pts in requested_pts:
                    actual_seconds = float(
                        current_pts * time_base
                    )

                    if actual_seconds >= cutoff_seconds:
                        raise VideoReadError(
                            "Decoded frame reaches cutoff"
                        )

                    # Explicitly convert BGR decoder output to RGB.
                    bgr = frame.to_ndarray(format="bgr24")

                    frames_by_pts[current_pts] = (
                        bgr[..., ::-1].copy()
                    )

                if len(frames_by_pts) == len(requested_pts):
                    break

    except (
        OSError,
        EOFError,
        av.error.FFmpegError,
    ) as error:
        raise VideoReadError(
            f"Cannot decode {path}: {error}"
        ) from error

    if len(frames_by_pts) != len(requested_pts):
        raise VideoReadError(
            f"Missing requested frames in {path}"
        )

    # Reconstruct original order, including repeated padding frames.
    return np.stack([
        frames_by_pts[int(pts[index])]
        for index in indices
    ])
    