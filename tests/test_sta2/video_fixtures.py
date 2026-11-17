"""Small encoded media used by real decoder and resume regressions."""
from fractions import Fraction

import numpy as np


def write_video(path, frames=48):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            rgb[..., 0] = 255
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 8)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return (
            np.array([frame.pts for frame in container.decode(stream)], dtype=np.int64),
            stream.time_base,
        )
