"""Job files for models outside diffusers (MAGI-1, Self-Forcing), which run in their own venvs.

pilot.generate --export-jobs writes runs/{exp}/{model}/{scene}_s{seed}/job.json + input.npz; a runner in
scripts/ext/ reads pending jobs and writes video.mp4 + meta.json there. Stdlib, numpy and imageio only, so the
runners can import it (repo root on sys.path) without the pilot environment.
"""
import json
from pathlib import Path

import numpy as np


def export(d, job, frames):
    """frames: (cond_frames, height, width, 3) uint8 at the model fps and resolution."""
    frames = np.asarray(frames, dtype=np.uint8).reshape(-1, job["height"], job["width"], 3)
    assert len(frames) == job["cond_frames"], (len(frames), job["cond_frames"])
    np.savez_compressed(Path(d) / "input.npz", frames=frames)
    (Path(d) / "job.json").write_text(json.dumps(job, indent=1))


def pending(runs, exp, model):
    return sorted(d for d in (Path(runs) / exp / model).glob("*")
                  if (d / "job.json").exists() and not (d / "meta.json").exists())


def load(d):
    return json.loads((Path(d) / "job.json").read_text(encoding="utf-8")), np.load(Path(d) / "input.npz")["frames"]


def write(d, job, video, wall_s, **extra):
    """video: (num_frames, H, W, 3) uint8 starting with the input window (frame k = time k/fps from its start)."""
    import imageio.v2 as imageio
    video = np.asarray(video)
    assert video.dtype == np.uint8 and video.ndim == 4 and len(video) == job["num_frames"], \
        f"{job['model']}: video {video.dtype} {video.shape}, expected {job['num_frames']} uint8 frames"
    imageio.mimwrite(Path(d) / "video.mp4", video, fps=job["fps"], quality=9, macro_block_size=1)
    meta = job | dict(wall_s=round(wall_s, 1), capture=False, **_gpu()) | extra
    (Path(d) / "meta.json").write_text(json.dumps(meta, indent=1))


def _gpu():
    try:
        import torch
    except ImportError:
        return dict(gpu=None, peak_gb=None)
    if not torch.cuda.is_available():
        return dict(gpu=None, peak_gb=None)
    return dict(gpu=torch.cuda.get_device_name(0), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
