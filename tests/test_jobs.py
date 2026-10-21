"""CPU check of the external-model job files: export (pilot.generate --export-jobs) and the runners' writer.
    python -m tests.test_jobs"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

from pilot import jobs

cfg = yaml.safe_load(open("configs/pilot.yaml"))
spec_file = Path("data/scenes/kmain/kmain_cv_00.json")
spec = json.loads(spec_file.read_text(encoding="utf-8"))
tmp = Path(tempfile.mkdtemp())
try:
    (tmp / "data/scenes/kmain").mkdir(parents=True)
    shutil.copy(spec_file, tmp / "data/scenes/kmain")
    (tmp / "data/renders/kmain").mkdir(parents=True)
    w, h = spec["camera"]["width"], spec["camera"]["height"]
    src = [np.full((h, w, 3), 4 * i, np.uint8) for i in range(64)]       # 48 fps render, grey level = 4 x index
    imageio.mimwrite(tmp / f"data/renders/kmain/{spec['id']}_input.mp4", src, fps=48, quality=10, macro_block_size=1)
    cfg["paths"] = {"data": str(tmp / "data"), "runs": str(tmp / "runs"), "results": str(tmp / "results")}
    (tmp / "pilot.yaml").write_text(yaml.safe_dump(cfg))

    for model in ("selfforcing", "magi45"):
        m = cfg["models"][model]
        subprocess.run([sys.executable, "-m", "pilot.generate", "--config", str(tmp / "pilot.yaml"), "--model", model,
                        "--exp", "b1", "--limit", "1", "--seeds", "0", "--export-jobs"], check=True)
        (d,) = jobs.pending(tmp / "runs", "b1", model)
        job, frames = jobs.load(d)
        print(model, d.name, frames.shape, {k: job[k] for k in ("fps", "cond_frames", "num_frames", "cond_seconds")})
        assert frames.shape == (m["cond_frames"], m["height"], m["width"], 3) and frames.dtype == np.uint8
        assert job["fps"] == m["fps"] and job["num_frames"] == m["num_frames"] and job["seed"] == 0
        assert job["cond_seconds"] == m["cond_frames"] / m["fps"] and job["scene"] == spec["id"]
        step = 48 // m["fps"]                                              # input frame k = render frame k * 48 / fps
        assert np.allclose(frames.mean((1, 2, 3)), 4 * step * np.arange(m["cond_frames"]), atol=3), model

        gen_meta = {"model", "repo", "revision", "seed", "prompt", "cond_frames", "cond_seconds", "fps", "height",
                    "width", "num_frames", "steps", "guidance", "wall_s", "exp", "scene", "box", "in_wh", "commit",
                    "gpu", "peak_gb"}                                      # pilot.generate's meta, minus token_grid
        video = np.concatenate([frames, np.zeros((m["num_frames"] - len(frames), *frames.shape[1:]), np.uint8)])
        try:
            jobs.write(d, job, video[:-1], 1.0)
            raise RuntimeError("short video accepted")
        except AssertionError:
            pass
        jobs.write(d, job, video, 12.34, ext_commit="abc")
        meta = json.loads((d / "meta.json").read_text())
        assert gen_meta <= meta.keys() and meta["capture"] is False and meta["wall_s"] == 12.3, meta.keys()
        cap = cv2.VideoCapture(str(d / "video.mp4"))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        ok, f0 = cap.read()
        cap.release()
        assert n == m["num_frames"] and abs(fps - m["fps"]) < 0.01 and abs(f0.mean() - frames[0].mean()) < 3, (n, fps)
        assert jobs.pending(tmp / "runs", "b1", model) == []                   # done: not pending any more
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("job checks passed")
