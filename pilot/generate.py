"""Run one generation experiment for one model.

    python -m pilot.generate --model vace13 --exp b1
    python -m pilot.generate --model ltx2b --exp b2nl --limit 2      # smoke test
    python -m pilot.generate --model magi45 --exp b1 --export-jobs   # external repo: job files for its runner

Writes runs/{exp}/{model}/{scene}_s{seed}/ video.mp4, meta.json, acts.npz, latent.pt. Existing runs are skipped.
--export-jobs writes job.json + input.npz there instead (pilot/jobs.py); scripts/ext/run_*.py generate from them.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from pilot import jobs
from pilot.models import Generator, _pil
from pilot.scenes import object_track, project


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def read_frames(path, times):
    """Frames nearest to the given times (s) from a video file, RGB uint8."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want = [int(round(t * fps)) for t in times]
    frames, i = {}, 0
    while len(frames) < len(set(want)):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f[..., ::-1]
        i += 1
    return [frames[k] for k in want], fps


def rendered_inputs(spec, render_dir, fps, n):
    """Input frames at exact times k/fps from the 48 fps input render, plus the object box over the window."""
    times = np.arange(n) / fps
    frames, src_fps = read_frames(Path(render_dir) / f"{spec['id']}_input.mp4", times)
    assert abs(src_fps - 48) < 0.5, "input clips must be rendered at 48 fps"
    boxes = []
    for o in spec["objects"]:
        x, z = object_track(spec, o, times)
        u, v = project(spec["camera"], x, z)
        r = o["size_m"] / 2 * spec["camera"]["fx"] / spec["camera"]["distance"]
        boxes.append([u.min() - r, v.min() - r, u.max() + r, v.max() + r])
    b = np.array(boxes)
    return frames, [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]


def scene_list(cfg, exp):
    e = cfg["experiments"][exp]
    if e["scenes"] == "real":
        return json.loads(Path("data/real/manifest.json").read_text(encoding="utf-8"))
    files = sorted((Path(cfg["paths"]["data"]) / "scenes" / e["scenes"]).glob("*.json"))
    specs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if e["scenes"] == "kmain":      # interleave families so --limit N is balanced: cv_00, ca_00, cv_01, ...
        cv = [s for s in specs if s["family"] == "cv"]
        ca = [s for s in specs if s["family"] != "cv"]
        specs = [s for pair in zip(cv, ca) for s in pair] + cv[len(ca):] + ca[len(cv):]
    return specs                    # kgrav is sorted by scene then gravity, so --limit 3k = k whole scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-capture", action="store_true", help="timing runs")
    ap.add_argument("--res", help="override resolution as HEIGHTxWIDTH, e.g. 480x832; runs go to {model}_{H}p")
    ap.add_argument("--export-jobs", action="store_true", help="write job.json + input.npz for an external runner")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mcfg, e = dict(cfg["models"][a.model]), cfg["experiments"][a.exp]
    run_name = a.model
    if a.res:
        mcfg["height"], mcfg["width"] = map(int, a.res.lower().split("x"))
        run_name = f"{a.model}_{mcfg['height']}p"
    seeds = a.seeds or e.get("seeds") or cfg["seeds"]
    scenes = scene_list(cfg, a.exp)[: a.limit or None]
    if mcfg.get("ext") and not a.export_jobs:
        raise SystemExit(f"{a.model} runs outside diffusers: --export-jobs, then scripts/ext/run_{mcfg['ext']}.py")
    gen = None if a.export_jobs else Generator(a.model, mcfg, cfg)
    sha = commit()
    for spec in scenes:
        if e["scenes"] == "real":
            n = mcfg["cond_frames"]                          # input window ends where the real continuation starts
            start = spec["start_s"] if "start_s" in spec else spec["input_end_s"] - n / mcfg["fps"]
            times = start + np.arange(n) / mcfg["fps"]
            frames, _ = read_frames(Path(spec["video"]), times)
            box, in_wh = spec["box"], tuple(spec["wh"])
            prompt = spec["caption"]
        else:
            frames, box = rendered_inputs(spec, Path(cfg["paths"]["data"]) / "renders" / e["scenes"],
                                          mcfg["fps"], mcfg["cond_frames"])
            in_wh = (spec["camera"]["width"], spec["camera"]["height"])
            prompt = spec[e["text"]] if e["text"] in spec else spec["caption"]
        if e.get("video") is False:
            frames, box = [], None
        for seed in seeds:
            out = Path(cfg["paths"]["runs"]) / a.exp / run_name / f"{spec['id']}_s{seed}"
            if (out / "meta.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            if a.export_jobs:
                job = dict(model=run_name, repo=mcfg["repo"], revision=mcfg.get("revision"), seed=seed, prompt=prompt,
                           cond_frames=len(frames), cond_seconds=len(frames) / mcfg["fps"], fps=mcfg["fps"],
                           height=mcfg["height"], width=mcfg["width"], num_frames=mcfg["num_frames"],
                           steps=mcfg.get("steps"), guidance=mcfg.get("guidance"), exp=a.exp, scene=spec["id"],
                           box=box, in_wh=list(in_wh), commit=sha)
                jobs.export(out, job, [np.asarray(f) for f in _pil(frames, mcfg["width"], mcfg["height"])])
                print(f"{run_name} {a.exp} {spec['id']} s{seed}: job written", flush=True)
                continue
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            r = gen.generate(frames, prompt, seed, box, in_wh, capture=not a.no_capture)
            imageio.mimwrite(out / "video.mp4", r["video"], fps=mcfg["fps"], quality=9, macro_block_size=1)
            if r["acts"]:
                np.savez_compressed(out / "acts.npz", **r["acts"],
                                    **{f"vae/{k}": v for k, v in r.get("vae", {}).items()})
            if r["final_latent"] is not None:
                torch.save(r["final_latent"].to(torch.bfloat16), out / "latent.pt")
            meta = r["meta"] | dict(exp=a.exp, scene=spec["id"], box=box, in_wh=list(in_wh), commit=sha,
                                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                    peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1)
                                    if torch.cuda.is_available() else None)
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"{run_name} {a.exp} {spec['id']} s{seed}: {meta['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
