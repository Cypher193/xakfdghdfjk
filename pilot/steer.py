"""Steering (C7 mini, gate G6): push object tokens along the probe direction and see if generated speed changes.

    python -m pilot.steer --model ltx2b --scenes 10 --alphas -2 -1 1 2
    python -m pilot.steer --model cosmos2b --scenes 5 --alphas -2 2

Direction: ridge probe for horizontal speed ('hspeed') on runs/b1/{model}, at the block/step with the best
cross-validated score (results/p6_{model}_hspeed_obj.csv; middle block if no probe results yet).
Delta = alpha * sigma * direction, added to the object tokens of the input frames at that block, on every
denoising step and both guidance passes. Seed 0 matches the B1 run, so alpha = 0 is the B1 clip itself.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import yaml

from pilot import probe
from pilot.generate import rendered_inputs, scene_list
from pilot.models import Generator, object_token_mask, token_grid


def steer_hook(direction, sigma, token_idx, alpha):
    delta = torch.as_tensor(alpha * sigma * direction)

    def fn(_module, _inputs, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        h2 = h.clone()
        v = h2.view(h2.shape[0], -1, h2.shape[-1])             # (B, N, D) view of the clone
        v[:, token_idx] += delta.to(h2.device, h2.dtype)
        return (h2, *out[1:]) if isinstance(out, (tuple, list)) else h2
    return fn


def pick_block(model, n_blocks):
    f = Path(f"results/p6_{model}_hspeed_obj.csv")
    if f.exists():
        df = pd.read_csv(f)
        df = df[df.block >= 0]
        if len(df):
            best = df.sort_values("sym_mra").iloc[-1]
            return int(best.block), int(best.step)
    return n_blocks // 2, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-2, -1, 1, 2])
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mcfg = cfg["models"][a.model]
    group = mcfg["blocks"][0]
    gen = Generator(a.model, mcfg, cfg)
    n_blocks = len(gen.blocks())
    block, step = pick_block(a.model, n_blocks)
    w, sigma = probe.direction(cfg["paths"]["runs"], a.model, ["b1"], "hspeed", group, block, step)
    grid = token_grid(mcfg)
    for spec in scene_list(cfg, "b1")[: a.scenes]:
        frames, box = rendered_inputs(spec, Path(cfg["paths"]["data"]) / "renders" / "kmain", mcfg["fps"],
                                      mcfg["cond_frames"])
        in_wh = (spec["camera"]["width"], spec["camera"]["height"])
        idx = torch.from_numpy(np.flatnonzero(object_token_mask(mcfg, grid, box, in_wh)))
        for alpha in a.alphas:
            out = Path(cfg["paths"]["runs"]) / "steer" / a.model / f"{spec['id']}_a{alpha:+g}_s0"
            if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("block") == block:
                continue                       # a clip steered at another block (an older probe file) is redone
            out.mkdir(parents=True, exist_ok=True)
            r = gen.generate(frames, spec["caption"], 0, box, in_wh, capture=False,
                             hooks=[(gen.blocks()[block], steer_hook(w, sigma, idx, alpha))])
            imageio.mimwrite(out / "video.mp4", r["video"], fps=mcfg["fps"], quality=9, macro_block_size=1)
            meta = r["meta"] | dict(exp="steer", scene=spec["id"], alpha=alpha, block=block, step=step, sigma=sigma,
                                    box=box, in_wh=list(in_wh))
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"steer {a.model} {spec['id']} alpha {alpha:+g}: {meta['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
