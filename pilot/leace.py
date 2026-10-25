"""C9: erase the speed concept (LEACE) from the object tokens during generation (docs/phase3.md).

    python -m pilot.leace --model ltx2b --shard 1/4 [--block 7 --step 0] [--conds leace rand]

Fits on B1 acts (all kmain scenes x seeds, pooled object tokens at one block, every captured step, target ln hspeed;
stops if the cross-fitted check misses its targets), then generates seed 0
of every (sharded) b1 scene per cond with the eraser applied to the object tokens of the input frames at that block, on
every denoising step and both guidance passes. -> runs/c9/{model}/{scene}_{cond}_s0/ video.mp4, acts.npz, meta.json.
"""
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from pilot import erase, probe
from pilot.generate import rendered_inputs
from pilot.models import Generator, object_token_mask, token_grid
from pilot.shard import scene_list, shard
from pilot.steer import pick_block


def erase_hook(er, token_idx):
    def fn(_module, _inputs, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        h2 = h.clone()
        v = h2.view(h2.shape[0], -1, h2.shape[-1])             # (B, N, D) view of the clone
        v[:, token_idx] = er.apply(v[:, token_idx])
        return (h2, *out[1:]) if isinstance(out, (tuple, list)) else h2
    return fn


def fit_check(runs, model, group, block):
    """-> (leace eraser, seed -> rand eraser, report, targets met). Fit on B1 pooled object tokens at `block`, all
    captured steps with cols = step (the hook runs on every step); the report is the cross-fitted check (erase.crossfit)."""
    feats, y, g = probe.load(runs, model, ["b1"], "hspeed")
    Xs, z = feats[f"{group}/obj"][block].astype(np.float64), np.log(y)       # (T steps, n clips, D)
    T, n, D = Xs.shape
    X, zz, cols = Xs.reshape(T * n, D), np.tile(z, T), np.repeat(np.arange(T), n)
    er = erase.fit(X, zz, cols)
    res = erase.crossfit(Xs, z, g)
    ok = bool((res[0, :, 1:3] <= 0.05).all() and (res[1, :, 1:3] >= 0.6).all() and (res[0, :, 3] <= 0.3).all()
              and (np.abs(np.log(res[0, :, 3] / res[1, :, 3])) <= np.log(2)).all())
    rep = "\n".join([f"eraser fit n={n} clips, {len(np.unique(g))} scenes, D={D}, block {block}, {T} captured steps, "
                     f"rank {er.U.shape[0]}, Ledoit-Wolf shrinkage {er.shrink:.3f}. Cross-fitted (5-fold by scene) "
                     "R2 clean / (a) clean probe on r(x) / (b) probe retrained on r(x) / median |r(x)-x|/|x-mu|:"]
                    + [f"  step {s} {name}: " + " / ".join(f"{v:.3f}" for v in res[c, s])
                       for s in range(T) for c, name in enumerate(("leace", "rand"))]
                    + [f"targets (a),(b) <= 0.05 leace, >= 0.6 rand, pert <= 0.3, leace/rand pert within 2x: "
                       f"{'met' if ok else 'NOT MET'}"])
    return er, lambda seed: erase.random_like(X, zz, seed, cols), rep, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard")
    ap.add_argument("--block", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--conds", nargs="+", default=["leace", "rand"], choices=["leace", "rand"])
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mcfg = cfg["models"][a.model]
    gen = Generator(a.model, mcfg, cfg)
    blocks = gen.blocks()
    block, step = pick_block(a.model, len(blocks))
    block, step = block if a.block is None else a.block, step if a.step is None else a.step
    leace, rand, rep, ok = fit_check(cfg["paths"]["runs"], a.model, mcfg["blocks"][0], block)
    print(rep, flush=True)
    if not ok:
        raise SystemExit("eraser misses its out-of-sample targets: not generating")
    allscenes = scene_list(cfg, "b1")
    grid = token_grid(mcfg)
    for spec in (shard(allscenes, a.shard) if a.shard else allscenes):
        frames, box = rendered_inputs(spec, Path(cfg["paths"]["data"]) / "renders" / "kmain", mcfg["fps"],
                                      mcfg["cond_frames"])
        in_wh = (spec["camera"]["width"], spec["camera"]["height"])
        idx = torch.from_numpy(np.flatnonzero(object_token_mask(mcfg, grid, box, in_wh)))
        for cond in a.conds:
            out = Path(cfg["paths"]["runs"]) / "c9" / a.model / f"{spec['id']}_{cond}_s0"
            if (out / "meta.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            er = leace if cond == "leace" else rand(allscenes.index(spec))
            r = gen.generate(frames, spec["caption"], 0, box, in_wh, capture=True,
                             hooks=[(blocks[block], erase_hook(er, idx))])
            imageio.mimwrite(out / "video.partial.mp4", r["video"], fps=mcfg["fps"], quality=9, macro_block_size=1)
            os.replace(out / "video.partial.mp4", out / "video.mp4")      # push never sees a half-written video
            np.savez_compressed(out / "acts.npz", **r["acts"], **{f"vae/{k}": v for k, v in r.get("vae", {}).items()})
            meta = r["meta"] | dict(exp="c9", cond=cond, scene=spec["id"], block=block, step=step, box=box,
                                    in_wh=list(in_wh))
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"c9 {a.model} {spec['id']} {cond}: {meta['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
