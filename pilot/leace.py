"""C9: erase the speed concept (LEACE) from the object tokens during generation (docs/phase3.md).

    python -m pilot.leace --model ltx2b --shard 1/4 [--block 7 --step 0] [--conds leace rand]
    python -m pilot.leace --model ltx2b --shard 1/4 --blocks 7-27 --exp c9b       # C9b: every block from 7 on

Fits on B1 acts (all kmain scenes x seeds, pooled object tokens at one block, every captured step, target ln hspeed;
stops if the cross-fitted check misses its targets), then generates seed 0
of every (sharded) b1 scene per cond with the eraser applied to the object tokens of the input frames at that block, on
every denoising step and both guidance passes. -> runs/{exp}/{model}/{scene}_{cond}_s0/ video.mp4, acts.npz, meta.json.
--blocks: one eraser per block, each fit and checked on that block's B1 activations. (Fitting later blocks on the C9
leace runs too, whose activations erasure already shifted, did not carry over to held-out scenes: (a) 0.18 at block 8.)
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


def rand_ok(res):
    """rand keeps the probe: (a), (b) >= min(0.6, 0.75 x clean) at every step where clean R2 >= 0.3 (late blocks hold
    little speed, so C9's fixed 0.6 is out of reach there). Set after seeing the C9b block 7-27 checks (docs/phase3.md)."""
    clean, kept = res[1, :, 0], res[1, :, 1:3]
    return bool(((kept >= np.minimum(0.6, 0.75 * clean)[:, None]).all(1) | (clean < 0.3)).all())


def check_ok(res):
    """leace (a), (b) <= 0.06; rand_ok; both perturbations <= 0.3, leace at most 2x rand (one-sided: a smaller leace edit
    only makes 'information used' harder to show; at blocks >= 20 leace is ~0.11 vs rand ~0.23)."""
    lp, rp = res[0, :, 3], res[1, :, 3]
    return bool((res[0, :, 1:3] <= 0.06).all() and rand_ok(res) and (lp <= 0.3).all() and (rp <= 0.3).all()
                and (lp <= 2 * rp).all())


def fit_check(Xs, z, g, block):
    """-> (leace eraser, seed -> rand eraser, report, targets met). Fit on B1 pooled object tokens Xs (T steps, n, D) of
    one block, all captured steps with cols = step (the hook runs on every step); the report is the cross-fitted check
    (erase.crossfit)."""
    Xs = Xs.astype(np.float64)
    T, n, D = Xs.shape
    X, zz, cols = Xs.reshape(T * n, D), np.tile(z, T), np.repeat(np.arange(T), n)
    er = erase.fit(X, zz, cols)
    res = erase.crossfit(Xs, z, g)
    ok = check_ok(res)
    rep = "\n".join([f"eraser fit n={n} clips, {len(np.unique(g))} scenes, D={D}, block {block}, {T} captured steps, "
                     f"rank {er.U.shape[0]}, Ledoit-Wolf shrinkage {er.shrink:.3f}. Cross-fitted (5-fold by scene) "
                     "R2 clean / (a) clean probe on r(x) / (b) probe retrained on r(x) / median |r(x)-x|/|x-mu|:"]
                    + [f"  step {s} {name}: " + " / ".join(f"{v:.3f}" for v in res[c, s])
                       for s in range(T) for c, name in enumerate(("leace", "rand"))]
                    + [f"targets (a),(b) <= 0.06 leace, >= min(0.6, 0.75 clean) rand where clean >= 0.3, pert <= 0.3, "
                       "leace pert <= 2x rand: "
                       f"{'met' if ok else 'NOT MET'}"])
    return er, lambda seed: erase.random_like(X, zz, seed, cols), rep, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard")
    ap.add_argument("--block", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--blocks", help="erase at every block of a range, e.g. 7-27 (default: --block only)")
    ap.add_argument("--exp", default="c9")
    ap.add_argument("--conds", nargs="+", default=["leace", "rand"], choices=["leace", "rand"])
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mcfg = cfg["models"][a.model]
    gen = Generator(a.model, mcfg, cfg)
    blocks = gen.blocks()
    block, step = pick_block(a.model, len(blocks))
    block, step = block if a.block is None else a.block, step if a.step is None else a.step
    lo, hi = map(int, a.blocks.split("-")) if a.blocks else (block, block)
    feats, y, g = probe.load(cfg["paths"]["runs"], a.model, ["b1"], "hspeed")
    A = feats[f"{mcfg['blocks'][0]}/obj"]                         # (blocks, steps, clips, D)
    del feats
    erasers = {}                                                  # block -> (leace eraser, seed -> rand eraser)
    for b in range(lo, hi + 1):
        leace, rand, rep, ok = fit_check(A[b], np.log(y), g, b)
        print(rep, flush=True)
        if not ok:
            raise SystemExit(f"block {b}: eraser misses its out-of-sample targets: not generating")
        erasers[b] = leace, rand
    del A
    allscenes = scene_list(cfg, "b1")
    grid = token_grid(mcfg)
    for spec in (shard(allscenes, a.shard) if a.shard else allscenes):
        frames, box = rendered_inputs(spec, Path(cfg["paths"]["data"]) / "renders" / "kmain", mcfg["fps"],
                                      mcfg["cond_frames"])
        in_wh = (spec["camera"]["width"], spec["camera"]["height"])
        idx = torch.from_numpy(np.flatnonzero(object_token_mask(mcfg, grid, box, in_wh)))
        for cond in a.conds:
            out = Path(cfg["paths"]["runs"]) / a.exp / a.model / f"{spec['id']}_{cond}_s0"
            if (out / "meta.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            ers = {b: lr[0] if cond == "leace" else lr[1](allscenes.index(spec)) for b, lr in erasers.items()}
            r = gen.generate(frames, spec["caption"], 0, box, in_wh, capture=True,
                             hooks=[(blocks[b], erase_hook(er, idx)) for b, er in ers.items()])
            imageio.mimwrite(out / "video.partial.mp4", r["video"], fps=mcfg["fps"], quality=9, macro_block_size=1)
            os.replace(out / "video.partial.mp4", out / "video.mp4")      # push never sees a half-written video
            np.savez_compressed(out / "acts.npz", **r["acts"], **{f"vae/{k}": v for k, v in r.get("vae", {}).items()})
            meta = r["meta"] | dict(exp=a.exp, cond=cond, scene=spec["id"], block=lo, blocks=list(erasers), step=step,
                                    box=box, in_wh=list(in_wh))
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"{a.exp} {a.model} {spec['id']} {cond}: {meta['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
