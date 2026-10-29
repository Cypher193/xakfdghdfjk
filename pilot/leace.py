"""C9: erase the speed concept (LEACE) from the object tokens during generation (docs/phase3.md).

    python -m pilot.leace --model ltx2b --shard 1/4 [--block 7 --step 0] [--conds leace rand]
    python -m pilot.leace --model ltx2b --shard 1/4 --blocks 7-27 --exp c9b       # C9b: every block from 7 on
    python -m pilot.leace --model ltx2b --chain leace [--groups 5 --fit-scenes 100]  # C9c: sequential chain (also rand)
    python -m pilot.leace --model ltx2b --shard 1/4 --exp c9c                     # C9c: generate with both chains

Fits on B1 acts (all kmain scenes x seeds, pooled object tokens at one block, every captured step, target ln hspeed;
stops if the cross-fitted check misses its targets), then generates seed 0
of every (sharded) b1 scene per cond with the eraser applied to the object tokens of the input frames at that block, on
every denoising step and both guidance passes. -> runs/{exp}/{model}/{scene}_{cond}_s0/ video.mp4, acts.npz, meta.json.
--blocks: one eraser per block, each fit and checked on that block's B1 activations. (Fitting later blocks on the C9
leace runs too, whose activations erasure already shifted, did not carry over to held-out scenes: (a) 0.18 at block 8.)
--chain (C9c): each group's erasers are fit on activations generated with every upstream group's erasers on (chain());
-> runs/c9c_chain/{model}/{cond}/. --exp c9c then loads both finished chains instead of fitting.
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


def rand_kept(res):
    """C9c rand chain: rand_ok and the rand perturbation <= 0.3 (the chain fits only rand, so no leace targets)."""
    return rand_ok(res) and bool((res[1, :, 3] <= 0.3).all())


def fit_check(Xs, z, g, block, ok_fn=check_ok):
    """-> (leace eraser, seed -> rand eraser, report, targets met, crossfit res). Fit on pooled object tokens Xs
    (T steps, n, D) of one block, all captured steps with cols = step (the hook runs on every step); the report is the
    cross-fitted check (erase.crossfit), judged by ok_fn."""
    Xs = Xs.astype(np.float64)
    T, n, D = Xs.shape
    X, zz, cols = Xs.reshape(T * n, D), np.tile(z, T), np.repeat(np.arange(T), n)
    er = erase.fit(X, zz, cols)
    res = erase.crossfit(Xs, z, g)
    ok = ok_fn(res)
    rep = "\n".join([f"eraser fit n={n} clips, {len(np.unique(g))} scenes, D={D}, block {block}, {T} captured steps, "
                     f"rank {er.U.shape[0]}, Ledoit-Wolf shrinkage {er.shrink:.3f}. Cross-fitted (5-fold by scene) "
                     "R2 clean / (a) clean probe on r(x) / (b) probe retrained on r(x) / median |r(x)-x|/|x-mu|:"]
                    + [f"  step {s} {name}: " + " / ".join(f"{v:.3f}" for v in res[c, s])
                       for s in range(T) for c, name in enumerate(("leace", "rand"))]
                    + [f"targets {ok_fn.__name__} ((a),(b) <= 0.06 leace, >= min(0.6, 0.75 clean) rand where clean "
                       f">= 0.3, pert <= 0.3, leace pert <= 2x rand): {'met' if ok else 'NOT MET'}"])
    return er, lambda seed: erase.random_like(X, zz, seed, cols), rep, ok, res


def _save(path, write):
    """write(file object) to path.tmp, then rename: a crash or a push never sees half a file (sync.py ignores *.tmp)."""
    with open(f"{path}.tmp", "wb") as f:
        write(f)
    os.replace(f"{path}.tmp", path)


def chain_dir(cfg, model, cond):
    return Path(cfg["paths"]["runs"]) / "c9c_chain" / model / cond     # not runs/c9c: no clip meta.json in it


def chain(gen, root, scenes, inputs, groups, cond, ok_fn=None):
    """C9c, sequential LEACE (docs/phase3.md): per group of blocks, the fit scenes' pooled object tokens at the group's
    blocks with every upstream group's eraser hooked (seed 0, denoising stopped after the last captured step, no decode;
    root/g{k}/{scene}.npy, one file per clip), then one eraser per block of the group, fit and checked as C9b
    (fit_check; cond rand: fit to permuted z, seed 0, judged by rand_kept). Resumable: finished groups (root/g{k}.npz)
    and clips are loaded, not redone. Stops at a block that misses its targets. Writes root/chain.json at the end.
    scenes: spec dicts (id, caption, camera, gt); inputs(spec) -> (frames, box). -> {block: eraser}"""
    root, ok_fn = Path(root), ok_fn or (check_ok if cond == "leace" else rand_kept)
    group, blocks, grid = gen.cfg["blocks"][0], gen.blocks(), token_grid(gen.cfg)
    z, ids = np.log([probe.target_value(s["gt"], "hspeed") for s in scenes]), np.array([s["id"] for s in scenes])
    ers = {}
    for k, grp in enumerate(groups):
        f = root / f"g{k}.npz"
        if not f.exists():
            (root / f"g{k}").mkdir(parents=True, exist_ok=True)
            for s in scenes:
                out = root / f"g{k}" / f"{s['id']}.npy"
                if out.exists():
                    continue
                frames, box = inputs(s)
                in_wh = (s["camera"]["width"], s["camera"]["height"])
                idx = torch.from_numpy(np.flatnonzero(object_token_mask(gen.cfg, grid, box, in_wh)))
                r = gen.generate(frames, s["caption"], 0, box, in_wh, acts_only=True,
                                 hooks=[(blocks[b], erase_hook(er, idx)) for b, er in ers.items()])
                _save(out, lambda fh: np.save(fh, r["acts"][f"{group}/obj"][grp]))
                print(f"c9c chain {cond} g{k} {s['id']}: {r['meta']['wall_s']} s, upstream erasers {sorted(ers)}", flush=True)
            A = np.stack([np.load(root / f"g{k}" / f"{i}.npy") for i in ids], 2)      # (blocks of grp, T, n, D)
            keep = {}
            for j, b in enumerate(grp):
                leace, rand, rep, ok, res = fit_check(A[j], z, ids, b, ok_fn)
                print(f"c9c chain {cond} g{k}: " + rep, flush=True)
                if not ok:
                    raise SystemExit(f"c9c chain {cond}: block {b} misses its out-of-sample targets: chain stopped")
                er = leace if cond == "leace" else rand(0)
                keep |= {f"{b}/{a}": getattr(er, a) for a in ("mu", "V", "U", "shrink")} | {f"{b}/res": res}
            _save(f, lambda fh: np.savez(fh, **keep))
        ers |= load_erasers(f)
    checks = {b: np.load(root / f"g{k}.npz")[f"{b}/res"].round(4).tolist() for k, grp in enumerate(groups) for b in grp}
    (root / "chain.json").write_text(json.dumps(dict(model=gen.name, cond=cond, groups=groups, fit_scenes=ids.tolist(),
                                                     seed=0, checks=checks), indent=1))
    return ers


def load_erasers(f):
    z = np.load(f)
    return {int(b): erase.Eraser(z[f"{b}/mu"], z[f"{b}/V"], z[f"{b}/U"], float(z[f"{b}/shrink"]))
            for b in sorted({k.split("/")[0] for k in z.files})}


def load_chain(root):
    """A finished chain (chain.json is written last) -> ({block: eraser}, info for meta.json)."""
    root = Path(root)
    if not (root / "chain.json").exists():
        raise SystemExit(f"c9c: no finished chain at {root} (chain.json missing): job c9cchain-{root.name} must finish "
                         "and push first")
    info = json.loads((root / "chain.json").read_text())
    ers = {b: er for k in range(len(info["groups"])) for b, er in load_erasers(root / f"g{k}.npz").items()}
    return ers, dict(file=(root / "chain.json").as_posix(), groups=info["groups"], fit_scenes=info["fit_scenes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard")
    ap.add_argument("--block", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--blocks", help="erase at every block of a range, e.g. 7-27 (default: --block only)")
    ap.add_argument("--exp", default="c9")
    ap.add_argument("--conds", nargs="+", default=["leace", "rand"], choices=["leace", "rand"])
    ap.add_argument("--chain", choices=["leace", "rand"], help="C9c: fit this cond's chain of erasers (no videos)")
    ap.add_argument("--groups", type=int, default=5, help="C9c chain: split the blocks into this many near-equal groups")
    ap.add_argument("--fit-scenes", type=int, default=100, help="C9c chain: fit on the first N b1 scenes, seed 0")
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    mcfg = cfg["models"][a.model]
    gen = Generator(a.model, mcfg, cfg)
    blocks, group = gen.blocks(), mcfg["blocks"][0]
    allscenes = scene_list(cfg, "b1")
    inputs = lambda spec: rendered_inputs(spec, Path(cfg["paths"]["data"]) / "renders" / "kmain", mcfg["fps"],
                                          mcfg["cond_frames"])
    if a.chain:
        lo, hi = map(int, (a.blocks or "7-27").split("-"))
        groups = [g.tolist() for g in np.array_split(np.arange(lo, hi + 1), a.groups)]   # 7-27, 5: 7-11, 12-15, ...
        fit = [s for s in allscenes if (probe.target_value(s["gt"], "hspeed") or 0) > 0][:a.fit_scenes]
        root = chain_dir(cfg, a.model, a.chain)
        for s in fit:         # group 0 has no upstream eraser: the B1 seed-0 clip is that pass
            out = root / "g0" / f"{s['id']}.npy"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                acts = np.load(Path(cfg["paths"]["runs"]) / "b1" / a.model / f"{s['id']}_s0" / "acts.npz")
                _save(out, lambda fh: np.save(fh, acts[f"{group}/obj"][groups[0]]))
        chain(gen, root, fit, inputs, groups, a.chain)
        return
    if a.exp == "c9c":        # both chains must be finished (jobs c9cchain-leace, c9cchain-rand)
        chains = {c: load_chain(chain_dir(cfg, a.model, c)) for c in a.conds}
        pick = lambda cond, i: chains[cond][0]
        lo, step = min(chains[a.conds[0]][0]), None
        extra = lambda cond: dict(blocks=sorted(chains[cond][0]), chain=chains[cond][1])
    else:
        block, step = pick_block(a.model, len(blocks))
        block, step = block if a.block is None else a.block, step if a.step is None else a.step
        lo, hi = map(int, a.blocks.split("-")) if a.blocks else (block, block)
        feats, y, g = probe.load(cfg["paths"]["runs"], a.model, ["b1"], "hspeed")
        A = feats[f"{group}/obj"]                                     # (blocks, steps, clips, D)
        del feats
        erasers = {}                                                  # block -> (leace eraser, seed -> rand eraser)
        for b in range(lo, hi + 1):
            leace, rand, rep, ok, _ = fit_check(A[b], np.log(y), g, b)
            print(rep, flush=True)
            if not ok:
                raise SystemExit(f"block {b}: eraser misses its out-of-sample targets: not generating")
            erasers[b] = leace, rand
        del A
        pick = lambda cond, i: {b: lr[0] if cond == "leace" else lr[1](i) for b, lr in erasers.items()}
        extra = lambda cond: dict(blocks=list(erasers))
    grid = token_grid(mcfg)
    for spec in (shard(allscenes, a.shard) if a.shard else allscenes):
        frames, box = inputs(spec)
        in_wh = (spec["camera"]["width"], spec["camera"]["height"])
        idx = torch.from_numpy(np.flatnonzero(object_token_mask(mcfg, grid, box, in_wh)))
        for cond in a.conds:
            out = Path(cfg["paths"]["runs"]) / a.exp / a.model / f"{spec['id']}_{cond}_s0"
            if (out / "meta.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            r = gen.generate(frames, spec["caption"], 0, box, in_wh, capture=True,
                             hooks=[(blocks[b], erase_hook(er, idx)) for b, er in pick(cond, allscenes.index(spec)).items()])
            imageio.mimwrite(out / "video.partial.mp4", r["video"], fps=mcfg["fps"], quality=9, macro_block_size=1)
            os.replace(out / "video.partial.mp4", out / "video.mp4")      # push never sees a half-written video
            np.savez_compressed(out / "acts.npz", **r["acts"], **{f"vae/{k}": v for k, v in r.get("vae", {}).items()})
            meta = r["meta"] | dict(exp=a.exp, cond=cond, scene=spec["id"], block=lo, step=step, box=box,
                                    in_wh=list(in_wh)) | extra(cond)
            (out / "meta.json").write_text(json.dumps(meta, indent=1))
            print(f"{a.exp} {a.model} {spec['id']} {cond}: {meta['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
