"""Re-capture activations with pooling regions that carry no trajectory information (red-team fix X2, 2026-11-11).

The original object pool (`obj`) averages tokens over the union of the object's ground-truth boxes across the input
window, so its size and shape depend on speed and gravity. This module re-runs generation in acts_only mode (seed 0,
one captured step at 25% of denoising, same scheduler, CFG and seed as the original runs) and pools over:
  obj     the original union box (reproduction check against the original acts)
  fix     a box of constant size per model and scene family, centred on the object's frame-0 centre and shifted, not
          clipped, into the frame, so every clip has the same number of tokens
  fix_pf  the same box, pooled per latent frame (primary pool, pre-specified)
  f0      the frame-0 object box, latent frame 0 only
The VAE latent of the input frames is pooled the same ways (plus `fix_flat`: every cell in the fix box, flattened).

    python -m pilot.recapture gen --model ltx2b --set kmain --shard 1/2 [--random-init]
    python -m pilot.recapture probe --model ltx2b --set kmain --target hspeed
    python -m pilot.recapture summary
Outputs: runs/rc/{model}[_rand]/{scene}_s0/{acts.npz,meta.json}; results/rc_{model}_{set}_{target}.csv; results/rc_summary.csv
Decision rule (fixed before any data, scratchpad fix_plan.md v2): speed counts as encoded beyond a matched readout of
the input for a model if the fix_pf probe beats the best of mean-only, VAE fix_pf, VAE fix_flat and the box-position
baseline by >= 0.05 sym MRA, with a scene-bootstrap 95% CI above 0, Holm-corrected across models.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

SETS = {"kmain": "b1", "kmainf": "b1f", "kgrav": "b5"}   # scene set -> experiment (scene list, caption key)
SIZE_EXP = {"kmain": "b1", "kmainf": "b1", "kgrav": "b5"}   # fix box sizes always come from the original set
SEPARATE_FAMILIES = ("cosmos14b",)   # added after the 8-test family ran (2026-11-11): own Holm family, pre-registered
FRACTION = [0.25]
PCTL = 90                                        # fix box size: this percentile of the union-box size per family


def family(spec):
    return "cv" if spec["family"] == "cv" else "ball"


def centre_and_union(spec, n, fps):
    """Frame-0 centre (u, v) of the first object and the union box over the window, input-frame pixels."""
    from pilot.scenes import object_track, project
    t = np.arange(n) / fps
    boxes, c0 = [], None
    for o in spec["objects"]:
        x, z = object_track(spec, o, t)
        u, v = project(spec["camera"], x, z)
        r = o["size_m"] / 2 * spec["camera"]["fx"] / spec["camera"]["distance"]
        boxes.append([u.min() - r, v.min() - r, u.max() + r, v.max() + r])
        c0 = c0 if c0 is not None else (float(u[0]), float(v[0]), float(r))
    b = np.array(boxes)
    return c0, [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]


def fix_sizes(specs, n, fps):
    """{family: (w, h)} in pixels: the PCTL-th percentile of union-box width and height (target-free per family)."""
    by = {}
    for s in specs:
        _, (x0, y0, x1, y1) = centre_and_union(s, n, fps)
        by.setdefault(family(s), []).append((x1 - x0, y1 - y0))
    return {f: tuple(np.percentile(np.array(v), PCTL, axis=0)) for f, v in by.items()}


def cell_mask(grid, n_frames, centre, size, in_wh, frame0_only=False):
    """bool (F, H, W): a fixed number of cells around centre (shifted into the grid, never clipped)."""
    F, H, W = grid
    sx, sy = W / in_wh[0], H / in_wh[1]
    cw, ch = min(W, max(1, math.ceil(size[0] * sx))), min(H, max(1, math.ceil(size[1] * sy)))
    x0 = int(np.clip(int(centre[0] * sx) - cw // 2, 0, W - cw))
    y0 = int(np.clip(int(centre[1] * sy) - ch // 2, 0, H - ch))
    m = np.zeros(grid, bool)
    m[: 1 if frame0_only else n_frames, y0:y0 + ch, x0:x0 + cw] = True
    return m


def box_mask(grid, n_frames, box, in_wh):
    F, H, W = grid
    sx, sy = W / in_wh[0], H / in_wh[1]
    x0, y0, x1, y1 = box
    m = np.zeros(grid, bool)
    m[:n_frames, max(int(y0 * sy), 0):max(int(np.ceil(y1 * sy)), 1), max(int(x0 * sx), 0):max(int(np.ceil(x1 * sx)), 1)] = True
    return m


def pools_for(spec, mcfg, sizes, union, in_wh):
    """Token masks (on the transformer grid) and VAE pools (on the latent grid) for one clip."""
    from pilot.models import token_grid
    (u, v, r), fam = centre_and_union(spec, mcfg["cond_frames"], mcfg["fps"])[0], family(spec)
    n_lat = 1 + (mcfg["cond_frames"] - 1) // mcfg["vae_stride"][0]
    tg = token_grid(mcfg)
    vgrid = (n_lat, mcfg["height"] // mcfg["vae_stride"][1], mcfg["width"] // mcfg["vae_stride"][2])
    tok = {"fix": cell_mask(tg, n_lat, (u, v), sizes[fam], in_wh),
           "fix_pf": cell_mask(tg, n_lat, (u, v), sizes[fam], in_wh),
           "f0": cell_mask(tg, n_lat, (u, v), (2 * r, 2 * r), in_wh, frame0_only=True)}
    fixv = cell_mask(vgrid, n_lat, (u, v), sizes[fam], in_wh)
    vae = {"obj": (box_mask(vgrid, n_lat, union, in_wh), "mean"), "fix": (fixv, "mean"), "fix_pf": (fixv, "pf"),
           "fix_flat": (fixv, "flat"), "f0": (cell_mask(vgrid, n_lat, (u, v), (2 * r, 2 * r), in_wh, True), "mean")}
    return tok, vae


def randomise(transformer, seed=0):
    """Random-init control: same architecture, positional encoding and attention mixing, no learned weights."""
    import torch
    torch.manual_seed(seed)
    n = 0
    for m in transformer.modules():
        if hasattr(m, "reset_parameters") and m is not transformer:
            m.reset_parameters()
            n += 1
    return n


def gen(a):
    import torch
    from pilot.generate import rendered_inputs
    from pilot.models import Generator
    from pilot.shard import scene_list, shard
    cfg = yaml.safe_load(open(a.config))
    cfg["capture_fractions"] = FRACTION
    mcfg, exp = dict(cfg["models"][a.model]), SETS[a.set]
    e = cfg["experiments"][exp]
    all_specs = scene_list(cfg, exp)
    sizes = fix_sizes(scene_list(cfg, SIZE_EXP[a.set]), mcfg["cond_frames"], mcfg["fps"])
    specs = shard(all_specs[: a.limit or None], a.shard) if a.shard else all_specs[: a.limit or None]
    g = Generator(a.model, mcfg, cfg)
    if a.random_init:
        print("random init:", randomise(g.pipe.transformer), "modules reset", flush=True)
    name = a.model + ("_rand" if a.random_init else "")
    for spec in specs:
        out = Path(cfg["paths"]["runs"]) / "rc" / name / f"{spec['id']}_s0"
        if (out / "meta.json").exists():
            continue
        frames, union = rendered_inputs(spec, Path(cfg["paths"]["data"]) / "renders" / e["scenes"], mcfg["fps"], mcfg["cond_frames"])
        in_wh = (spec["camera"]["width"], spec["camera"]["height"])
        tok, vae = pools_for(spec, mcfg, sizes, union, in_wh)
        r = g.generate(frames, spec[e["text"]], 0, union, in_wh, acts_only=True, masks=tok, vae_pools=vae)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "acts.npz", **r["acts"], **{f"vae/{k}": v for k, v in r["vae"].items()})
        meta = dict(model=name, scene=spec["id"], set=a.set, seed=0, fractions=FRACTION, union=[float(x) for x in union],
                    fix_size=list(sizes[family(spec)]), wall_s=r["meta"]["wall_s"],
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{name} {spec['id']}: {meta['wall_s']} s", flush=True)


# ----------------------------------------------------------------------------------------------- probe (CPU)

def load(model, sset, target, runs="runs"):
    from pilot import probe
    rows = []
    for d in sorted((Path(runs) / "rc" / model).glob("*/acts.npz")):
        meta = json.loads((d.parent / "meta.json").read_text(encoding="utf-8"))
        if meta["set"] not in sset.split("+"):
            continue
        spec = json.loads(next(Path("data/scenes").glob(f"*/{meta['scene']}.json")).read_text(encoding="utf-8"))
        y = probe.target_value(spec["gt"], target)
        if y is None or y <= 0:
            continue
        rows.append((d, meta, spec, y))
    return rows


def nested(Xb, y, groups, folds):
    """Xb: (n_blocks, n, D). Block chosen inside each outer training fold by inner out-of-fold sym MRA. -> pred, picks"""
    from pilot import probe
    pred, picks = np.empty_like(y), []
    for tr, te in folds:
        best = max(range(Xb.shape[0]), key=lambda b: probe.score(Xb[b][tr], y[tr], groups[tr])[0])
        picks.append(best)
        pred[te] = np.exp(probe.fit(Xb[best][tr], np.log(y[tr]), groups[tr]).predict(Xb[best][te]))
    return pred, picks


def flat_pred(X, y, groups, folds):
    from pilot import probe
    pred = np.empty_like(y)
    for tr, te in folds:
        pred[te] = np.exp(probe.fit(X[tr], np.log(y[tr]), groups[tr]).predict(X[te]))
    return pred


def vae_pred(Z, rows, pool, y, groups, folds):
    """Out-of-fold ridge prediction from one VAE pool. fix_flat has one length per family (the fix box has one size per
    family), so it is fitted per family, with the same outer folds restricted to that family."""
    X = [z[f"vae/{pool}"].astype(np.float32) for z in Z]
    if len({x.shape for x in X}) == 1:
        return flat_pred(np.stack(X), y, groups, folds)
    fam = np.array([family(r[2]) for r in rows])
    pred = np.empty_like(y)
    for f in np.unique(fam):
        i = np.flatnonzero(fam == f)
        for tr, te in folds:
            tr, te = np.intersect1d(tr, i), np.intersect1d(te, i)
            if len(te):
                k = np.r_[tr, te]
                pred[te] = flat_pred(np.stack([X[j] for j in k]), y[k], groups[k],
                                     [(np.arange(len(tr)), len(tr) + np.arange(len(te)))])[len(tr):]
    return pred


def boot_ci(d, groups, n=10000, seed=0):
    ug = np.unique(groups)
    per = np.array([d[groups == g].mean() for g in ug])
    rng = np.random.default_rng(seed)
    bs = per[rng.integers(0, len(per), (n, len(per)))].mean(1)
    flips = (rng.choice([-1, 1], (n, len(per))) * per).mean(1)
    return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((np.abs(flips) >= abs(per.mean())).mean())


def probe_cmd(a):
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from pilot import metrics
    rows = load(a.model, a.set, a.target)
    y = np.array([r[3] for r in rows])
    groups = np.array([r[1]["scene"].split("_g")[0] for r in rows])
    Z = [np.load(r[0]) for r in rows]
    folds = list(GroupKFold(5).split(y, y, groups))
    groups_ = sorted({k.split("/")[0] for k in Z[0].files if not k.startswith("vae/") and "/" in k})  # VACE: 2 groups
    preds, info = {}, {}
    for pool in ("fix_pf", "fix", "f0", "obj"):
        ks = [f"{gname}/{pool}" for gname in groups_ if f"{gname}/{pool}" in Z[0].files]
        if not ks:
            continue
        # (blocks of every group, n, D) at the one captured step
        Xb = np.concatenate([np.stack([z[k][:, 0].astype(np.float32) for z in Z], 1) for k in ks])
        preds[f"acts_{pool}"], picks = nested(Xb, y, groups, folds)
        info[f"acts_{pool}"] = "blocks " + ",".join(map(str, picks))
    for pool in ("fix_pf", "fix_flat", "fix", "f0", "obj"):
        preds[f"vae_{pool}"] = vae_pred(Z, rows, pool, y, groups, folds)
    # box-position baseline: the fix box has constant size per family, so all it carries is family + frame-0 centre
    from pilot.recapture import centre_and_union
    mcfg = yaml.safe_load(open(a.config))["models"][a.model.removesuffix("_rand")]
    G = np.array([[*centre_and_union(r[2], mcfg["cond_frames"], mcfg["fps"])[0][:2], family(r[2]) == "cv"] for r in rows], float)
    preds["geo_fixbox"] = flat_pred(G, y, groups, folds)
    preds["mean"] = np.empty_like(y)
    for tr, te in folds:
        preds["mean"][te] = np.exp(np.log(y[tr]).mean())
    mra = {k: metrics.sym_mra(p, y) for k, p in preds.items()}
    base = ["mean", "vae_fix_pf", "vae_fix_flat", "geo_fixbox"]
    out = []
    for k in preds:
        row = dict(model=a.model, set=a.set, target=a.target, readout=k, sym_mra=float(mra[k].mean()), n=len(y),
                   n_scenes=len(np.unique(groups)), info=info.get(k, ""))
        if k.startswith("acts_"):
            ref = max(base, key=lambda b: mra[b].mean())
            m, lo, hi, p = boot_ci(mra[k] - mra[ref], groups)
            row |= dict(vs=ref, margin=m, lo=lo, hi=hi, p=p)
        out.append(row)
    df = pd.DataFrame(out)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/rc_{a.model}_{a.set}_{a.target}.csv", index=False)
    print(df.to_string(index=False))


def check_cmd(a):
    """Reproduction: does each re-captured obj pool identify its own original seed-0 clip (25% step)?
    Raw cosine is not a usable bar: two different scenes already reach 0.92-0.98, and bf16 drift across GPUs (the
    originals ran on A100 and 3090) gives a raw minimum of ~0.997 (LTX, first 50 clips, 2026-11-11). So per block,
    after centring on the originals' mean, every new clip must be nearest to its own original among all originals."""
    import pandas as pd
    exp, N, O, names = SETS[a.set], {}, {}, []
    for d in sorted((Path("runs") / "rc" / a.model).glob("*/acts.npz")):
        meta = json.loads((d.parent / "meta.json").read_text(encoding="utf-8"))
        orig = Path("runs") / exp / a.model / f"{meta['scene']}_s0" / "acts.npz"
        if meta["set"] != a.set or not orig.exists():
            continue
        z, o = np.load(d), np.load(orig)
        if not all(np.isfinite(o[k].astype(np.float32)).all() for k in o.files):  # dropped by probe.py too (fp16 overflow)
            print("skip non-finite original", meta["scene"])
            continue
        for k in (k for k in z.files if k.endswith("/obj") and not k.startswith("vae/")):
            N.setdefault(k, []).append(z[k][:, 0].astype(np.float32))
            O.setdefault(k, []).append(o[k][:, 0].astype(np.float32))
        names.append(meta["scene"])
    rows = []
    for k in N:
        n, o = np.array(N[k]), np.array(O[k])                      # (clips, blocks, D)
        for b in range(n.shape[1]):
            mu = o[:, b].mean(0)
            nb, ob = n[:, b] - mu, o[:, b] - mu
            nb /= np.linalg.norm(nb, axis=1, keepdims=True)
            ob /= np.linalg.norm(ob, axis=1, keepdims=True)
            S = nb @ ob.T
            raw = (n[:, b] * o[:, b]).sum(-1) / np.linalg.norm(n[:, b], axis=-1) / np.linalg.norm(o[:, b], axis=-1)
            rows.append(dict(key=k, block=b, clips=len(nb), ident_acc=float((S.argmax(1) == np.arange(len(nb))).mean()),
                             self_cos_centred=float(np.median(np.diag(S))), raw_cos_min=float(raw.min())))
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/rc_check_{a.model}_{a.set}.csv", index=False)
    ok = len(df) > 0 and df.ident_acc.min() >= a.min_ident
    print(df.describe().to_string() if len(df) else "no overlapping clips")
    print(f"REPRODUCTION {'OK' if ok else 'FAIL'} (min identification {df.ident_acc.min() if len(df) else float('nan'):.3f}, "
          f"need >= {a.min_ident}; {len(names)} clips)")
    raise SystemExit(0 if ok else 3)


INSHOW = [("ltx2b", "kmain", "hspeed"), ("ltx2b", "kgrav", "g"), ("cosmos2b", "kmain", "hspeed"), ("cosmos2b", "kgrav", "g"),
          ("vace13", "kmain", "hspeed"), ("cosmos3_nano", "kmain", "hspeed"), ("cosmos3_nano", "kgrav", "g")]


def inshow_cmd(a):
    """Input versus show on the same clips: does the video keep what the input frames already carry?
    Input = out-of-fold VAE fix_flat prediction (the best input readout in rcprobe). Show = the tracker's seed-0
    estimate (CoTracker3, results/c3_{model}_v2_nested_clips.csv), raw and calibrated (out-of-fold linear map of ln show
    to ln y, scene-grouped, as in redteam_fixes.calib). Paired on measurable seed-0 clips, scene-bootstrap CI."""
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from pilot import metrics
    out = []
    for model, sset, target in INSHOW:
        rows = load(model, sset, target)
        y = np.array([r[3] for r in rows])
        groups = np.array([r[1]["scene"].split("_g")[0] for r in rows])
        Z = [np.load(r[0]) for r in rows]
        folds = list(GroupKFold(5).split(y, y, groups))
        inp = metrics.sym_mra(vae_pred(Z, rows, "fix_flat", y, groups, folds), y)
        c = pd.read_csv(f"results/c3_{model}_v2_nested_clips.csv")
        c = c[(c.target == target) & (c.method == "nested")].set_index("dir")
        show = np.array([c.show_pred.get(f"{r[1]['scene']}_s0", np.nan) for r in rows], float)
        m = np.isfinite(show) & (show > 0)
        ys, ls, g = y[m], np.log(show[m]), groups[m]
        cal = np.empty_like(ls)
        for tr, te in GroupKFold(5).split(ls, ys, g):
            b, k = np.polyfit(ls[tr], np.log(ys[tr]), 1)
            cal[te] = b * ls[te] + k
        raw_mra, cal_mra = metrics.sym_mra(show[m], ys), metrics.sym_mra(np.exp(cal), ys)
        d_raw, d_cal = boot_ci(inp[m] - raw_mra, g), boot_ci(inp[m] - cal_mra, g)
        out.append(dict(model=model, set=sset, target=target, clips=len(y), measurable=int(m.sum()),
                        input_all=float(inp.mean()), input_meas=float(inp[m].mean()), show=float(raw_mra.mean()),
                        show_cal=float(cal_mra.mean()), input_minus_show=d_raw[0], lo=d_raw[1], hi=d_raw[2], p=d_raw[3],
                        input_minus_show_cal=d_cal[0], lo_cal=d_cal[1], hi_cal=d_cal[2], p_cal=d_cal[3]))
        print(out[-1], flush=True)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv("results/rc_inshow.csv", index=False)


def summary_cmd(a):
    import pandas as pd
    df = pd.concat([pd.read_csv(f) for f in sorted(Path("results").glob("rc_*_*_*.csv")) if "summary" not in f.name])
    df = df[df.model == a.family] if a.family else df[~df.model.isin(SEPARATE_FAMILIES)]
    prim = df[df.readout == "acts_fix_pf"].copy().sort_values("p")
    m = len(prim)
    prim["p_holm"] = [min(1.0, max(p * (m - i) for i, p in enumerate(prim.p.iloc[: j + 1]))) for j in range(m)]
    prim["pass"] = (prim.margin >= 0.05) & (prim.lo > 0) & (prim.p_holm < 0.05)
    prim.to_csv(f"results/rc_summary{'_' + a.family if a.family else ''}.csv", index=False)
    print(prim[["model", "set", "target", "sym_mra", "vs", "margin", "lo", "hi", "p_holm", "pass"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "probe", "check", "summary", "inshow"])
    ap.add_argument("--model")
    ap.add_argument("--set", default="kmain", help="gen: one of %s; probe: sets joined by +, e.g. kmain+kmainf" % list(SETS))
    ap.add_argument("--target", default="hspeed")
    ap.add_argument("--shard")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--min-ident", type=float, default=0.95)
    ap.add_argument("--family", help="summary: Holm over this model's tests only (models in SEPARATE_FAMILIES)")
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    dict(gen=gen, probe=probe_cmd, check=check_cmd, summary=summary_cmd, inshow=inshow_cmd)[a.cmd](a)


if __name__ == "__main__":
    main()
