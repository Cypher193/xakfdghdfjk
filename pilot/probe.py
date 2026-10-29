"""Know readout: ridge probes on pooled activations, scored with symmetric MRA on held-out scenes.

    python -m pilot.probe --model vace13 --exps b1 b5 --target g
    python -m pilot.probe --model ltx2b --exps b1 --target v

Targets are the true values (ground truth), so "know" = can the true value be read from the model.
Probes predict ln(target); splits are by scene (GroupKFold), so seeds of one scene never straddle train/test.
Baselines with the same pipeline: VAE latents of the input frames, and a control task (Hewitt & Liang):
each scene gets a random target drawn from the real targets, fixed per scene. Selectivity = real - control.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pilot import metrics

ALPHAS = np.logspace(-2, 4, 13)


def target_value(gt, target):
    """'hspeed' = horizontal speed for every kmain scene (cv: v, bounce: vx); other targets read directly."""
    return gt.get("vx", gt.get("v")) if target == "hspeed" else gt.get(target)


def load(runs, model, exps, target, scene_root="data/scenes"):
    """-> features dict {name: (n, D) or (n_blocks, n_steps, n, D)}, y (n,), groups (n,)."""
    feats, ys, groups = {}, [], []
    for exp in exps:
        for d in sorted((Path(runs) / exp / model).glob("*/acts.npz")):
            meta = json.loads((d.parent / "meta.json").read_text(encoding="utf-8"))
            spec_file = next(Path(scene_root).glob(f"*/{meta['scene']}.json"), None)
            if spec_file is None:
                continue
            y = target_value(json.loads(spec_file.read_text(encoding="utf-8"))["gt"], target)
            if y is None or y <= 0:
                continue
            z = np.load(d)
            for k in z.files:
                if k != "steps":
                    feats.setdefault(k, []).append(z[k].astype(np.float32))
            ys.append(y)
            groups.append(meta["scene"].split("_g")[0])     # gravity variants of one scene form one group
    n = len(ys)
    feats = {k: np.stack(v, axis=-2) for k, v in feats.items() if len(v) == n}   # sample axis before D
    return feats, np.array(ys), np.array(groups)


def fit(X, y_log, groups, n_splits=3):
    """Ridge on standardised X, penalty chosen by scene-split CV (GroupKFold, MSE of ln y)."""
    return GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                        cv=GroupKFold(min(n_splits, len(np.unique(groups)))),
                        scoring="neg_mean_squared_error").fit(X, y_log, groups=groups)


def cv_predict(X, y_log, groups, n_splits=5):
    """Out-of-fold predictions of ln(y); ridge penalty chosen inside each fold, also split by scene."""
    pred = np.empty_like(y_log)
    k = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(X, y_log, groups):
        pred[te] = fit(X[tr], y_log[tr], groups[tr]).predict(X[te])
    return pred


def score(X, y, groups):
    p = np.exp(cv_predict(X, np.log(y), groups))
    return float(metrics.sym_mra(p, y).mean()), p


def control_targets(y, groups, seed=0):
    """Random target per scene, drawn from the real targets (memorising scene identity is all a probe can do)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    lookup = dict(zip(ug, rng.choice(y, len(ug))))
    return np.array([lookup[g] for g in groups])


def run(runs, model, exps, target, out_csv, pool="obj"):
    feats, y, g = load(runs, model, exps, target)
    if len(y) < 10:
        raise SystemExit(f"only {len(y)} samples with target {target}")
    rows = []
    y_ctrl = control_targets(y, g)
    # floor: predicting the training mean (targets cluster, e.g. most g = 9.81, so this is not 0)
    rows.append(dict(feature="mean", block=-2, step=-1, sym_mra=score(np.zeros((len(y), 1)), y, g)[0]))
    vae = feats.get(f"vae/{pool}")
    if vae is not None:
        rows.append(dict(feature=f"vae/{pool}", block=-1, step=-1, sym_mra=score(vae, y, g)[0]))
    for name, A in feats.items():
        if not name.endswith(f"/{pool}") or name.startswith("vae/"):
            continue
        n_blocks, n_steps = A.shape[:2]
        for b in range(n_blocks):
            for s in range(n_steps):
                X = A[b, s]
                rows.append(dict(feature=name, block=b, step=s, sym_mra=score(X, y, g)[0],
                                 control=score(X, y_ctrl, g)[0]))
        print(f"{name}: done {n_blocks} blocks x {n_steps} steps", flush=True)
    df = pd.DataFrame(rows)
    df["selectivity"] = df.sym_mra - df.control
    vae_mra = df.loc[df.block == -1, "sym_mra"].max() if (df.block == -1).any() else np.nan
    df["margin_over_vae_and_control"] = df.sym_mra - np.fmax(vae_mra, df.control)
    df.to_csv(out_csv, index=False)
    best = df[df.block >= 0].sort_values("margin_over_vae_and_control").iloc[-1]
    print(f"{model} {target}: n={len(y)} scenes={len(np.unique(g))} mean-only={df.sym_mra.iloc[0]:.3f} VAE={vae_mra:.3f} "
          f"best block {best.block} step {best.step}: MRA {best.sym_mra:.3f}, control {best.control:.3f}, "
          f"margin {best.margin_over_vae_and_control:.3f}")
    return df


def direction(runs, model, exps, target, group, block, step, pool="obj"):
    """Probe direction for steering: ridge on all samples at one block/step, mapped back to raw activation space.
    Returns (unit vector, std of projections onto it); +direction raises the predicted ln(target)."""
    feats, y, g = load(runs, model, exps, target)
    X = feats[f"{group}/{pool}"][block, step]
    scaler, ridge = fit(X, np.log(y), g, n_splits=5).best_estimator_.named_steps.values()
    w = ridge.coef_ / scaler.scale_
    w = w / np.linalg.norm(w)
    return w.astype(np.float32), float(np.std(X @ w))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--exps", nargs="+", default=["b1"])
    ap.add_argument("--target", required=True, help="v, vx, g or e")
    ap.add_argument("--pool", default="obj", choices=["obj", "all"])
    a = ap.parse_args()
    Path("results").mkdir(exist_ok=True)
    run(a.runs, a.model, a.exps, a.target, f"results/p6_{a.model}_{a.target}_{a.pool}.csv", a.pool)
