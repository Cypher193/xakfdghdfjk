"""Review follow-ups on existing data (CPU job `reviewcpu`, python -m pilot.review_recompute):
(1) input versus calibrated show with the input probe trained on measurable clips only, same folds as the calibration;
(2) ln(show/truth) on seed-0 measurable clips: per-scene mean and median, sign-flip p, for Tab. bias.
Writes results/review_inshow_meas.csv and results/review_bias_seed0.csv."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pilot import metrics
from pilot.recapture import INSHOW, boot_ci, load, nested, vae_pred

rows_out, bias_out = [], []
for model, sset, target in [t for t in INSHOW if t != ("vace13", "kgrav", "g")]:
    rows = load(model, sset, target)
    y = np.array([r[3] for r in rows])
    groups = np.array([r[1]["scene"].split("_g")[0] for r in rows])
    Z = [np.load(r[0]) for r in rows]
    folds_all = list(GroupKFold(5).split(y, y, groups))
    c = pd.read_csv(f"results/c3_{model}_v2_nested_clips.csv")
    c = c[(c.target == target) & (c.method == "nested")].set_index("dir")
    show = np.array([c.show_pred.get(f"{r[1]['scene']}_s0", np.nan) for r in rows], float)
    m = np.isfinite(show) & (show > 0)
    idx = np.flatnonzero(m)
    ys, ls, g = y[m], np.log(show[m]), groups[m]
    folds = list(GroupKFold(5).split(ls, ys, g))
    cal = np.empty_like(ls)
    for tr, te in folds:
        b, k = np.polyfit(ls[tr], np.log(ys[tr]), 1)
        cal[te] = b * ls[te] + k
    cal_mra = metrics.sym_mra(np.exp(cal), ys)
    # input probe trained and tested on measurable clips only, with the calibration's folds
    Zm, rm = [Z[i] for i in idx], [rows[i] for i in idx]
    inp_m = metrics.sym_mra(vae_pred(Zm, rm, "fix_flat", ys, g, folds), ys)
    # the original input readout (trained on all clips), restricted to measurable clips, for reference
    inp_all = metrics.sym_mra(vae_pred(Z, rows, "fix_flat", y, groups, folds_all), y)[m]
    # know (fixed-box fix_pf probe, nested CV on all clips, as in Tab. main) restricted to the measurable clips
    gnames = sorted({k.split("/")[0] for k in Z[0].files if not k.startswith("vae/") and "/" in k})
    ks = [f"{gn}/fix_pf" for gn in gnames if f"{gn}/fix_pf" in Z[0].files]
    Xb = np.concatenate([np.stack([z[k][:, 0].astype(np.float32) for z in Z], 1) for k in ks])
    know = metrics.sym_mra(nested(Xb, y, groups, folds_all)[0], y)
    d = boot_ci(inp_m - cal_mra, g)
    d0 = boot_ci(inp_all - cal_mra, g)
    rows_out.append(dict(model=model, target=target, measurable=int(m.sum()), input_meas_trained=float(inp_m.mean()),
                         input_all_trained=float(inp_all.mean()), show_cal=float(cal_mra.mean()),
                         diff_meas_trained=d[0], lo=d[1], hi=d[2], p=d[3], diff_all_trained=d0[0], p_all=d0[3], know_all=float(know.mean()), know_meas=float(know[m].mean())))
    # seed-0 log ratio on measurable clips, per scene then over scenes
    lr = pd.Series(np.log(show[m] / ys)).groupby(g).mean()
    b = boot_ci(lr.values, lr.index.values)
    bias_out.append(dict(model=model, target=target, measurable=int(m.sum()), scenes=len(lr),
                         mean_ln=float(lr.mean()), median_ln=float(lr.median()), p=b[3]))
    print(rows_out[-1], bias_out[-1], flush=True)

pd.DataFrame(rows_out).to_csv("results/review_inshow_meas.csv", index=False)
pd.DataFrame(bias_out).to_csv("results/review_bias_seed0.csv", index=False)
