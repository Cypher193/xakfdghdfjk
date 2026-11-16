"""Acceleration family (kacc, 2026-11-16): input, know and show of |horizontal acceleration| per generator.

    python -m pilot.accel --models ltx2b cosmos2b vace13     # after `pilot.recapture probe --set kacc --target ax`

Know: the fixed-box probe margin from results/rc_{model}_kacc_ax.csv (pilot.recapture probe). Input: out-of-fold VAE
fix_flat prediction, as rc_inshow. Show: the CoTracker3 quadratic fit on the generated part (score_v2.json "ax"), raw
and calibrated out of fold (scene-grouped linear map of ln show to ln y), plus whether the video keeps the sign
(speeding up or slowing down). Paired on measurable clips with a scene bootstrap. Per-scenario rows (rd, ru, sd, br)
are descriptive only (25 scenes each). -> results/acc_summary.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pilot import metrics
from pilot.recapture import boot_ci, load, vae_pred


def show_values(model, rows, runs="runs"):
    ax, sign = [], []
    for r in rows:
        f = Path(runs) / "bacc" / model / f"{r[1]['scene']}_s0" / "score_v2.json"
        s = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        ok = s.get("measurable") and s.get("ax") is not None
        ax.append(s["ax"] if ok else np.nan)
        sign.append(float(s["speeding"] == r[2]["gt"]["speeding"]) if ok and "speeding" in s else np.nan)
    return np.array(ax, float), np.array(sign, float)


def calibrate(show, y, groups):
    ls, cal = np.log(show), np.empty_like(show)
    for tr, te in GroupKFold(5).split(ls, y, groups):
        b, k = np.polyfit(ls[tr], np.log(y[tr]), 1)
        cal[te] = np.exp(b * ls[te] + k)
    return cal


def summarise(model, runs="runs"):
    rows = load(model, "kacc", "ax", runs)
    if not rows:
        return []
    y = np.array([r[3] for r in rows])
    groups = np.array([r[1]["scene"] for r in rows])
    scn = np.array([r[2]["gt"]["scenario"] for r in rows])
    Z = [np.load(r[0]) for r in rows]
    folds = list(GroupKFold(5).split(y, y, groups))
    inp = metrics.sym_mra(vae_pred(Z, rows, "fix_flat", y, groups, folds), y)
    show, sign = show_values(model, rows, runs)
    m = np.isfinite(show) & (show > 0)
    raw = np.full_like(y, np.nan)
    cal = np.full_like(y, np.nan)
    raw[m] = metrics.sym_mra(show[m], y[m])
    if m.sum() >= 10:
        cal[m] = metrics.sym_mra(calibrate(show[m], y[m], groups[m]), y[m])
    know = Path(f"results/rc_{model}_kacc_ax.csv")
    k = pd.read_csv(know).query("readout == 'acts_fix_pf'").iloc[0] if know.exists() else {}
    out = []
    for name, sel in [("all", np.ones_like(m))] + [(s, scn == s) for s in ("rd", "ru", "sd", "br")]:
        sm = sel & m
        row = dict(model=model, scenario=name, clips=int(sel.sum()), measurable=int(sm.sum()),
                   input_all=float(inp[sel].mean()), input_meas=float(inp[sm].mean()) if sm.any() else np.nan,
                   show=float(raw[sm].mean()) if sm.any() else np.nan,
                   show_cal=float(np.nanmean(cal[sm])) if sm.any() and np.isfinite(cal[sm]).any() else np.nan,
                   sign_kept=float(np.nanmean(sign[sm])) if sm.any() else np.nan)
        if name == "all":
            row |= {f"know_{c}": k.get(c, np.nan) for c in ("sym_mra", "vs", "margin", "lo", "hi", "p")}
            if sm.sum() >= 10:
                d = boot_ci(inp[sm] - raw[sm], groups[sm])
                dc = boot_ci(inp[sm] - cal[sm], groups[sm])
                row |= dict(input_minus_show=d[0], lo=d[1], hi=d[2], p=d[3],
                            input_minus_show_cal=dc[0], lo_cal=dc[1], hi_cal=dc[2], p_cal=dc[3])
        out.append(row)
        print(row, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ltx2b", "cosmos2b", "vace13", "cosmos3_nano"])
    a = ap.parse_args()
    out = [r for m in a.models for r in summarise(m)]
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv("results/acc_summary.csv", index=False)


if __name__ == "__main__":
    main()
