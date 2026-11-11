"""CPU analyses requested by the 2026-11-11 red-team (scratchpad fix_plan.md v2, Part A). Reads final results from HF
(per-clip nested files, likelihood rows, acts) and writes results/rt_*.csv.
    python -m pilot.redteam_fixes all
  geo     box-geometry baselines (linear and nonlinear) and V-information of the activations beyond geometry (X2)
  calib   calibrated and seed-aggregated show, strict gap, know on all clips, measurable shares (X5, X6, E3)
  bias    signed median ln(estimate / truth) of show and know (X3)
  lik     k* bootstrap CI, share at the grid edge, preference by family and in-frame only (X4, X14)
  knowscale  equivalence of know at 2B and 14B (Cosmos-Predict2.5): per-scene know(14B) - know(2B), seeds 0-1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import metrics, probe
from pilot.recapture import centre_and_union
import yaml

MODELS = ["ltx2b", "cosmos2b", "cosmos14b", "cosmos3_nano", "vace13"]
TARGETS = {"hspeed": ["b1"], "g": ["b1", "b5"]}
CFG = yaml.safe_load(open("configs/pilot.yaml"))
OUT = Path("results")


def boot(per, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = per[rng.integers(0, len(per), (n, len(per)))].mean(1)
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def spec_of(scene):
    return json.loads(next(Path("data/scenes").glob(f"*/{scene}.json")).read_text(encoding="utf-8"))


def clips(model):
    f = OUT / f"c3_{model}_v2_nested_clips.csv"
    d = pd.read_csv(f)
    return d[d.method == "nested"]


# ---------------------------------------------------------------------------------------------- geo (X2)

def geo_features(spec, mcfg):
    (u, v, r), (x0, y0, x1, y1) = centre_and_union(spec, mcfg["cond_frames"], mcfg["fps"])
    # only what the pooling region's shape exposes; adding object size gives the exact displacement (score 1.0),
    # which no pooled feature carries, so it is not a fair baseline
    return [x1 - x0, y1 - y0, (x1 - x0) * (y1 - y0)]


def geo(a):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    rows = []
    for m in MODELS:
        mcfg = CFG["models"][m]
        d = clips(m)
        for t in TARGETS:
            x = d[d.target == t].assign(cid=lambda f: f.dir.str.rsplit("_s", n=1).str[0]).drop_duplicates("cid")
            specs = [spec_of(s) for s in x.cid]
            y = np.array([probe.target_value(s["gt"], t) for s in specs])
            g = x.scene.values
            G = np.array([geo_features(s, mcfg) for s in specs])
            lin, _ = probe.score(G, y, g)
            nl = np.empty_like(y)
            for tr, te in GroupKFold(5).split(G, y, g):
                nl[te] = np.exp(GradientBoostingRegressor(random_state=0).fit(G[tr], np.log(y[tr])).predict(G[te]))
            know = float(d[d.target == t].groupby("scene").know_mra.mean().mean())
            mean = float(metrics.sym_mra(np.full_like(y, np.exp(np.log(y).mean())), y).mean())
            rows.append(dict(model=m, target=t, n_scenes=len(y), mean_only=mean, geo_ridge=lin,
                             geo_gbm=float(metrics.sym_mra(nl, y).mean()), know_nested=know))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rt_geo.csv", index=False)


# -------------------------------------------------------------------------------------- calib (X5, X6)

def calib(a):
    from sklearn.model_selection import GroupKFold
    rows = []
    for m in MODELS:
        d = clips(m)
        for t in TARGETS:
            x = d[d.target == t].copy()
            meas = x.show_pred.notna() & (x.show_pred > 0)
            x["show_mra0"] = np.where(meas, x.show_mra, 0.0)
            per_strict = (x.groupby("scene").know_mra.mean() - x.groupby("scene").show_mra0.mean()).values
            xm = x[meas].copy()
            g = xm.scene.values
            # calibrated show: out-of-fold linear map ln y ~ ln show, scene-grouped
            ly, ls = np.log(xm.true.values), np.log(xm.show_pred.values)
            cal = np.empty_like(ly)
            for tr, te in GroupKFold(5).split(ls, ly, g):
                b, c = np.polyfit(ls[tr], ly[tr], 1)
                cal[te] = b * ls[te] + c
            xm["cal_mra"] = metrics.sym_mra(np.exp(cal), xm.true.values)
            agg = xm.groupby("scene").agg(true=("true", "first"), show=("show_pred", "median"), know=("know_mra", "mean"))
            agg["agg_mra"] = metrics.sym_mra(agg.show.values, agg.true.values)
            s = xm.groupby("scene")
            gap = boot((s.know_mra.mean() - s.show_mra.mean()).values)
            gcal = boot((s.know_mra.mean() - s.cal_mra.mean()).values)
            gagg = boot((agg.know - agg.agg_mra).values)
            rows.append(dict(model=m, target=t, clips=len(x), measurable_share=float(meas.mean()),
                             know_all=float(x.know_mra.mean()), know_meas=float(xm.know_mra.mean()),
                             show=float(xm.show_mra.mean()), show_cal=float(xm.cal_mra.mean()),
                             show_seedagg=float(agg.agg_mra.mean()),
                             gap=gap[0], gap_lo=gap[1], gap_hi=gap[2], gap_cal=gcal[0], gap_cal_lo=gcal[1],
                             gap_cal_hi=gcal[2], gap_seedagg=gagg[0], gap_seedagg_lo=gagg[1], gap_seedagg_hi=gagg[2],
                             gap_strict=boot(per_strict)[0], gap_strict_lo=boot(per_strict)[1],
                             gap_strict_hi=boot(per_strict)[2]))
            # measurable share by target tercile (selection on the outcome?)
            q = pd.qcut(x.true.rank(method="first"), 3, labels=["low", "mid", "high"])
            for lvl, share in meas.groupby(q, observed=True).mean().items():
                rows[-1][f"meas_{lvl}_tercile"] = float(share)
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rt_calib.csv", index=False)


# ------------------------------------------------------------------------------------------- bias (X3)

def bias(a):
    rows = []
    for m in MODELS:
        d = clips(m)
        for t in TARGETS:
            x = d[(d.target == t) & d.show_pred.notna() & (d.show_pred > 0)]
            for kind, col in (("show", "show_pred"), ("know", "know_pred")):
                per = x.assign(r=np.log(x[col] / x.true)).groupby("scene").r.median().values
                mu, lo, hi = boot(per)
                rows.append(dict(model=m, target=t, readout=kind, median_ln_ratio=float(np.median(per)),
                                 mean_ln_ratio=mu, lo=lo, hi=hi, n_scenes=len(per)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rt_bias.csv", index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------------------------- lik (X4, X14)

LIK = {"ltx2b": ("kvell", "kcontl", "kturnl"), "cosmos2b": ("kvelc", "kcont", "kturnc"),
       "cosmos14b": ("kvelc", "kcont", "kturnc"), "cosmos3_nano": ("kveln", "kcontn", "kturnn"),
       "vace13": ("kvelv", "kcontv", "kturnv")}


def lik_rows(prefix, s, m):
    fs = [f for f in OUT.glob(f"{prefix}_{s}_{m}_ltx_*.csv") if not f.name.endswith("_summary.csv")]
    return pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("id") if fs else None


def lik(a):
    from pilot.likelihood import implied_multiplier
    rows = []
    rng = np.random.default_rng(0)
    for m, (vel, cont, turn) in LIK.items():
        for kind, prefix, s in (("speed", "c2", vel), ("gravity", "c2g", cont), ("direction", "c2", turn)):
            d = lik_rows(prefix, s, m)
            if d is None:
                continue
            per, ks = [], []
            for base, b in d.groupby("base"):
                true = b[b.k == (0 if kind == "direction" else 1.0)]
                if true.empty:
                    continue
                alt = b[b.k != true.k.iloc[0]]
                per.append(dict(base=base, fam="cv" if "_cv_" in base else "ball",
                                pref=float((alt.loss > true.loss.iloc[0]).mean()),
                                pref_in=float((alt[alt.in_frame >= 0.999].loss > true.loss.iloc[0]).mean())
                                if "in_frame" in alt and (alt.in_frame >= 0.999).any() else np.nan))
                if kind != "direction":
                    b = b.sort_values("k")
                    ks.append(implied_multiplier(b.k.values, b.loss.values))
                    per[-1]["edge"] = ks[-1] in (b.k.min(), b.k.max())
            p = pd.DataFrame(per)
            row = dict(model=m, test=kind, bases=len(p), pref=float(p.pref.mean()),
                       pref_cv=float(p[p.fam == "cv"].pref.mean()), pref_ball=float(p[p.fam == "ball"].pref.mean()),
                       pref_in_frame=float(p.pref_in.mean()))
            if ks:
                ks = np.array(ks)
                med = [np.median(ks[rng.integers(0, len(ks), len(ks))]) for _ in range(10000)]
                row |= dict(k_star=float(np.median(ks)), k_lo=float(np.percentile(med, 2.5)),
                            k_hi=float(np.percentile(med, 97.5)), edge_share=float(p.edge.mean()))
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rt_lik.csv", index=False)


# ------------------------------------------------------------------------------------------ knowscale

EQ_MARGIN = 0.05   # the H6 equivalence margin (sym MRA), fixed before the 14B runs


def knowscale(a):
    """Does know change from Cosmos-2B to Cosmos-14B? Paired on clips both models have (same scene and seed, seeds 0-1),
    averaged per base scene (gravity triplets are one scene), scene bootstrap. Equivalent if the 95% CI lies inside
    +-EQ_MARGIN. Know is the nested-CV probe on the original (union-box) pool; the fixed-box 14B probe is separate."""
    rows = []
    two, big = clips("cosmos2b"), clips("cosmos14b")
    for t in sorted(set(two.target) & set(big.target)):
        m = two[two.target == t].merge(big[big.target == t], on="dir", suffixes=("_2b", "_14b"))
        m = m[m.dir.str.rsplit("_s", n=1).str[1].astype(int) <= 1]
        m["base"] = m.scene_2b.str.split("_g").str[0]
        both = m.show_pred_2b.notna() & (m.show_pred_2b > 0) & m.show_pred_14b.notna() & (m.show_pred_14b > 0)
        for subset, x in (("all", m), ("measurable_both", m[both])):
            if x.empty:
                continue
            per = x.groupby("base").apply(lambda f: (f.know_mra_14b - f.know_mra_2b).mean()).values
            d, lo, hi = boot(per)
            rows.append(dict(target=t, subset=subset, clips=len(x), scenes=len(per),
                             know_2b=float(x.know_mra_2b.mean()), know_14b=float(x.know_mra_14b.mean()),
                             diff=d, lo=lo, hi=hi, margin=EQ_MARGIN, equivalent=bool(lo > -EQ_MARGIN and hi < EQ_MARGIN)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rt_knowscale.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["geo", "calib", "bias", "lik", "knowscale", "all"])
    a = ap.parse_args()
    for c in (["bias", "calib", "lik", "geo"] if a.cmd == "all" else [a.cmd]):
        globals()[c](a)


if __name__ == "__main__":
    main()
