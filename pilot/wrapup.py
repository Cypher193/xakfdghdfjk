"""Wrap-up analyses on the scored clips and earlier results (CPU, pod job `wrapup`): H1, H2, H5, H7, C4, B-ratio, rederr.

    python -m pilot.wrapup        # results/wrapup_{h1,h2,h5,h7,c4,bratio,rederr}_ltx_wrapup.csv + wrapup_ltx_wrapup.md

- Show per clip is `{target}_sym_mra` of each tracker's readout, reported separately: SAM2 (score.json) and CoTracker3
  (score_v2.json, deviation D3; LTX's phase-2 B2-nl and B3 clips have SAM2 scores only). "meas" averages the
  measurable clips; "strict" counts an unmeasurable clip as 0, so text-only runs that make no trackable object score low.
- Paired differences are per scene (mean over seeds), with a scene bootstrap (metrics.paired_bootstrap).
- H1 reads results/c3_{model}_v2_nested.csv (gap) and results/p6_{model}_{target}_obj.csv (margin over VAE and control).
- C4 reads the nested per-clip predictions (results/c3_{model}_v2_nested_clips.csv): per scene, the spread across seeds
  of ln(probe prediction) against that of ln(tracked value); H4 asks for probe variance < 25% of output variance.
- H5 compares the best probe on noised latents (C1-noise) with inverted ones (C1-inv), from the p6 CSVs; the proposal
  states the 0.15 threshold in R^2, the CSVs hold sym MRA, so the table reports MRA.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import metrics

MODELS = ["ltx2b", "cosmos2b", "vace13"]
RUN, RES = Path("runs"), Path("results")
TRACKERS = {"sam2": "score.json", "cotracker3": "score_v2.json"}
TARGETS = {"hspeed": "all", "g": "ca", "e": "ca"}        # target: kmain family it is defined on
rows_md = []


def scores(exp, model, score="score_v2.json"):
    out = []
    for s in sorted((RUN / exp / model).glob(f"*/{score}")):
        meta = json.loads((s.parent / "meta.json").read_text(encoding="utf-8"))
        d = json.loads(s.read_text(encoding="utf-8"))
        out.append(dict(exp=exp, model=model, dir=s.parent.name, scene=meta["scene"], seed=meta.get("seed"),
                        measurable=bool(d.get("measurable")),
                        **{k: v for k, v in d.items() if k.endswith("_sym_mra")}))
    return pd.DataFrame(out)


def per_scene(df, target, strict):
    col = f"{target}_sym_mra"
    v = df[col] if col in df else pd.Series(np.nan, df.index)
    v = v.where(df.measurable, 0.0 if strict else np.nan)
    return v.groupby(df.scene).mean().dropna()


def paired_diff(a, b):
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return dict(n_scenes=len(common))
    m, (lo, hi) = metrics.paired_bootstrap(a[common].to_numpy(), b[common].to_numpy(), np.arange(len(common)))
    return dict(n_scenes=len(common), diff=m, lo=lo, hi=hi)


def table(name, df, note):
    df.to_csv(RES / f"wrapup_{name}_ltx_wrapup.csv", index=False)       # _ltx_wrapup: sync.py pushes the job's own files
    rows_md.append(f"## {name}\n\n{note}\n\n```\n{df.to_string(index=False, float_format=lambda v: f'{v:.3f}')}\n```\n")
    print(f"\n== {name}\n{df.to_string(index=False, float_format=lambda v: f'{v:.3f}')}", flush=True)


def h1():
    rows = []
    for m in MODELS:
        f = RES / f"c3_{m}_v2_nested.csv"
        if not f.exists():
            continue
        nested = pd.read_csv(f)
        for _, r in nested[nested.method == "nested"].iterrows():
            p6 = RES / f"p6_{m}_{r.target}_obj.csv"
            d = pd.read_csv(p6) if p6.exists() else pd.DataFrame()
            margin = d[d.block >= 0].margin_over_vae_and_control.max() if len(d) else np.nan
            ok = r.gap >= 0.10 and r.gap_lo > 0 and margin >= 0.05
            rows.append(dict(model=m, target=r.target, gap=r.gap, gap_lo=r.gap_lo, gap_hi=r.gap_hi, margin=margin,
                             n_scenes=r.n_scenes, pass_h1=bool(ok)))
    df = pd.DataFrame(rows)
    table("h1", df, "H1 per model and target: nested-CV gap >= 0.10 with CI above 0, and probe margin >= 0.05 over "
                    "VAE and control. H1 needs 3 of 4 probed generators; 3 are probed here (Nano not run).")


def h2():
    rows = []
    for (m, tr) in [(m, tr) for m in MODELS for tr in TRACKERS]:
        sc = {e: scores(e, m, TRACKERS[tr]) for e in ("b1", "b2nl", "b2num", "b3")}
        for t, fam in TARGETS.items():
            for strict in (False, True):
                ps = {}
                for e, df in sc.items():
                    if len(df):
                        df = df if fam == "all" else df[df.scene.str.contains(f"_{fam}_")]
                        ps[e] = per_scene(df, t, strict)
                base = dict(model=m, tracker=tr, target=t, scoring="strict" if strict else "meas")
                if "b1" not in ps:
                    continue
                for e, ref, test in (("b2nl", "b1", "B1 - B2-nl (H2: >= 0.10)"), ("b2num", "b2nl", "B2-nl - B2-num (> 0: numbers worse)"),
                                     ("b3", "b1", "B3 - B1 (H2: |.| < 0.03)")):
                    if e in ps and ref in ps:
                        a, b = (ps[ref], ps[e]) if e != "b3" else (ps[e], ps[ref])
                        rows.append(base | dict(test=test, mra_a=a.mean(), mra_b=b.mean()) | paired_diff(a, b))
        meas = {e: df.measurable.mean() for e, df in sc.items() if len(df)}
        rows_md.append(f"measurable share, {m} {tr}: " + ", ".join(f"{e} {v:.2f}" for e, v in meas.items()) + "\n")
    table("h2", pd.DataFrame(rows), "H2: video against matched text. mra_a / mra_b are the two sides of the difference; "
                                    "meas = measurable clips only, strict = unmeasurable counted as 0.")


def jt_decreasing(groups, n_perm=5000, seed=0):
    """Jonckheere-Terpstra statistic for x(L1) > x(L2) > x(L3), one-sided permutation p."""
    def stat(gs):
        return sum((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
                   for i, a in enumerate(gs) for b in gs[i + 1:])
    gs = [np.asarray(g, float) for g in groups]
    obs, allv, sizes = stat(gs), np.concatenate(gs), [len(g) for g in gs]
    rng, hits = np.random.default_rng(seed), 0
    for _ in range(n_perm):
        p = rng.permutation(allv)
        hits += stat(np.split(p, np.cumsum(sizes)[:-1])) >= obs
    return float(obs), (hits + 1) / (n_perm + 1)


def h7():
    rows = []
    for (m, tr) in [(m, tr) for m in MODELS for tr in TRACKERS]:
        b1, l3 = scores("b1", m, TRACKERS[tr]), scores("bl3", m, TRACKERS[tr])
        if not len(l3) or not len(b1):
            continue
        for strict in (False, True):
            def clip_vals(df, col):
                v = df[col] if col in df else pd.Series(np.nan, df.index)
                return v.where(df.measurable, 0.0 if strict else np.nan).dropna().to_numpy()
            L1 = clip_vals(b1[b1.scene.str.contains("_cv_")], "hspeed_sym_mra")
            L2 = clip_vals(b1[b1.scene.str.contains("_ca_")], "g_sym_mra")
            L3 = np.concatenate([clip_vals(l3[l3.scene.str.startswith("kl3_co")], "v_sym_mra"),
                                 clip_vals(l3[l3.scene.str.startswith("kl3_ra")], "a_slope_sym_mra")])
            s, p = jt_decreasing([L1, L2, L3])
            rows.append(dict(model=m, tracker=tr, scoring="strict" if strict else "meas", L1=L1.mean(), L2=L2.mean(), L3=L3.mean(),
                             n1=len(L1), n2=len(L2), n3=len(L3), jt=s, p=p, pass_h7=bool(p < 0.05 and L1.mean() > L2.mean() > L3.mean())))
    table("h7", pd.DataFrame(rows), "H7: MRA falls from L1 (cv speed) to L2 (bounce gravity) to L3 (collision speed, ramp "
                                    "acceleration); Jonckheere-Terpstra, one-sided permutation p < 0.05.")


def c4():
    rows = []
    for m in MODELS:
        f = RES / f"c3_{m}_v2_nested_clips.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.method == "nested") & np.isfinite(d.show_pred) & (d.show_pred > 0)]
        d["base"] = d.dir.str.replace(r"_s\d+$", "", regex=True)
        for (t, exp), g in d.groupby(["target", "exp"]):
            v = g.groupby("base").agg(n=("dir", "size"), vk=("know_pred", lambda x: np.var(np.log(x), ddof=1)),
                                      vs=("show_pred", lambda x: np.var(np.log(x), ddof=1)))
            v = v[v.n >= 2]
            if len(v) < 3:
                continue
            ratio = v.vk.mean() / v.vs.mean()
            rows.append(dict(model=m, target=t, exp=exp, scenes=len(v), probe_var=v.vk.mean(), output_var=v.vs.mean(),
                             ratio=ratio, pass_h4_seed=bool(ratio < 0.25)))
    table("c4", pd.DataFrame(rows), "C4: per scene, variance across seeds of ln(probe prediction) and of ln(tracked "
                                    "value), averaged over scenes with >= 2 measurable seeds. H4 seed part: ratio < 0.25. The probes read the "
                                    "input-window tokens, which are nearly the same across seeds, so a small ratio is "
                                    "expected by design: it shows the seed, not the encoded value, moves the output.")


def h5():
    rows = []
    for m in ("ltx2b", "wan22_5b"):
        best = {}
        for k in ("noise", "inv"):
            f = RES / f"p6_{m}_{k}_hspeed_obj.csv"
            if f.exists():
                d = pd.read_csv(f)
                best[k] = d[d.block >= 0].sym_mra.max()
        if len(best) == 2:
            rows.append(dict(model=m, noise=best["noise"], inv=best["inv"], diff=best["noise"] - best["inv"],
                             pass_h5=bool(abs(best["noise"] - best["inv"]) >= 0.15)))
    table("h5", pd.DataFrame(rows), "H5: best speed probe on noised real latents (C1-noise) against inverted ones "
                                    "(C1-inv), 30 real clips each; sym MRA, threshold 0.15.")


def bratio():
    rows = []
    for m in MODELS:
        df = scores("bratio2", m)
        if not len(df):
            continue
        r = dict(model=m, clips=len(df), measurable=df.measurable.mean())
        for c in [c for c in df if c.endswith("_sym_mra")]:
            r[c] = df.loc[df.measurable, c].mean()
        faults = pd.concat([pd.read_csv(f) for f in RES.glob("faults_ltx_*rescore*.csv")])
        fb = faults[(faults.exp == "bratio2") & (faults.model == m)].drop_duplicates(["dir", "obj"])
        for k in ("vanished", "offframe", "split", "recoloured", "shrunk", "grown"):
            r[f"balls_{k}"] = (fb[k] > 0).mean() if len(fb) else np.nan
        rows.append(r)
    table("bratio", pd.DataFrame(rows), "B-ratio: measurable share, sym MRA of the clock-free ratios on measurable clips, "
                                        "and the share of tracked balls with each fault in at least one frame.")


def rederr():
    dirs = Path("configs/rederr.txt").read_text(encoding="utf-8").split()
    rows = []
    for d in dirs:
        p = Path(d)
        st = {}
        for s in ("score.json", "score_v2.json"):
            f = p / s
            st[s] = "missing" if not f.exists() else ("error" if "error" in json.loads(f.read_text()).get("reason", "") else "ok")
        rows.append(dict(dir=d, sam2=st["score.json"], cotracker3=st["score_v2.json"]))
    df = pd.DataFrame(rows)
    bad = df[(df.sam2 != "ok") | (df.cotracker3 != "ok")]
    table("rederr", bad if len(bad) else pd.DataFrame([dict(dir="all 153 listed clips have both score files without errors")]),
          f"rederr: {len(df)} listed clips; rows shown are the ones still missing a score or holding an error.")


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    for f in (rederr, h1, h2, h7, c4, h5, bratio):
        try:
            f()
        except Exception as e:                    # one failing analysis must not lose the others
            rows_md.append(f"## {f.__name__}\n\nfailed: {type(e).__name__}: {e}\n")
            print(f"{f.__name__} failed: {type(e).__name__}: {e}", flush=True)
    (RES / "wrapup_ltx_wrapup.md").write_text("# Wrap-up analyses\n\n" + "\n".join(rows_md), encoding="utf-8")
