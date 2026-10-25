"""Phase 3 CPU analyses for LTX-2B (docs/phase3.md: C5, C6, C9 readout, statistics and gates report).

    python -m pilot.analysis fetch      # download what the analyses need from HF (never video.mp4)
    python -m pilot.analysis gap        # know-show gap per target (B1 hspeed; B1+B5 g, e), Holm across targets
    python -m pilot.analysis c5         # final-latent probe vs mid-network probe vs show
    python -m pilot.analysis c6         # conditional V-information beyond VAE, online MDL, p7 noise vs inv
    python -m pilot.analysis c9         # LEACE readout (skips cleanly if runs/c9 is absent)
    python -m pilot.analysis gates      # results/gates_ltx2b.md
    python -m pilot.analysis all [--no-fetch]
    python -m pilot.analysis selftest

Outputs: results/c3_ltx2b_*.csv / .json / gates_ltx2b.md. Probe targets are ground truth, splits are by scene.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pilot import metrics, probe

M = "ltx2b"
REPO = "CypherNotHere/cvpr-vail-pilot"
OUT = "results/c3_ltx2b_"
TARGET_EXPS = {"hspeed": ["b1"], "g": ["b1", "b5"], "e": ["b1", "b5"]}
RUN = Path("runs")


# ---------------------------------------------------------------- data access
def fetch():
    from huggingface_hub import snapshot_download
    pats = ["data/scenes/**"]
    for e in "b1 b5 b2nl b3 b1real bratio steer c9".split():
        pats += [f"runs/{e}/{M}/*/{f}" for f in ("meta.json", "score.json", "acts.npz", "latent.pt")]
    pats += [f"runs/p7/{M}_*/*/{f}" for f in ("meta.json", "acts.npz")]
    pats += [f"results/p6_{M}_*.csv", "results/show_ltx_*.csv"]
    snapshot_download(REPO, repo_type="dataset", local_dir=".", allow_patterns=pats, max_workers=8)
    print("fetched", REPO)


def clips(exps, target, model=M):
    """Clips probe.load keeps, in its order: dicts with exp, dir, meta, y, scene (group)."""
    out = []
    for exp in exps:
        for d in sorted((RUN / exp / model).glob("*/acts.npz")):
            meta = json.loads((d.parent / "meta.json").read_text(encoding="utf-8"))
            spec = next(Path("data/scenes").glob(f"*/{meta['scene']}.json"), None)
            if spec is None:
                continue
            y = probe.target_value(json.loads(spec.read_text(encoding="utf-8"))["gt"], target)
            if y is not None and y > 0:
                out.append(dict(exp=exp, dir=d.parent.name, meta=meta, y=y, scene=meta["scene"].split("_g")[0]))
    return out


def load(exps, target, model=M):
    feats, y, g = probe.load(str(RUN), model, exps, target)
    cl = clips(exps, target, model)
    assert len(cl) == len(y) and np.allclose([c["y"] for c in cl], y), "clip order mismatch with probe.load"
    return cl, feats, y, g


def show(exp, model=M):
    rows = []
    for s in sorted((RUN / exp / model).glob("*/score.json")):
        m = json.loads((s.parent / "meta.json").read_text(encoding="utf-8"))
        rows.append(dict(exp=exp, dir=s.parent.name, scene=m["scene"].split("_g")[0], seed=m["seed"],
                         cond=m.get("cond"), alpha=m.get("alpha")) | json.loads(s.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    if len(df):
        df["measurable"] = df["measurable"].fillna(False).astype(bool)
    return df


def best(target, tag=""):
    """Best mid-network probe row (max sym MRA over blocks and steps) of results/p6_ltx2b{tag}_{target}_obj.csv."""
    d = pd.read_csv(f"results/p6_{M}{tag}_{target}_obj.csv")
    d = d[(d.block >= 0) & d.feature.str.endswith("/obj") & ~d.feature.str.startswith("vae/")]
    return d.loc[d.sym_mra.idxmax()]


def save(df, name):
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"{OUT}{name}.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def sign_flip_p(per, n=20000, seed=0):
    """Two-sided sign-flip test p-value that the mean of per-scene differences is 0."""
    per = np.asarray(per, float)
    m = (np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(per))) * per).mean(1)
    return float((np.abs(m) >= abs(per.mean()) - 1e-12).mean())


# ---------------------------------------------------------------- gap
def gap():
    rows, per_clip = [], []
    for t, exps in TARGET_EXPS.items():
        cl, feats, y, g = load(exps, t)
        b = best(t)
        pred = np.exp(probe.cv_predict(feats[b.feature][int(b.block), int(b.step)], np.log(y), g))
        sh = pd.concat([show(e) for e in exps]).set_index(["exp", "dir"])
        sv = np.array([sh[t].get((c["exp"], c["dir"]), np.nan) if t in sh and sh["measurable"].get((c["exp"], c["dir"]), False)
                       else np.nan for c in cl], float)
        ok = np.isfinite(sv) & (sv > 0)
        know = metrics.sym_mra(pred, y)
        shw = np.zeros(len(y))
        shw[ok] = metrics.sym_mra(sv[ok], y[ok])
        per_clip.append(pd.DataFrame(dict(target=t, exp=[c["exp"] for c in cl], dir=[c["dir"] for c in cl], scene=g, true=y,
                                          know_pred=pred, show_pred=sv, know_mra=know, show_mra=np.where(ok, shw, np.nan))))
        row = dict(target=t, block=int(b.block), step=int(b.step), n_clips_all=len(y), n_clips=int(ok.sum()),
                   n_scenes=len(np.unique(g[ok])), know_mra_all=know.mean())
        if ok.sum() >= 4 and len(np.unique(g[ok])) >= 3:
            m, (lo, hi) = metrics.paired_bootstrap(know[ok], shw[ok], g[ok])
            per = np.array([np.mean(know[ok][g[ok] == k] - shw[ok][g[ok] == k]) for k in np.unique(g[ok])])
            row |= dict(know_mra=know[ok].mean(), show_mra=shw[ok].mean(), gap=m, gap_lo=lo, gap_hi=hi,
                        sd_scene=per.std(ddof=1), p=sign_flip_p(per))
        rows.append(row)
    df = pd.DataFrame(rows)
    pv = df.p.fillna(1.0).to_numpy() if "p" in df else np.ones(len(df))
    df["p_holm"] = metrics.holm(pv)
    pd.concat(per_clip).to_csv(f"{OUT}gap_clips.csv", index=False)
    print("know-show gap (paired over measurable clips; gap = know - show, sym MRA; CI = scene bootstrap)")
    save(df, "gap")
    return df


# ---------------------------------------------------------------- C5
def pooled_latent(c):
    import torch
    F, H, W = c["meta"]["token_grid"]
    x = torch.load(RUN / c["exp"] / M / c["dir"] / "latent.pt").float().numpy().reshape(F, H, W, -1)
    return x.reshape(F, 5, H // 5, 13, W // 13, -1).mean((2, 4)).ravel().astype(np.float32)


def c5():
    cache, rows = {}, []
    for t, exps in TARGET_EXPS.items():
        cl, _, y, g = load(exps, t)
        for c in cl:
            k = (c["exp"], c["dir"])
            if k not in cache:
                cache[k] = pooled_latent(c)
        X = np.stack([cache[(c["exp"], c["dir"])] for c in cl])
        mra, pred = probe.score(X, y, g)
        sh = pd.concat([show(e) for e in exps]).set_index(["exp", "dir"])
        meas = np.array([bool(sh["measurable"].get((c["exp"], c["dir"]), False)) for c in cl])
        s = sh[sh.measurable][f"{t}_sym_mra"].dropna() if f"{t}_sym_mra" in sh else pd.Series(dtype=float)
        rows.append(dict(target=t, n=len(y), n_measurable=int(meas.sum()), latent_mra=mra,
                         latent_mra_on_measurable=metrics.sym_mra(pred, y)[meas].mean() if meas.any() else np.nan,
                         mid_probe_mra=best(t).sym_mra, show_mra_median=s.median(), show_mra_mean=s.mean()))
    print("C5: final-latent probe vs best mid-network probe vs show (latent~mid>video: decoder loses it)")
    save(pd.DataFrame(rows), "c5")


# ---------------------------------------------------------------- C6
def vinfo(Xv, Xa, y, g, n_boot=2000, seed=0):
    """I = 0.5 ln(MSE_vae / MSE_vae+acts) of out-of-fold ln-target predictions; bootstrap CI over scenes."""
    ly = np.log(y)
    ev = (probe.cv_predict(Xv, ly, g) - ly) ** 2
    eb = (probe.cv_predict(np.hstack([Xv, Xa]), ly, g) - ly) ** 2
    ug = np.unique(g)
    sv = np.array([ev[g == k].sum() for k in ug])
    sb = np.array([eb[g == k].sum() for k in ug])
    idx = np.random.default_rng(seed).integers(0, len(ug), (n_boot, len(ug)))
    boots = 0.5 * np.log(sv[idx].sum(1) / sb[idx].sum(1))
    return 0.5 * np.log(ev.mean() / eb.mean()), tuple(np.percentile(boots, [2.5, 97.5]))


def mdl(X, y, g, seed=0, fracs=(.1, .2, .4, .6, .8, 1.0), K=256):
    """Online (prequential) code length in bits of ln(y), binned to K levels (uniform code = log2 K per clip).
    Scenes in a fixed random order; the first block is coded uniformly, each later block with a Gaussian
    whose mean is a ridge fit on earlier blocks and whose variance is that fit's cross-validated MSE
    (mixed 50/50 with the uniform code so outliers cost at most 1 bit more than uniform)."""
    ly = np.log(y)
    ug = np.unique(g)
    rank = {s: i for i, s in enumerate(np.random.default_rng(seed).permutation(ug))}
    r = np.array([rank[k] for k in g])
    cuts = [int(round(f * len(ug))) for f in fracs]
    delta = max(np.ptp(ly), 1e-3) / K
    uni = np.log2(K)
    bits = (r < cuts[0]).sum() * uni
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr, te = r < a, (r >= a) & (r < b)
        if len(np.unique(g[tr])) < 4:
            bits += te.sum() * uni
            continue
        m = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": probe.ALPHAS},
                         cv=GroupKFold(3), scoring="neg_mean_squared_error").fit(X[tr], ly[tr], groups=g[tr])
        s2 = max(-m.best_score_, delta ** 2 / 12)     # cannot beat the bin width
        res = ly[te] - m.predict(X[te])
        pn = np.exp(-res ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2) * delta        # mass of the true bin under the Gaussian
        bits += -np.log2(0.5 * pn + 0.5 / K).sum()                                  # 1/2 Gaussian + 1/2 uniform: <= uniform + 1 bit
    return dict(bits=float(bits), uniform_bits=float(len(y) * uni), compression=float(len(y) * uni / bits))


def c6():
    rows = []
    for t, exps in TARGET_EXPS.items():
        _, feats, y, g = load(exps, t)
        b = best(t)
        Xa, Xv = feats[b.feature][int(b.block), int(b.step)], feats["vae/obj"]
        I, (lo, hi) = vinfo(Xv, Xa, y, g)
        yc = probe.control_targets(y, g)
        ma, mv, mc = mdl(Xa, y, g), mdl(Xv, y, g), mdl(Xa, yc, g)
        rows.append(dict(target=t, block=int(b.block), step=int(b.step), n=len(y), I_nats=I, I_lo=lo, I_hi=hi,
                         uniform_bits=ma["uniform_bits"], bits_acts=ma["bits"], bits_vae=mv["bits"], bits_control=mc["bits"],
                         compress_acts=ma["compression"], compress_vae=mv["compression"], compress_control=mc["compression"]))
    print("C6: conditional V-information beyond VAE latents (nats) and online MDL (bits; lower = more learnable)")
    save(pd.DataFrame(rows), "c6")
    rows = []
    for tag in ("noise", "inv"):
        model = f"{M}_{tag}"
        try:
            _, feats, y, g = load(["p7"], "hspeed", model)
            b = best("hspeed", f"_{tag}")
        except (FileNotFoundError, AssertionError, ValueError, KeyError) as e:
            print(f"p7 {tag}: unavailable ({type(e).__name__}: {e})")
            continue
        X = feats[b.feature][int(b.block), int(b.step)]
        ma, mc = mdl(X, y, g), mdl(X, probe.control_targets(y, g), g)
        rows.append(dict(variant=tag, block=int(b.block), step=int(b.step), n=len(y), n_scenes=len(np.unique(g)),
                         probe_mra=b.sym_mra, control_mra=b.get("control", np.nan), bits_acts=ma["bits"],
                         bits_control=mc["bits"], uniform_bits=ma["uniform_bits"], compress_acts=ma["compression"],
                         compress_control=mc["compression"]))
    if rows:
        print("H5: p7 noise vs inverted-start initial latents (hspeed)")
        save(pd.DataFrame(rows), "c6_p7")


# ---------------------------------------------------------------- C9
def c9(n_boot=5000):
    if not any((RUN / "c9" / M).glob("*/meta.json")):
        print("c9: no c9 runs yet")
        return
    s = pd.concat([show("c9"), show("b1").query("seed == 0").assign(cond="b1_s0")])
    s = s[s.measurable & s.hspeed.notna()]
    piv = {c: d.set_index("scene") for c, d in s.groupby("cond")}
    conds = [c for c in ("b1_s0", "leace", "rand") if c in piv]
    common = sorted(set.intersection(*[set(piv[c].index) for c in conds]))
    if len(common) < 5:
        print(f"c9: only {len(common)} scenes measurable in all conditions {conds}; not enough")
        return
    H = {c: piv[c].loc[common, "hspeed"].to_numpy() for c in conds}
    T = {c: piv[c].loc[common, "hspeed_gt"].to_numpy() for c in conds}
    mra = {c: piv[c].loc[common, "hspeed_sym_mra"].to_numpy() for c in conds}
    idx = np.random.default_rng(0).integers(0, len(common), (n_boot, len(common)))
    rho = lambda c, i: metrics.spearman(H[c][i], T[c][i])
    boot = {c: np.array([rho(c, i) for i in idx]) for c in conds}
    rows = [dict(cond=c, n_scenes=len(common), rho=rho(c, slice(None)), rho_lo=np.percentile(boot[c], 2.5),
                 rho_hi=np.percentile(boot[c], 97.5), median_mra=np.median(mra[c]),
                 mra_lo=np.percentile([np.median(mra[c][i]) for i in idx[:1000]], 2.5),
                 mra_hi=np.percentile([np.median(mra[c][i]) for i in idx[:1000]], 97.5)) for c in conds]
    print(f"C9 readout on the {len(common)} scenes measurable in all conditions")
    save(pd.DataFrame(rows), "c9_readout")
    diffs = []
    for a, b in (("rand", "leace"), ("b1_s0", "leace"), ("b1_s0", "rand")):
        if a in boot and b in boot:
            d = boot[a] - boot[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diffs.append(dict(contrast=f"rho({a}) - rho({b})", diff=rho(a, slice(None)) - rho(b, slice(None)),
                              lo=lo, hi=hi, excludes_0=bool(lo > 0 or hi < 0)))
    save(pd.DataFrame(diffs), "c9_rho_diff")
    if "leace" in boot and "rand" in boot:
        used = diffs[0]["excludes_0"] and diffs[0]["diff"] > 0
        print("information used (rho(leace) < rho(rand), CI excludes 0):", used)
    # probe MRA on the c9 acts at blocks >= the erased block
    cl9 = clips(["c9"], "hspeed")
    cl1 = [c for c in clips(["b1"], "hspeed") if c["meta"]["seed"] == 0]
    erased = int(cl9[0]["meta"]["block"])
    f9, y9, g9 = probe.load(str(RUN), M, ["c9"], "hspeed")
    f1, y1, g1 = probe.load(str(RUN), M, ["b1"], "hspeed")
    s0 = np.array([c["meta"]["seed"] == 0 for c in clips(["b1"], "hspeed")])
    sets = {c: (f9, y9, g9, np.array([x["meta"].get("cond") == c for x in cl9])) for c in ("leace", "rand")}
    sets["b1_s0"] = (f1, y1, g1, s0)
    rows = []
    for c, (f, y, g, mask) in sets.items():
        A = f["transformer_blocks/obj"]
        for b in range(erased, A.shape[0]):
            for st in range(A.shape[1]):
                rows.append(dict(cond=c, block=b, step=st, n=int(mask.sum()), probe_mra=probe.score(A[b, st][mask], y[mask], g[mask])[0]))
    print(f"C9 probe MRA on c9 acts, blocks >= {erased} (probe fit within each condition)")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}c9_probe.csv", index=False)
    print(df.groupby("cond").probe_mra.agg(["mean", "max"]).join(df.groupby("cond").apply(lambda d: d.loc[d.probe_mra.idxmax(), ["block", "step"]], include_groups=False)).to_string())


# ---------------------------------------------------------------- gates
def gates():
    gp = f"{OUT}gap.csv"
    gd = pd.read_csv(gp) if Path(gp).exists() else gap()
    y = __import__("yaml").safe_load(Path("configs/gates.yaml").read_text(encoding="utf-8"))
    rows = []
    add = lambda gate, value, thr, verdict, src: rows.append((gate, value, thr, verdict, src))

    add("G1a tracker vs GT", "-", f">= {y['G1_tracker']['a_min_sym_mra_vs_gt']} sym MRA", "not evaluated",
        "needs the tracker-vs-GT run on K-main renders and QuantiPhy val (results/p0_renders.csv, p0_qval.csv are not LTX clips)")
    meas = {e: show(e).measurable.mean() for e in ("b1", "b5", "b2nl", "b3", "bratio") if len(show(e))}
    thr = y["G1_tracker"]["b_min_measurability"]
    add("G1b measurability", ", ".join(f"{e} {v:.2f}" for e, v in meas.items()), f">= {thr} (verdict: min over exps)",
        "PASS" if min(meas.values()) >= thr else "FAIL", "runs/{b1,b5,b2nl,b3,bratio}/ltx2b/*/score.json")
    b1 = show("b1")
    m1 = b1[b1.measurable & b1.hspeed_sym_mra.notna()]
    med = m1.hspeed_sym_mra.median()
    lo, hi = y["G2_score_range"]["b1_sym_mra_min"], y["G2_score_range"]["b1_sym_mra_max"]
    add("G2 B1 score range", f"median hspeed sym MRA over {len(m1)} measurable B1 clips = {med:.3f} (statistic interpreted)",
        f"[{lo}, {hi}]", "PASS" if lo <= med <= hi else "FAIL", "runs/b1/ltx2b/*/score.json")
    mg = {t: pd.read_csv(f"results/p6_{M}_{t}_obj.csv").query("block >= 0").margin_over_vae_and_control.max() for t in TARGET_EXPS}
    thr = y["G3_probe_signal"]["min_margin_over_vae_and_control"]
    add("G3 probe signal", ", ".join(f"{t} {v:.3f}" for t, v in mg.items()), f">= {thr} in >= 1 target",
        "PASS" if max(mg.values()) >= thr else "FAIL", "results/p6_ltx2b_{hspeed,g,e}_obj.csv (best block)")
    b5 = show("b5")
    m5 = b5[b5.measurable & (b5.get("g", pd.Series(dtype=float)) > 0) & (b5.get("g_gt", pd.Series(dtype=float)) > 0)]
    if len(m5) >= 6 and m5.g_gt.nunique() > 1:
        sl, (a, b) = metrics.gamma_fit(m5.g.to_numpy(), m5.g_gt.to_numpy(), groups=m5.scene.to_numpy())
        add("G4 gamma CI width", f"gamma {sl:.2f}, 95% CI [{a:.2f}, {b:.2f}], width {b - a:.2f} ({len(m5)} B5 clips; fitted here)",
            f"<= {y['G4_gamma_fit']['max_ci95_width']}", "PASS" if b - a <= y["G4_gamma_fit"]["max_ci95_width"] else "FAIL",
            "runs/b5/ltx2b/*/score.json")
    else:
        add("G4 gamma CI width", f"only {len(m5)} measurable B5 clips", f"<= {y['G4_gamma_fit']['max_ci95_width']}", "not evaluated", "runs/b5")
    ale, sil = metrics.abs_log_err(m1.hspeed, m1.hspeed_gt), metrics.silog(m1.hspeed, m1.hspeed_gt)
    lim = y["G5_clock"]["max_abs_log_err_over_silog"]
    add("G5 clock", f"median |ln(pred/true)| {ale:.3f} / SILog {sil:.3f} = {ale / sil:.2f}; {len(m1)} B1 clips (bias {metrics.signed_log_bias(m1.hspeed, m1.hspeed_gt):+.2f})",
        f"<= {lim} (> {lim}: clock dominates)", "PASS" if ale / sil <= lim else "FAIL (clock dominates)", "runs/b1/ltx2b/*/score.json")
    st = pd.concat([show("steer"), b1[b1.seed == 0].assign(alpha=0.0)])
    st = st[st.scene.isin(show("steer").scene.unique()) & st.measurable & (st.hspeed > 0) & (st.hspeed_gt > 0)]
    if len(st) >= 6 and st.alpha.nunique() > 2:
        # gamma_fit regresses ln(pred) on ln(true): pass pred = hspeed/truth and true = exp(alpha) to get the alpha slope
        sl, (a, b) = metrics.gamma_fit((st.hspeed / st.hspeed_gt).to_numpy(), np.exp(st.alpha.to_numpy(float)), groups=st.scene.to_numpy())
        rat = st.groupby("alpha").apply(lambda d: (d.hspeed / d.hspeed_gt).median(), include_groups=False)
        add("G6 steering", f"slope of ln(fitted/true hspeed) per alpha = {sl:+.3f}, 95% CI [{a:+.3f}, {b:+.3f}], {len(st)} measurable clips; "
            f"median ratio by alpha: " + ", ".join(f"{k:+g}:{v:.2f}" for k, v in rat.items()), "CI excludes 0",
            "PASS" if a > 0 or b < 0 else "FAIL", "runs/steer/ltx2b/*/score.json + runs/b1/ltx2b seed 0 (alpha 0)")
    else:
        add("G6 steering", f"only {len(st)} measurable steer clips", "CI excludes 0", "not evaluated", "runs/steer/ltx2b")
    add("G7 budget", "-", "<= 0.25 rel. error at 1 estimate/clip (LTX)", "not evaluated", "needs the timing table (runs/timing)")
    r = gd[gd.target == "hspeed"].iloc[0]
    if "sd_scene" in r and np.isfinite(r.sd_scene):
        z = 0.10 / (r.sd_scene / np.sqrt(r.n_scenes))
        pw = stats.norm.cdf(z - 1.96) + stats.norm.cdf(-z - 1.96)
        add("G8 power", f"per-scene gap SD {r.sd_scene:.3f}, {int(r.n_scenes)} scenes (hspeed, measurable clips) -> power {pw:.2f} at gap 0.10, alpha 0.05 (paired normal approx.)",
            f">= {y['G8_power']['target_power']}", "PASS" if pw >= y["G8_power"]["target_power"] else "FAIL", f"{OUT}gap.csv")
    else:
        add("G8 power", "no gap estimate", "", "not evaluated", f"{OUT}gap.csv")
    add("G9 VACE continuation", "-", "VACE only", "not evaluated", "not applicable to LTX-2B")
    real = show("b1real")
    if len(real):
        d = 100 * (b1.measurable.mean() - real.measurable.mean())
        add("G10 domain gap", f"measurability rendered B1 {b1.measurable.mean():.2f} vs real b1real {real.measurable.mean():.2f} = {d:+.1f} points ({len(real)} real clips; residual-ratio part not evaluated)",
            f"|gap| <= {y['G10_domain_gap']['max_measurability_gap_points']} points", "PASS" if abs(d) <= y["G10_domain_gap"]["max_measurability_gap_points"] else "FAIL",
            "runs/{b1,b1real}/ltx2b/*/score.json")
    else:
        add("G10 domain gap", "no b1real scores", "", "not evaluated", "runs/b1real")
    md = ["# Gates, LTX-2B (generated by pilot/analysis.py)", "", "| Gate | Value | Threshold | Verdict | Source |", "|---|---|---|---|---|"]
    md += [f"| {' | '.join(str(c).replace('|', '/') for c in row)} |" for row in rows]
    Path("results").mkdir(exist_ok=True)
    Path("results/gates_ltx2b.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------- self-check
def selftest():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(80), 3)
    Xa, Xn, Xv = rng.normal(size=(240, 20)), rng.normal(size=(240, 20)), rng.normal(size=(240, 5))
    y = np.exp(0.5 * Xv[:, 0] + 1.0 * Xa[:, 0] + 0.1 * rng.normal(size=240))
    I, (lo, hi) = vinfo(Xv, Xa, y, g, n_boot=300)
    assert I > 0 and lo > 0, (I, lo)                   # acts carry info the VAE features lack
    In, _ = vinfo(Xv, Xn, y, g, n_boot=300)
    assert In < I / 2, (In, I)                         # pure-noise "acts" add (almost) nothing
    ylearn, ynoise = np.exp(Xa[:, 0] + 0.1 * rng.normal(size=240)), np.exp(rng.normal(size=240))
    a, b = mdl(Xa, ylearn, g), mdl(Xa, ynoise, g)
    assert a["bits"] < b["bits"] and a["compression"] > b["compression"], (a, b)
    print(f"selftest ok: I={I:.2f} nats (noise {In:.2f}); MDL learnable {a['bits']:.0f} bits vs noise {b['bits']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "gap", "c5", "c6", "c9", "gates", "all", "selftest"])
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    fns = dict(fetch=fetch, gap=gap, c5=c5, c6=c6, c9=c9, gates=gates, selftest=selftest)
    for name in (["fetch"] * (not a.no_fetch) + ["gap", "c5", "c6", "c9", "gates"]) if a.cmd == "all" else [a.cmd]:
        print(f"\n=== {name} ===", flush=True)
        fns[name]()
