"""C2 / C2-g breakdown from the pod CSVs on HF (results/c2*_ltx_*.csv): pref_acc with a bootstrap CI over bases,
by velocity or gravity multiplier k, by family, in-frame copies only, symmetric pairs, and argmin k.

    python scripts/c2_report.py
"""
import glob, re
from pathlib import Path

import numpy as np, pandas as pd
rng = np.random.default_rng(0)
groups = sorted({re.sub(r"_ltx_.*", "", Path(f).name) for f in glob.glob("results/c2*_ltx_*.csv")})
for pat in (f"results/{g}_ltx_*.csv" for g in groups):
    df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(pat)) if not f.endswith("_summary.csv"))
    df = df.drop_duplicates(["base", "id"])
    t = df.true.astype(bool) if "true" in df else df.k == 1.0
    true = df[t].set_index("base").loss
    alt = df[~t].copy()
    alt["win"] = (alt.loss > alt.base.map(true)).astype(float)
    alt["fam"] = alt.base.str.split("_").str[1]
    per = alt.groupby("base").win.mean()
    bs = [rng.choice(per.values, len(per)).mean() for _ in range(5000)]
    print(f"\n== {pat}  bases={df.base.nunique()} rows={len(df)}")
    print(f"pref_acc {per.mean():.3f} [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    print("by k:", alt.groupby("k").win.mean().round(3).to_dict())
    print("by fam:", alt.groupby("fam").win.mean().round(3).to_dict(), alt.groupby("fam").base.nunique().to_dict())
    if alt.fam.nunique() <= 2:                   # cv / ca; kcont bases are numbered, not families
        print("by fam and k:", alt.groupby(["fam", "k"]).win.mean().round(3).to_dict())
    if alt.in_frame.notna().any():
        inf = alt[alt.in_frame == 1]
        print(f"in-frame only: {inf.win.mean():.3f} (n={len(inf)} of {len(alt)})")
    L = df.pivot(index="base", columns="k", values="loss")
    ks = sorted(L.columns)
    for a, b in [(a, b) for a, b in ((0.8, 1.25), (0.75, 1.5), (0.5, 2.0), (0.25, 4.0)) if a in ks and b in ks]:
        print(f"loss({a}) < loss({b}): {(L[a] < L[b]).mean():.2f}")
    print("argmin k:", L.idxmin(axis=1).value_counts().sort_index().to_dict())
    summ = sorted(glob.glob(pat.replace(".csv", "_summary.csv")))     # written when a pod's step ends
    if (df.k > 0).all() and summ:                                     # the turn sets have no implied multiplier
        s = pd.concat(map(pd.read_csv, summ))
        print(f"k_implied median {s.k_implied.median():.3f}, sym_mra mean {s.sym_mra.mean():.3f}")
