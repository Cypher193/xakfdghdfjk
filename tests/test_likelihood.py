"""CPU check of C2 / C2-g / C2-turn on tiny Cosmos, LTX and VACE: finite losses per k, resume, summary in range, velocity copies.
Needs data/scenes from `python -m pilot.scenes`.  python -m tests.test_likelihood"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pilot import scenes
from pilot.likelihood import LOSSES, bases, implied_multiplier, is_true, mult, run, summarize
from tests import tiny

assert abs(implied_multiplier(np.array([0.25, 0.5, 1, 2, 4]), (np.log([0.25, 0.5, 1, 2, 4]) - np.log(1.5)) ** 2) - 1.5) < 1e-6
s = summarize(pd.DataFrame(dict(base="b", k=[0.5, 1.0, 2.0], loss=[0.3, 0.1, 0.05])))
assert s.pref_acc.iloc[0] == 0.5 and abs(s.k_implied.iloc[0] - 2 ** (5 / 6)) < 1e-9   # parabola through 3 points

# velocity copies: window unchanged, speed scaled after it, k = 1 is the kmain track
m = json.loads(Path("data/scenes/kmain/kmain_cv_03.json").read_text(encoding="utf-8"))
for k in (0.5, 2.0):
    c = json.loads(Path(f"data/scenes/kvell/kvell_cv_03_x{k}.json").read_text(encoding="utf-8"))
    ts = c["vswitch"][0]
    (x,), _ = scenes.object_track(c, c["objects"][0], [ts + 1.0])
    (x0,), _ = scenes.object_track(m, m["objects"][0], [ts])
    assert math.isclose(x - x0, k * m["objects"][0]["vx"], rel_tol=1e-9) and ts == 25 / 24
same = json.loads(Path("data/scenes/kvelc/kvelc_ca_03_x1.0.json").read_text(encoding="utf-8"))
assert same["tracks"] == json.loads(Path("data/scenes/kmain/kmain_ca_03.json").read_text(encoding="utf-8"))["tracks"]
b = json.loads(Path("data/scenes/kvelc/kvelc_ca_03_x2.0.json").read_text(encoding="utf-8"))
z = np.array(b["tracks"][0]["z_m"])
assert np.allclose(z[:8], same["tracks"][0]["z_m"][:8]) and not np.allclose(z[8:12], same["tracks"][0]["z_m"][8:12])
assert len(bases("data/scenes/kvelc", "2/4")) == 25 and all(len(v) == 5 for v in bases("data/scenes/kvelc").values())

# turned copies: centred at the turn, same speed after it, 180 = reversed; bounces have no 30 degree copy
for sid, d in (("kturnc_cv_03", 30), ("kturnl_ca_10", 180)):
    c = json.loads(Path(f"data/scenes/{sid[:6]}/{sid}_x{d}.json").read_text(encoding="utf-8"))
    t0 = json.loads(Path(f"data/scenes/{sid[:6]}/{sid}_x0.json").read_text(encoding="utf-8"))
    ts, e = c["vturn"][0], 1e-5
    x, z = scenes.object_track(c, c["objects"][0], [ts - e, ts, ts + e])
    assert abs(x[1]) < 1e-9 and math.isclose(np.hypot(x[2] - x[1], z[2] - z[1]), np.hypot(x[1] - x[0], z[1] - z[0]), rel_tol=1e-3)
    assert c["tracks"][0]["x_m"][:5] == t0["tracks"][0]["x_m"][:5] and is_true(t0) and not is_true(c)
    if d == 180:
        assert c["gt"]["in_frame"] == t0["gt"]["in_frame"]
assert {len(v) for v in bases("data/scenes/kturnc").values()} == {2, 3}
l = json.loads(Path("data/scenes/kcontl/kcontl_03_x2.0.json").read_text(encoding="utf-8"))
assert l["gravity"][1][0] == 25 / 24 == l["gt"]["t_switch"] and l["gt"]["cont_mult"] == 2.0
s = summarize(pd.DataFrame(dict(base="b", k=[0, 30, 180], true=[True, False, False], loss=[0.2, 0.3, 0.1])))
assert s.pref_acc.iloc[0] == 0.5 and np.isnan(s.k_implied.iloc[0])

tmp = Path("tests/_likelihood_tmp")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
for model, scene_set in (("cosmos2b", "kcont"), ("ltx2b", "kvell"), ("ltx2b", "kturnl"), ("vace13", "kvelv")):
    pipe, cfg = tiny.BUILDERS[model]()
    lo = LOSSES[model](pipe, cfg, "cpu", torch.float32)
    out = tmp / f"{model}_{scene_set}.csv"
    frames = lambda spec: tiny.frames(cfg["num_frames"], seed=int(mult(spec) * 4))
    df, summ = run(lo, cfg, f"data/scenes/{scene_set}", None, out, n_sigmas=2, n_noise=1, shard="1/20", frames_fn=frames)
    n = len(df)
    df, summ = run(lo, cfg, f"data/scenes/{scene_set}", None, out, n_sigmas=2, n_noise=1, shard="1/20", frames_fn=frames)
    print(model, df.round(5).to_string(index=False), summ.round(3).to_string(index=False), sep="\n")
    assert len(df) == n == sum(map(len, bases(f"data/scenes/{scene_set}", "1/20").values())), "resume skips done bases"
    assert np.isfinite(df.loss).all() and summ.pref_acc.between(0, 1).all() and df.true.sum() == len(summ)
    assert summ.k_implied.isna().all() if scene_set == "kturnl" else summ.k_implied.between(0.25, 4).all()
    assert out.with_name(f"{model}_{scene_set}_summary.csv").exists()
shutil.rmtree(tmp)
print("likelihood checks passed")
