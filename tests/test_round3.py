"""CPU check of the round 3 scenes, readouts and jobs. Needs data/scenes from `python -m pilot.scenes`.
    python -m tests.test_round3"""
import argparse
import importlib
import json
import math
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import yaml

from pilot import scenes, shard, track
from pilot.score import target_value

sys.path.insert(0, "scripts")
from run_pilot import LTX_JOBS, LTX_RUNS, ROUND3, job_repos
from vast_launch import worker_label

cfg = yaml.safe_load(open("configs/pilot.yaml"))
built = scenes.build(cfg)
ids = {k: [s["id"] for s in v] for k, v in built.items()}
assert {k: len(v) for k, v in built.items()} == dict(kmain=100, kgrav=90, kratio=10, kratio2=40, kl3=30, kcont=70)

# collisions: momentum holds, restitution holds, contact after the longest input window, balls touch at contact
for s in built["kl3"][:15]:
    a, b = s["objects"]
    ma, mb = a["size_m"] ** 3, b["size_m"] ** 3
    assert math.isclose(ma * a["vx"], ma * a["v_after"] + mb * b["v_after"], rel_tol=1e-9), s["id"]
    assert math.isclose(b["v_after"] - a["v_after"], s["gt"]["e"] * a["vx"], rel_tol=1e-9), s["id"]
    assert s["gt"]["t_c"] >= 1.3 and s["gt"]["v"] == abs(b["v_after"]) and a["color"] != b["color"]
    (xa,), (xb,) = (scenes.object_track(s, o, [s["gt"]["t_c"]])[0] for o in (a, b))
    assert math.isclose(abs(xb - xa), (a["size_m"] + b["size_m"]) / 2, rel_tol=1e-9), s["id"]

# ramps: track starts at rest at the top, reaches the floor at t_foot, a_slope is the horizontal acceleration, box stops
for s in built["kl3"][15:]:
    o, t_foot = s["objects"][0], s["gt"]["t_foot"]
    assert t_foot >= 1.6, s["id"]
    x, z = scenes.object_track(s, o, np.array([0.0, 0.5, t_foot, 5.9, 6.0]))
    assert x[0] == o["x0"] and z[0] == o["z0"] and math.isclose(z[2], s["floor_z"] + o["size_m"] / 2, abs_tol=1e-9)
    assert math.isclose(abs(x[1] - x[0]), 0.5 * s["gt"]["a_slope"] * 0.25, rel_tol=1e-9), s["id"]
    assert x[3] == x[4], s["id"]

# tracker readouts recover ground truth from the exact tracks at both generator frame rates, past the input window
for s in built["kl3"]:
    for fps, t_start in ((16, 5 / 16), (24, 25 / 24)):
        t = np.arange(int(5 * fps) + 1) / fps
        xs, zs = zip(*(scenes.object_track(s, o, t) for o in s["objects"]))
        r = track.readout(s["family"], t, list(xs), list(zs), t_start)
        k = "v" if s["family"] == "collide" else "a_slope"
        assert abs(r[k] / s["gt"][k] - 1) < (0.01 if k == "v" else 0.12), (s["id"], fps, r, s["gt"])
assert target_value(built["kl3"][0]["gt"], "hspeed") == built["kl3"][0]["gt"]["v"]

# text: every prompt field exists, b2num and b2nl carry the same scene, no prompt nears the shortest encoder limit
main = {s["id"]: s for s in built["kmain"]}
for s in built["kmain"] + built["kl3"]:
    assert "b2nl" in s and s["caption"] and len(s["b2nl"].split()) < 90, s["id"]
assert all(not any(c.isdigit() for c in s["b2nl"]) for s in built["kl3"])
assert all(any(c.isdigit() for c in s["b2num"]) and len(s["b2num"].split()) < 70 for s in main.values())
assert "0.15 m wide" in main["kmain_cv_00"]["b2num"] and "0.32 m/s" in main["kmain_cv_00"]["b2num"]

# every job's ids are what pilot.generate --limit / --shard picks, and each model's pods cover each experiment once
sc = dict(cfg, paths=dict(cfg["paths"], data="data"))
clips = {"cosmos2b": 0, "vace13": 0, "ltx2b": 0}
for model, (n, _, blocks) in ROUND3.items():
    for exp, limit, seeds in blocks:
        full = [s["id"] for s in shard.scene_list(sc, exp)][: limit or None]
        got = []
        for k in range(1, n + 1):
            step = next(s for s in LTX_JOBS[f"{model}g{k}of{n}"] if s["name"] == f"{exp}:{model}")
            assert step["ids"] == [s["id"] for s in shard.shard(shard.scene_list(sc, exp)[: limit or None], f"{k}/{n}")], (exp, k)
            assert step["seeds"] == seeds
            got += step["ids"]
        assert sorted(got) == sorted(full), (model, exp)
    clips[model] = sum(s.get("clips", 0) for k in range(1, n + 1) for s in LTX_JOBS[f"{model}g{k}of{n}"])
# planned clips + the b5 clips that exist already (skipped at run time): cosmos b5 30 x 3, ltx b5 30 x 5 + bratio2 10 x 3
assert clips == dict(cosmos2b=1340 + 90, vace13=500, ltx2b=750 + 150 + 30), clips

assert job_repos("ltx2bg1of2") == [("Lightricks/LTX-Video-0.9.5", None), ("facebook/sam2.1-hiera-large", None)]
assert job_repos("render3s2of3") == [("facebook/sam2.1-hiera-large", None)]
label = lambda j: worker_label(argparse.Namespace(job=j, n=1), 1)
assert label("cosmos2bg3of20") == "kvs-cosmos2bg-3of20" and label("ltx2bg1of2") == "kvs-ltx2bg-1of2"
assert label("cosmos2bs2of5") == "kvs-cosmos2b-2of5" and label("vace13r4of7") == "kvs-vace13r-4of7"
assert label("render3s1of3") == "kvs-render3-1of3"


def sync_for(job):
    os.environ |= {"LTX_JOB": job, "HF_DATA_REPO": "dummy/dummy"}
    return importlib.reload(importlib.import_module("sync"))


match = lambda f, pats: any(fnmatch(f, p) for p in pats)
s = sync_for("cosmos2bg7of20")
assert set(LTX_RUNS["cosmos2bg7of20"]) == {f"runs/{e}/cosmos2b" for e in ("b2nl", "b2num", "b3", "b5", "bratio2", "bl3", "bl3nl")}
for f in ("data/renders/kl3/kl3_co_00_input.mp4", "data/renders/kratio2/kratio2_39_input.mp4", "data/renders/kgrav/kgrav_29_g2.0_input.mp4",
          "data/renders/kmain/kmain_cv_49_input.mp4", "runs/b5/cosmos2b/kgrav_00_g0.5_s0/meta.json", "data/scenes/kl3/kl3_co_00.json"):
    assert match(f, s.PATTERNS), f
assert match("runs/bl3nl/cosmos2b/kl3_ra_00_s0/video.mp4", s.PUSH_PATTERNS) and match("results/show_ltx_cosmos2bg7of20.csv", s.PUSH_PATTERNS)
assert not match("runs/b1/cosmos2b/x/video.mp4", s.PUSH_PATTERNS) and not match("runs/b5/ltx2b/x/video.mp4", s.PUSH_PATTERNS)
s = sync_for("render3s1of3")
assert match("assets_cache/x.json", s.PATTERNS) and match("data/renders/kgrav/kgrav_00_g0.5_input.mp4", s.PATTERNS)
assert match("data/renders/kl3/kl3_co_00_input.mp4", s.PUSH_PATTERNS) and match("data/renders/kl3/kl3_co_00_input.assets.json", s.PUSH_PATTERNS)
assert match("results/p0_kl3_ltx_render3s1of3.csv", s.PUSH_PATTERNS) and not match("data/renders/kmain/x_input.mp4", s.PATTERNS)
assert match("data/renders/kl3full/kl3_co_00.mp4", s.PUSH_PATTERNS)
assert label("cosmos2brescore3") == "kvs-cosmos2brescore3-1of1"
for m, n_steps in (("ltx2b", 1), ("cosmos2b", 1), ("vace13", 2)):
    j, s = f"{m}rescore3", sync_for(f"{m}rescore3")
    assert len(LTX_JOBS[j]) == n_steps and job_repos(j) == [("facebook/sam2.1-hiera-large", None)]
    written = [a for t in LTX_JOBS[j] for a in t["cmd"] if a.startswith("results/")]
    assert len(written) == n_steps + 1 and all(match(f, s.PUSH_PATTERNS) for f in written), written
    assert match(f"runs/bratio2/{m}/kratio2_00_s0/video.mp4", s.PATTERNS) and match(f"runs/bratio2/{m}/kratio2_00_s0/score_v2.json", s.PUSH_PATTERNS)
    assert not match(f"runs/bratio2/{m}/kratio2_00_s0/video.mp4", s.PUSH_PATTERNS), "scores only: finalize must not re-verify videos"
    assert not match("runs/c9/ltx2b/x/video.mp4", s.PATTERNS) and match("data/scenes/kl3/kl3_co_00.json", s.PATTERNS)
assert match("data/renders/kl3full/kl3_ra_00.mp4", s.PATTERNS)      # s is the VACE pod here
assert not match("data/renders/kl3full/kl3_ra_00.mp4", sync_for("ltx2brescore3").PATTERNS)
del os.environ["LTX_JOB"], os.environ["HF_DATA_REPO"]
print("round 3 checks passed:", clips)
