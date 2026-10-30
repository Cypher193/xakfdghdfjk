"""CPU check of the per-model pod jobs (run_pilot.MODEL_JOBS), the C9c jobs, and what sync.py / setup_runpod.sh /
vast_launch derive from them.
Needs data/scenes (kmain, kgrav) and data/real/manifest.json: python scripts/sync.py pull, or a laptop checkout.
    python -m tests.test_run_pilot"""
import argparse
import importlib
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

sys.path.insert(0, "scripts")
from run_pilot import LTX_JOBS, job_repos

kmain = sorted(p.stem for p in Path("data/scenes/kmain").glob("*.json"))
kgrav = sorted(p.stem for p in Path("data/scenes/kgrav").glob("*.json"))
assert len(kmain) == 100 and len(kgrav) == 30, (len(kmain), len(kgrav))
for model in ("cosmos2b", "vace13"):
    b1, b5 = [], []
    for k in range(1, 6):
        steps = LTX_JOBS[f"{model}s{k}of5"]
        assert [s["name"] for s in steps] == [f"b1:{model}", f"b5:{model}"] + ([f"b1real:{model}"] if k == 1 else []) \
            + [f"score:{model}s{k}of5"]
        assert len(steps[0]["ids"]) == 20 and len(steps[1]["ids"]) == 6
        assert {i.split("_")[1] for i in steps[0]["ids"]} == {"cv", "ca"}      # both families in every shard
        assert sum(s.get("clips", 0) for s in steps) == len(steps[0]["ids"]) + len(steps[1]["ids"]) + (7 if k == 1 else 0)
        assert all(s["seeds"] == [0] for s in steps[:2]) and steps[-1]["cmd"][-1] == f"results/show_ltx_{model}s{k}of5.csv"
        b1 += steps[0]["ids"]
        b5 += steps[1]["ids"]
    assert sorted(b1) == kmain and sorted(b5) == kgrav, model       # every scene in exactly one shard
    assert [s["cmd"][s["cmd"].index("--model") + 1] for s in LTX_JOBS[f"{model}probe"]] == [model] * 6

for model, b5_seeds in (("cosmos2b", 2), ("vace13", 2)):           # round 2: 7 pods, every scene once
    jobs = [LTX_JOBS[f"{model}r{k}of7"] for k in range(1, 8)]
    assert all([s["name"] for s in j] == [f"b1:{model}", f"b2nl:{model}", f"b5:{model}", f"score:{model}r{k}of7"]
               for k, j in enumerate(jobs, 1))
    assert sorted(i for j in jobs for i in j[0]["ids"]) == kmain and sorted(i for j in jobs for i in j[2]["ids"]) == kgrav
    assert all(j[0]["seeds"] == [1, 2] and j[1]["seeds"] == [0] and len(j[2]["seeds"]) == b5_seeds for j in jobs)
    assert sum(s.get("clips", 0) for j in jobs for s in j) == 100 * 3 + 30 * b5_seeds
for c in ("leace", "rand"):                         # C9c: one chain job per cond, then 4 shards over kmain x {leace, rand}
    (st,) = LTX_JOBS[f"c9cchain-{c}"]
    assert st["cmd"][st["cmd"].index("--chain") + 1] == c and st["est_s"] < 7.5 * 3600, st
c9c = [i for k in range(1, 5) for i in LTX_JOBS[f"c9cs{k}of4"][0]["ids"]]
assert sorted(c9c) == sorted(f"{s}_{c}" for s in kmain for c in ("leace", "rand"))
assert LTX_JOBS["c9cs1of4"][0]["cmd"][-4:] == ["--exp", "c9c", "--use-groups", "3"]

sam2 = ("facebook/sam2.1-hiera-large", None)
assert job_repos("cosmos2bs1of5") == [("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"), sam2]
assert job_repos("vace13s3of5") == [("Wan-AI/Wan2.1-VACE-1.3B-diffusers", None), sam2]
assert job_repos("s1of7") == job_repos("gen4") == job_repos("c9bs2of4") == job_repos("c9cs3of4") == \
    job_repos("c9cchain-rand") == [("Lightricks/LTX-Video-0.9.5", None), sam2]
assert job_repos("score2") == job_repos("cosmos2bprobe") == job_repos("probe2") == [sam2]

from vast_launch import worker_label
assert worker_label(argparse.Namespace(job="c9cchain-rand", n=1), 1) == "kvs-c9cchain_rand-1of1"
assert worker_label(argparse.Namespace(job="c9cs3of4", n=1), 1) == "kvs-c9c-3of4"


def sync_for(job):
    os.environ |= {"LTX_JOB": job, "HF_DATA_REPO": "dummy/dummy"}
    return importlib.reload(importlib.import_module("sync"))


sync = sync_for("cosmos2bs2of5")
assert {"runs/b1/cosmos2b/**", "runs/b5/cosmos2b/**"} <= set(sync.PUSH_PATTERNS), sync.PUSH_PATTERNS
assert "runs/b1real/cosmos2b/**" not in sync.PUSH_PATTERNS and "results/*_ltx_cosmos2bs2of5*" in sync.PUSH_PATTERNS
assert {"data/renders/kmain/*_input.mp4", "data/renders/kgrav/*_input.mp4", "runs/b1/cosmos2b/*/meta.json",
        "runs/b5/cosmos2b/*/score.json", "runs/b1/cosmos2b/*/video.mp4"} <= set(sync.PATTERNS), sync.PATTERNS
assert not any("b1real" in p or "ltx2b" in p for p in sync.PATTERNS)
assert job_repos("cosmos2bsteer") == job_repos("cosmos2bs1of5")

os.environ["LTX_JOB"] = "vace13r4of7"
sync = importlib.reload(sync)
assert {"runs/b1/vace13/**", "runs/b2nl/vace13/**", "runs/b5/vace13/**", "results/*_ltx_vace13r4of7*"} <= \
    set(sync.PUSH_PATTERNS), sync.PUSH_PATTERNS
assert {"runs/b2nl/vace13/*/meta.json", "data/renders/kgrav/*_input.mp4"} <= set(sync.PATTERNS), sync.PATTERNS

os.environ["LTX_JOB"] = "cosmos2bsteer"
sync = importlib.reload(sync)
assert sync.PUSH_PATTERNS == ["data/logs/*_ltx_cosmos2bsteer*", "results/*_ltx_cosmos2bsteer*", "runs/steer/cosmos2b/**"]
assert {"runs/b1/cosmos2b/*/acts.npz", "results/p6_cosmos2b_hspeed_obj.csv"} <= set(sync.PATTERNS)

# stats2: pulls acts/meta/score + p6 + full renders, pushes only its own results and logs
assert [s["name"] for s in LTX_JOBS["stats2"]] == ["nested:ltx2b", "nested:cosmos2b", "nested:vace13"]
assert job_repos("stats2") == [sam2]
os.environ["LTX_JOB"] = "stats2"
sync = importlib.reload(sync)
pulled = lambda f: any(fnmatch(f, p) for p in sync.PATTERNS)
pushed = lambda f: any(fnmatch(f, p) for p in sync.PUSH_PATTERNS)
assert all(pulled(f"runs/{e}/{m}/kmain_cv_00_s0/{x}") for e in ("b1", "b5") for m in ("ltx2b", "cosmos2b", "vace13")
           for x in ("acts.npz", "meta.json", "score.json"))
assert pulled("results/p6_vace13_e_obj.csv") and pulled("data/scenes/kmain/kmain_ca_07.json")
assert not any(pulled(f) for f in ("data/renders/kmain/kmain_ca_07_input.mp4", "runs/b1/ltx2b/kmain_cv_00_s0/video.mp4",
                                   "runs/b1/ltx2b/kmain_cv_00_s0/latent.pt", "runs/b1real/vace13/x/acts.npz"))
outs = [s["cmd"][s["cmd"].index("--out") + 1] for s in LTX_JOBS["stats2"] if "--out" in s["cmd"]] + \
    [f"results/c3_{m}_nested{x}.csv" for m in ("ltx2b", "cosmos2b", "vace13") for x in ("", "_clips")]
assert all(pushed(f) for f in outs + ["data/logs/pilot_ltx_stats2.log"]), outs
assert not any(pushed(f) for f in ("results/p6_ltx2b_g_obj.csv", "results/c3_ltx2b_gap.csv", "runs/b1/ltx2b/x/acts.npz",
                                   "data/renders/kmain/kmain_ca_07.mp4", "data/scenes/kmain/kmain_ca_07.json"))
from vast_launch import worker_label
assert worker_label(argparse.Namespace(job="stats2", n=1), 1) == "kvs-ltxstats2-1of1"

# rescore (configs/deviations.md D3): one SAM2-only pod; every result file it writes is in its own push globs
steps = LTX_JOBS["rescore"]
assert [s["name"] for s in steps] == ["p0:cotracker3", "score:rescore", "p0:sam2"] and job_repos("rescore") == [sam2]
assert all(s["cmd"][s["cmd"].index("--tracker") + 1] == "cotracker3" for s in steps[:2])
os.environ["LTX_JOB"] = "rescore"
sync = importlib.reload(sync)
match = lambda f, pats: any(fnmatch(f, p) for p in pats)
written = [a for s in steps for a in s["cmd"] if a.startswith("results/")]
assert len(written) == 4 and all(match(f, sync.PUSH_PATTERNS) for f in written), written
for e, m in (("b1", "ltx2b"), ("b5", "vace13"), ("b1real", "cosmos2b"), ("bratio2", "ltx2b")):
    d = f"runs/{e}/{m}/x_s0"
    assert match(f"{d}/score_v2.json", sync.PUSH_PATTERNS) and not match(f"{d}/score.json", sync.PUSH_PATTERNS)
    assert match(f"{d}/video.mp4", sync.PATTERNS) and match(f"{d}/meta.json", sync.PATTERNS)
    assert not match(f"{d}/acts.npz", sync.PATTERNS)
assert match("data/renders/kmain/kmain_cv_07.mp4", sync.PATTERNS)
assert not match("data/renders/kmain/kmain_cv_07_input.mp4", sync.PATTERNS) and not match("runs/b2nl/ltx2b/x_s0/video.mp4", sync.PATTERNS)
sync = sync_for("c9cchain-leace")                   # pulls B1 seed-0 acts + its own chain (resume), pushes only the chain
assert {"runs/b1/ltx2b/*_s0/acts.npz", "runs/c9c_chain/ltx2b/leace/**"} <= set(sync.PATTERNS), sync.PATTERNS
assert sync.PUSH_PATTERNS == ["data/logs/*_ltx_c9cchain-leace*", "results/*_ltx_c9cchain-leace*",
                              "runs/c9c_chain/ltx2b/leace/**"], sync.PUSH_PATTERNS
sync = sync_for("c9cs2of4")                         # pulls both finished chains, no B1 acts; pushes only runs/c9c
assert {"runs/c9c_chain/ltx2b/*/chain.json", "runs/c9c_chain/ltx2b/*/g*.npz", "runs/c9c/ltx2b/*/meta.json",
        f"runs/c9c/ltx2b/{LTX_JOBS['c9cs2of4'][0]['ids'][0]}_s0/video.mp4"} <= set(sync.PATTERNS), sync.PATTERNS
assert not any("runs/b1/" in p for p in sync.PATTERNS)
assert sync.PUSH_PATTERNS[-1] == "runs/c9c/ltx2b/**" and not any("chain" in p for p in sync.PUSH_PATTERNS)
print("run_pilot job checks passed")
