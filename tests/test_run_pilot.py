"""CPU check of the per-model pod jobs (run_pilot.MODEL_JOBS) and what sync.py / setup_runpod.sh derive from them.
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

sam2 = ("facebook/sam2.1-hiera-large", None)
assert job_repos("cosmos2bs1of5") == [("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"), sam2]
assert job_repos("vace13s3of5") == [("Wan-AI/Wan2.1-VACE-1.3B-diffusers", None), sam2]
assert job_repos("s1of7") == job_repos("gen4") == job_repos("c9bs2of4") == [("Lightricks/LTX-Video-0.9.5", None), sam2]
assert job_repos("score2") == job_repos("cosmos2bprobe") == job_repos("probe2") == [sam2]

os.environ |= {"LTX_JOB": "cosmos2bs2of5", "HF_DATA_REPO": "dummy/dummy"}
sync = importlib.import_module("sync")
assert {"runs/b1/cosmos2b/**", "runs/b5/cosmos2b/**"} <= set(sync.PUSH_PATTERNS), sync.PUSH_PATTERNS
assert "runs/b1real/cosmos2b/**" not in sync.PUSH_PATTERNS and "results/*_ltx_cosmos2bs2of5*" in sync.PUSH_PATTERNS
assert {"data/renders/kmain/*_input.mp4", "data/renders/kgrav/*_input.mp4", "runs/b1/cosmos2b/*/meta.json",
        "runs/b5/cosmos2b/*/score.json", "runs/b1/cosmos2b/*/video.mp4"} <= set(sync.PATTERNS), sync.PATTERNS
assert not any("b1real" in p or "ltx2b" in p for p in sync.PATTERNS)

# stats2: pulls acts/meta/score + p6 + full renders, pushes only its own results and logs
assert [s["name"] for s in LTX_JOBS["stats2"]] == ["g1a", "nested:ltx2b", "nested:cosmos2b", "nested:vace13"]
assert job_repos("stats2") == [sam2]
os.environ["LTX_JOB"] = "stats2"
sync = importlib.reload(sync)
pulled = lambda f: any(fnmatch(f, p) for p in sync.PATTERNS)
pushed = lambda f: any(fnmatch(f, p) for p in sync.PUSH_PATTERNS)
assert all(pulled(f"runs/{e}/{m}/kmain_cv_00_s0/{x}") for e in ("b1", "b5") for m in ("ltx2b", "cosmos2b", "vace13")
           for x in ("acts.npz", "meta.json", "score.json"))
assert pulled("data/renders/kmain/kmain_ca_07.mp4") and pulled("results/p6_vace13_e_obj.csv") and pulled("data/scenes/kmain/kmain_ca_07.json")
assert not any(pulled(f) for f in ("data/renders/kmain/kmain_ca_07_input.mp4", "runs/b1/ltx2b/kmain_cv_00_s0/video.mp4",
                                   "runs/b1/ltx2b/kmain_cv_00_s0/latent.pt", "runs/b1real/vace13/x/acts.npz"))
outs = [s["cmd"][s["cmd"].index("--out") + 1] for s in LTX_JOBS["stats2"] if "--out" in s["cmd"]] + \
    [f"results/c3_{m}_nested{x}.csv" for m in ("ltx2b", "cosmos2b", "vace13") for x in ("", "_clips")]
assert all(pushed(f) for f in outs + ["data/logs/pilot_ltx_stats2.log"]), outs
assert not any(pushed(f) for f in ("results/p6_ltx2b_g_obj.csv", "results/c3_ltx2b_gap.csv", "runs/b1/ltx2b/x/acts.npz",
                                   "data/renders/kmain/kmain_ca_07.mp4", "data/scenes/kmain/kmain_ca_07.json"))
from vast_launch import worker_label
assert worker_label(argparse.Namespace(job="stats2", n=1), 1) == "kvs-ltxstats2-1of1"
print("run_pilot job checks passed")
