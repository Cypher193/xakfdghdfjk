"""CPU check of the per-model pod jobs (run_pilot.MODEL_JOBS), the C9c jobs, and what sync.py / setup_runpod.sh /
vast_launch derive from them.
Needs data/scenes (kmain, kgrav) and data/real/manifest.json: python scripts/sync.py pull, or a laptop checkout.
    python -m tests.test_run_pilot"""
import argparse
import importlib
import os
import sys
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

for c in ("leace", "rand"):                         # C9c: one chain job per cond, then 4 shards over kmain x {leace, rand}
    (st,) = LTX_JOBS[f"c9cchain-{c}"]
    assert st["cmd"][st["cmd"].index("--chain") + 1] == c and st["est_s"] < 7.5 * 3600, st
c9c = [i for k in range(1, 5) for i in LTX_JOBS[f"c9cs{k}of4"][0]["ids"]]
assert sorted(c9c) == sorted(f"{s}_{c}" for s in kmain for c in ("leace", "rand"))
assert LTX_JOBS["c9cs1of4"][0]["cmd"][-2:] == ["--exp", "c9c"]

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
