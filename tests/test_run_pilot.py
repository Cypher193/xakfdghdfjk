"""CPU check of the per-model pod jobs (run_pilot.MODEL_JOBS) and what sync.py / setup_runpod.sh derive from them.
Needs data/scenes (kmain, kgrav) and data/real/manifest.json: python scripts/sync.py pull, or a laptop checkout.
    python -m tests.test_run_pilot"""
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

for model, b5_seeds in (("cosmos2b", 2), ("vace13", 2)):           # round 2: 7 pods, every scene once
    jobs = [LTX_JOBS[f"{model}r{k}of7"] for k in range(1, 8)]
    assert all([s["name"] for s in j] == [f"b1:{model}", f"b2nl:{model}", f"b5:{model}", f"score:{model}r{k}of7"]
               for k, j in enumerate(jobs, 1))
    assert sorted(i for j in jobs for i in j[0]["ids"]) == kmain and sorted(i for j in jobs for i in j[2]["ids"]) == kgrav
    assert all(j[0]["seeds"] == [1, 2] and j[1]["seeds"] == [0] and len(j[2]["seeds"]) == b5_seeds for j in jobs)
    assert sum(s.get("clips", 0) for j in jobs for s in j) == 100 * 3 + 30 * b5_seeds

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
print("run_pilot job checks passed")
