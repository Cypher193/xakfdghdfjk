"""Scene selection shared by pilot.generate and the pod scripts, importable without torch.

    python -m pilot.shard --exp b1 --shard 3/7 --write-shard data/scenes/shard_3
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml


def scene_list(cfg, exp):
    e = cfg["experiments"][exp]
    if e["scenes"] == "real":
        return json.loads(Path("data/real/manifest.json").read_text(encoding="utf-8"))
    files = sorted((Path(cfg["paths"]["data"]) / "scenes" / e["scenes"]).glob("*.json"))
    specs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if e["scenes"] == "kmain":      # interleave families so --limit N is balanced: cv_00, ca_00, cv_01, ...
        cv = [s for s in specs if s["family"] == "cv"]
        ca = [s for s in specs if s["family"] != "cv"]
        specs = [s for pair in zip(cv, ca) for s in pair] + cv[len(ca):] + ca[len(cv):]
    return specs                    # kgrav is sorted by scene then gravity, so --limit 3k = k whole scenes


def shard(scenes, spec):
    """spec 'K/N' (1-based): every N-th scene starting at the K-th."""
    k, n = map(int, spec.split("/"))
    return scenes[k - 1::n]


def write_shard(cfg, exp, scenes, out):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    src = Path(cfg["paths"]["data"]) / "scenes" / cfg["experiments"][exp]["scenes"]
    for s in scenes:
        shutil.copy(src / f"{s['id']}.json", out)
    print(f"{len(scenes)} scenes -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--write-shard", required=True)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    write_shard(cfg, a.exp, shard(scene_list(cfg, a.exp)[: a.limit or None], a.shard), a.write_shard)
