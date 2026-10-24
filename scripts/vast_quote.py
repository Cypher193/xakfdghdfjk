"""Live vast.ai quote for the full Know vs Show run: which GPU class is the best deal right now, per block of work.

    python scripts/vast_quote.py                   # datacenter hosts only (no personal GPUs), on-demand prices
    python scripts/vast_quote.py --hosts verified  # also verified non-datacenter hosts (cheaper, individuals' machines)
    python scripts/vast_quote.py --bid             # interruptible (min_bid) prices: ~2x cheaper, can be pre-empted
    python scripts/vast_quote.py --pod-h 8         # target hours of generation per instance (more pods = more parallel)

Laptop only: reads the account key from VAST_API_KEY or ~/.vast_key. Prices change by the hour, so re-run before
launching. Speed factors marked "est" are guesses until a 30 min timing run on that card replaces them (docs/vast_plan.md).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

VASTAI = shutil.which("vastai") or str(Path.home() / "AppData/Roaming/Python/Python313/Scripts/vastai.exe")

# wall-time factor vs A100 SXM 80 GB (the pilot's card), bf16 diffusion. Turing/Volta cards have no bf16: not listed.
GPUS = {"A100 SXM4": (1.00, "measured"), "A100 PCIE": (1.15, "est"), "L40S": (1.27, "measured, Cosmos-14B offload"),
        "L40": (1.35, "est"), "RTX 6000Ada": (1.30, "est"), "RTX 5880Ada": (1.35, "est"), "RTX A6000": (2.00, "est"),
        "A40": (2.10, "est"), "RTX 4090": (1.30, "est"), "RTX 5090": (1.00, "est"), "RTX 4080S": (1.80, "est"),
        "RTX PRO 6000 WS": (0.70, "est"), "RTX PRO 6000 Max-Q": (0.75, "est"), "RTX PRO 6000 S": (0.70, "est"),
        "RTX PRO 5000": (1.20, "est"), "RTX PRO 4000": (2.00, "est"), "RTX 6000D": (1.00, "est"),
        "RTX 3090": (2.00, "est"), "RTX 3090 Ti": (1.90, "est"), "RTX A5000": (2.40, "est"), "RTX 4500Ada": (2.00, "est"),
        "H100 SXM": (0.50, "est"), "H100 PCIE": (0.55, "est"), "H100 NVL": (0.50, "est"), "H200": (0.45, "est"),
        "H200 NVL": (0.45, "est"), "B200": (0.30, "est")}

# A100-hours per block (docs/vast_plan.md has the derivation), minimum VRAM / host RAM in GB, model cache on disk in GB,
# output written per block in GB (videos ~5 MB/clip + activations). wave: 1 generation, 2 probes+metrics, 3 causal.
BLOCKS = [
    dict(name="cosmos2b B-runs (2,160 clips)", a100_h=146, vram=24, ram=48, cache=76, out=40, wave=1),
    dict(name="vace13 B-runs (1,680 clips)", a100_h=133, vram=24, ram=48, cache=19, out=30, wave=1),
    dict(name="ltx2b B-runs (2,160 clips)", a100_h=16, vram=24, ram=48, cache=26, out=30, wave=1),
    dict(name="cosmos14b (360 clips)", a100_h=83, vram=45, ram=64, cache=58, out=8, wave=1),
    dict(name="cosmos3_nano (300 clips + questions)", a100_h=20, vram=45, ram=64, cache=35, out=6, wave=1),
    dict(name="wan22_5b C1-inv/noise (200 clips)", a100_h=3, vram=24, ram=48, cache=34, out=2, wave=1),
    dict(name="tracking not absorbed by gen pods", a100_h=20, vram=16, ram=32, cache=3, out=1, wave=2),
    dict(name="probing C1-C6, V-JEPA 2, baselines", a100_h=80, vram=24, ram=64, cache=30, out=5, wave=2),
    dict(name="B-ratio, C2-g, MDL, V-information", a100_h=39, vram=24, ram=48, cache=76, out=5, wave=2),
    dict(name="causal C7-C9 (cosmos2b + ltx2b, 2,020 clips)", a100_h=152, vram=24, ram=48, cache=102, out=20, wave=3),
]
SETUP_H = 0.3          # image pull, pip, data pull, before the model download (measured ~15-25 min on RunPod/vast)
DISK_BASE_GB = 40      # image layers, repo, renders, venvs; the disk is reserved at creation and billed whether used or not


def disk_gb(block):
    return block["cache"] + block["out"] + DISK_BASE_GB


def offers(hosts, bid):
    q = ("num_gpus=1 verified=true rentable=true reliability>0.98 inet_down>=400 duration>2 cuda_vers>=12.8 "
         "gpu_ram>=16 disk_space>=100" + (" datacenter=true" if hosts == "datacenter" else ""))
    cmd = [VASTAI, "search", "offers", q, "-o", "dph", "--raw", "--limit", "1000"] + (["--bid"] if bid else [])
    env = dict(os.environ)
    if "VAST_API_KEY" not in env and (Path.home() / ".vast_key").exists():
        env["VAST_API_KEY"] = (Path.home() / ".vast_key").read_text().strip()
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8-sig", env=env).stdout
    return json.loads(out.lstrip("﻿") or "[]")


def fits(block, o):
    return (o["gpu_ram"] / 1024 >= block["vram"] - 1 and o["cpu_ram"] / 1024 >= block["ram"] - 2
            and o["disk_space"] >= disk_gb(block))


def quote(block, gpu, o, n, pod_h, bid):
    """Cost of one block on one GPU class at one offer's prices (compute, reserved disk, bandwidth)."""
    factor = GPUS[gpu][0]
    gpu_h = block["a100_h"] * factor
    pods = max(1, min(math.ceil(gpu_h / pod_h), n))                   # bounded by how many offers exist right now
    dl_h = block["cache"] * 8 / o["inet_down"] / 3600                 # model download at the host's line speed
    hours = gpu_h + pods * (SETUP_H + dl_h)
    price = o["min_bid"] if bid else o["dph_base"]
    compute = hours * price
    storage = hours * disk_gb(block) * o["storage_cost"] / 730       # storage_cost is $/GB/month
    bandwidth = pods * block["cache"] * o["inet_down_cost"] + block["out"] * o["inet_up_cost"]
    return dict(gpu=gpu, price=price, n=n, pods=pods, wall_h=hours / pods, hours=hours, compute=compute,
                storage=storage, bandwidth=bandwidth, total=compute + storage + bandwidth, factor=factor, id=o["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", choices=["datacenter", "verified"], default="datacenter")
    ap.add_argument("--bid", action="store_true", help="interruptible pricing (min_bid)")
    ap.add_argument("--pod-h", type=float, default=10, help="target generation hours per instance")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    by_gpu = defaultdict(list)
    for o in offers(a.hosts, a.bid):
        if o["gpu_name"] in GPUS:
            by_gpu[o["gpu_name"]].append(o)
    print(f"vast.ai {a.hosts} hosts, {'interruptible' if a.bid else 'on-demand'}, {time.strftime('%Y-%m-%d %H:%M')}, "
          f"{sum(map(len, by_gpu.values()))} offers in {len(by_gpu)} GPU classes (reliability > 0.98, >= 400 Mb/s, "
          f"CUDA >= 12.8; disk sized per block = model cache + outputs + {DISK_BASE_GB} GB)\n")

    print(f"{'GPU':20} {'n':>3} {'VRAM':>5} {'RAM':>5} {'$/h':>6} {'x A100':>7} {'$/A100-h':>9}  {'store $/GB/mo':>13} "
          f"{'dl Mb/s':>8} {'dl $/TB':>8}  factor")
    cheapest = {}
    for gpu, os_ in sorted(by_gpu.items(), key=lambda kv: min(x["min_bid" if a.bid else "dph_base"] for x in kv[1])):
        o = cheapest[gpu] = min(os_, key=lambda x: x["min_bid" if a.bid else "dph_base"])
        f, src = GPUS[gpu]
        p = o["min_bid"] if a.bid else o["dph_base"]
        print(f"{gpu:20} {len(os_):3} {o['gpu_ram'] / 1024:5.0f} {o['cpu_ram'] / 1024:5.0f} {p:6.3f} {f:7.2f} {p * f:9.3f}"
              f"  {o['storage_cost']:13.3f} {o['inet_down']:8.0f} {o['inet_down_cost'] * 1000:8.2f}  {src}")

    print(f"\nPer block (top {a.top} GPU classes by total $, each at its best offer; pods = parallel instances, "
          f"wall = hours per instance incl. setup):")
    best_total = best_wall = 0
    for b in BLOCKS:
        qs = []
        for g, os_ in by_gpu.items():
            ok = [quote(b, g, o, len(os_), a.pod_h, a.bid) for o in os_ if fits(b, o)]
            if ok:
                qs.append(min(ok, key=lambda q: q["total"]))
        qs.sort(key=lambda q: q["total"])
        print(f"\n  wave {b['wave']}  {b['name']}  ({b['a100_h']} A100-h, >= {b['vram']} GB VRAM, {b['ram']} GB RAM, "
              f"{disk_gb(b)} GB disk)")
        for q in qs[:a.top] or [None]:
            if q is None:
                print("    no offer fits")
                continue
            print(f"    {q['gpu']:20} ${q['total']:7.2f}  = compute {q['compute']:6.2f} + storage {q['storage']:5.2f} "
                  f"+ bandwidth {q['bandwidth']:5.2f}   {q['pods']:2} pods x {q['wall_h']:5.1f} h  "
                  f"(${q['price']:.3f}/h, {q['n']} offers, offer {q['id']})")
        if qs:
            best_total += qs[0]["total"]
            best_wall = max(best_wall, qs[0]["wall_h"]) if b["wave"] == 1 else best_wall
    print(f"\nSum of per-block best: ${best_total:,.0f}   (longest wave-1 instance: {best_wall:.1f} h; waves 2 and 3 "
          f"follow). Plus ~10% for reruns, and HF dataset storage (private repos: 100 GB free, then $25/TB/month).")


if __name__ == "__main__":
    main()
