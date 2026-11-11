"""GPU health check, run on the pod before setup: abandon throttled or capped hosts (user policy, 2026-11-11).

    python scripts/gpu_check.py --tag _ltx_JOB        # exit 0 = healthy, 4 = abandon the host

Checks, each against the card's own limits:
- sustained bf16 matmul throughput (8192^2, about 30 s) >= MIN_FRAC of the card's dense bf16 spec
- power limit not lowered by the host: enforced limit >= 0.9 x default limit
- under load: mean power draw >= 0.6 x enforced power limit, and mean SM clock >= 0.75 x max SM clock unless
  the card is at its power limit (>= 0.9 x), where low clocks are normal
- no hardware or thermal slowdown reported during the load
- host RAM >= --min-ram-gb (CPU offload of a 14B model keeps the weights in host RAM)
Writes data/logs/gpucheck{tag}.json, which the pod's normal sync pushes to HF.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

# Dense bf16 tensor TFLOPS from NVIDIA data sheets (no sparsity). MIN_FRAC is loose: well-run cards reach 0.6-0.8 of
# spec on this matmul. ponytail: floors are from spec sheets, not measured per card; tighten after the first runs.
SPEC_TFLOPS = {"A40": 149.7, "RTX A6000": 154.8, "L40S": 362.0, "RTX 6000 Ada": 364.0, "A100": 312.0,
               "H100": 756.0, "RTX 3090": 71.0, "RTX 4090": 165.0}
MIN_FRAC = 0.45
BAD_REASONS = ("hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown", "hw_power_brake_slowdown")
Q = ["name", "power.limit", "power.default_limit", "power.draw", "clocks.sm", "clocks.max.sm", "temperature.gpu"] + \
    [f"clocks_throttle_reasons.{r}" for r in BAD_REASONS]


def smi():
    q = Q
    r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(q)}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if r.returncode:                                  # driver without the throttle-reason fields: read the rest
        q = Q[:7]
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(q)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    row = {k: "not supported" for k in Q} | dict(zip(q, (v.strip() for v in r.stdout.splitlines()[0].split(","))))
    for k in Q[1:7]:
        try:
            row[k] = float(row[k])
        except ValueError:
            row[k] = float("nan")
    return row


def spec_for(name):
    return next((v for k, v in sorted(SPEC_TFLOPS.items(), key=lambda kv: -len(kv[0])) if k in name), None)


def bench(seconds):
    import torch
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0, it = time.time(), 0
    while time.time() - t0 < seconds:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        it += 10
    return 2 * n ** 3 * it / (time.time() - t0) / 1e12


def check(seconds, min_ram_gb):
    idle = smi()
    samples, done = [], threading.Event()

    def sample():
        time.sleep(seconds * 0.3)                   # skip the clock ramp-up
        while not done.is_set():
            samples.append(smi())
            time.sleep(1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    tflops = bench(seconds)
    done.set()
    th.join()
    mean = lambda k: sum(s[k] for s in samples) / max(len(samples), 1)
    spec = spec_for(idle["name"])
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    flags = sorted({r for s in samples for r in BAD_REASONS if s[f"clocks_throttle_reasons.{r}"].lower() == "active"})
    r = dict(gpu=idle["name"], tflops=round(tflops, 1), spec_tflops=spec, power_limit=idle["power.limit"],
             power_default=idle["power.default_limit"], power_draw_load=round(mean("power.draw"), 1),
             sm_clock_load=round(mean("clocks.sm")), sm_clock_max=idle["clocks.max.sm"],
             temp_max=max((s["temperature.gpu"] for s in samples), default=float("nan")),
             slowdown=flags, host_ram_gb=round(ram_gb), samples=len(samples))
    fails = []
    if spec is None:
        fails.append(f"no spec for {idle['name']}")
    elif tflops < MIN_FRAC * spec:
        fails.append(f"bf16 {tflops:.0f} TFLOPS < {MIN_FRAC} x spec {spec}")
    if idle["power.limit"] < 0.9 * idle["power.default_limit"]:
        fails.append(f"power limit {idle['power.limit']} W lowered from {idle['power.default_limit']} W")
    # a card held at its power limit runs well below its boost clock under matmul load (L40S: 1239 of 2520 MHz at
    # 350 of 350 W, 2026-11-11), so a low clock counts only when the card is not drawing its power limit
    power_bound = r["power_draw_load"] >= 0.9 * idle["power.limit"]
    if r["sm_clock_load"] < 0.75 * idle["clocks.max.sm"] and not power_bound:
        fails.append(f"SM clock {r['sm_clock_load']} MHz under load < 0.75 x max {idle['clocks.max.sm']}, "
                     f"while drawing only {r['power_draw_load']} of {idle['power.limit']} W")
    if r["power_draw_load"] < 0.6 * idle["power.limit"]:
        fails.append(f"power draw {r['power_draw_load']} W under load < 0.6 x limit {idle['power.limit']}")
    if flags:
        fails.append(f"slowdown reported: {', '.join(flags)}")
    if ram_gb < min_ram_gb:
        fails.append(f"host RAM {ram_gb:.0f} GB < {min_ram_gb}")
    r["fails"], r["ok"] = fails, not fails
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--min-ram-gb", type=float, default=0)
    a = ap.parse_args()
    r = check(a.seconds, a.min_ram_gb)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path(f"data/logs/gpucheck{a.tag}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r), flush=True)
    print("GPU CHECK", "OK" if r["ok"] else "FAIL: " + "; ".join(r["fails"]), flush=True)
    raise SystemExit(0 if r["ok"] else 4)


if __name__ == "__main__":
    main()
