"""Crash guard for a pod: if the pod's work stops without stopping the pod, push everything to HF and stop the pod.

    setsid nohup python scripts/guard.py --tier A --hard-at 2026-10-22T12:36Z > /workspace/guard.out 2>&1 < /dev/null &
    python scripts/guard.py --tag _full_cosmos2b_3 --hard-at ... --monitor http://IP:PORT   # full-run shard, with beats
    python scripts/guard.py --tier A --hard-at ... --dry-run      # print what the guard sees (the beat payload), do nothing

--tier X is --tag _screen_X plus the screen sync patterns. Every 30 s: when none of the pod's processes is alive (setup,
screen_pod.sh, run_pilot.py, finalize.py, restart2.sh), or the time is 30 min past --hard-at (finalize hung), it pushes
(sync.py push, at most 20 min), verifies on HF, and stops the pod with finalize.stop_pod (vast.ai: destroyed only if
verified, else stopped). RunPod: deleting the stopped pod needs the account key (laptop side).
A normal finish never triggers it: finalize stops the pod first, which also ends the guard.
With --monitor URL (default env MONITOR_URL) each loop also POSTs a heartbeat to URL/hb (docs/monitor_design.md); a
failed beat is ignored and logged once per change.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIVE = ["setup_runpod.sh", "setup_ext.sh", "screen_pod.sh", "ltx_pod.sh", "run_pilot.py", "finalize.py", "restart2.sh"]
CG = "/sys/fs/cgroup/"


def alive():
    try:
        return [p for p in ALIVE if subprocess.run(["pgrep", "-f", p], capture_output=True).returncode == 0]
    except FileNotFoundError:                         # no pgrep (laptop dry run)
        return []


def read(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def sample():
    """Raw counters for the delta metrics: cpu (cgroup v2 usec, else /proc/stat), net bytes, monotonic time."""
    s = dict(t=time.monotonic(), cg=None, cpu=None, net=None)
    if (cs := read(CG + "cpu.stat")) and (m := re.search(r"usage_usec (\d+)", cs)):
        s["cg"] = int(m[1])
    elif st := read("/proc/stat"):
        f = [int(x) for x in st.splitlines()[0].split()[1:9]]
        s["cpu"] = (f[3] + f[4], sum(f))
    if nd := read("/proc/net/dev"):
        rt = [(int(f[0]), int(f[8])) for n, r in (l.split(":", 1) for l in nd.splitlines()[2:] if ":" in l)
              if n.strip() != "lo" and len(f := r.split()) > 8]
        s["net"] = (sum(r for r, _ in rt), sum(t for _, t in rt))
    return s


def cpu_pct(p, c):
    if p["cg"] is not None and c["cg"] is not None:
        mx = (read(CG + "cpu.max") or "max").split()
        cpus = int(mx[0]) / int(mx[1]) if mx[0] != "max" else os.cpu_count()
        return (c["cg"] - p["cg"]) / ((c["t"] - p["t"]) * 1e6 * cpus) * 100
    if p["cpu"] and c["cpu"] and c["cpu"][1] > p["cpu"][1]:
        return (1 - (c["cpu"][0] - p["cpu"][0]) / (c["cpu"][1] - p["cpu"][1])) * 100


def mbps(p, c, i):
    if p["net"] and c["net"] and c["net"][i] >= p["net"][i]:
        return (c["net"][i] - p["net"][i]) * 8 / (c["t"] - p["t"]) / 1e6


def ram():
    """(used, total) bytes: cgroup v2 first, else /proc/meminfo."""
    mi = read("/proc/meminfo")
    tot = int(re.search(r"MemTotal:\s+(\d+)", mi)[1]) * 1024 if mi else None
    cur, mx = (read(CG + "memory.current") or "").strip(), (read(CG + "memory.max") or "max").strip()
    if cur.isdigit():
        return int(cur), int(mx) if mx.isdigit() else tot
    if mi:
        return tot - int(re.search(r"MemAvailable:\s+(\d+)", mi)[1]) * 1024, tot
    return None, None


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        n, u, mu, mt = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        return n, int(u), int(mu) / 1024, int(mt) / 1024
    except Exception:                                 # no nvidia-smi / no GPU
        return None, None, None, None


def step_of(text):
    """Name of the last START with no later END of it; 'done' after PILOT DONE; else None."""
    cur = None
    for line in text.splitlines():
        if m := re.search(r"\bSTART (\S+) ", line):
            cur = m[1]
        elif (m := re.search(r"\bEND (\S+):", line)) and m[1] == cur:
            cur = None
    return cur or ("done" if "PILOT DONE" in text else None)


def collect(st, status, prev, cur):
    """The heartbeat payload; st holds the pilot-log offset and the labels. Returns (payload, new log offset)."""
    r = lambda x, n=1: None if x is None else round(x, n)
    tag, logs = st["tag"], ROOT / "data/logs"
    name, util, vram, vram_t = gpu()
    used, tot = ram()
    pl = read(logs / f"pilot{tag}.log")
    off = st["off"]
    if pl is None:
        events, new_off = [], off
    else:
        b = pl.encode()
        if off is None:
            events, new_off = pl.splitlines()[-20:], len(b)
        else:
            off = 0 if off > len(b) else off          # file shrank: start over
            events, new_off = b[off:].decode(errors="replace").splitlines(), len(b)
    bud = {}
    try:
        bud = json.loads(read(logs / f"budget{tag}.json") or "{}")
    except ValueError:
        pass
    t0 = read(logs / f"pilot_start{tag}")
    steps = sorted(logs.glob(f"step{tag}_*.log"), key=lambda p: p.stat().st_mtime)
    m = re.match(r"kvs-([a-z0-9_]+)-(\d+)of(\d+)$", st["label"])
    p = dict(inst=int(os.environ.get("CONTAINER_ID") or 0), label=st["label"],
             block=m[1] if m else None, shard=f"{m[2]}/{m[3]}" if m else None,
             host=socket.gethostname(), gpu=name, status=status,
             step=step_of(pl) if pl else None,
             clips_done=len(list((ROOT / "runs").glob("*/*/*/meta.json"))) if t0 else None,
             clips_total=bud.get("clips_total"),
             used_h=r((time.time() - float(t0)) / 3600, 2) if t0 else None, budget_h=bud.get("budget_h"),
             gpu_util=util, vram_gb=r(vram), vram_total_gb=r(vram_t),
             cpu_pct=r(cpu_pct(prev, cur)), ram_gb=r(used and used / 2**30), ram_total_gb=r(tot and tot / 2**30),
             disk_gb=r(shutil.disk_usage(ROOT).free / 2**30), net_rx_mbps=r(mbps(prev, cur, 0)),
             net_tx_mbps=r(mbps(prev, cur, 1)), events=events,
             tail="\n".join((read(steps[-1]) or "").splitlines()[-30:]) if steps else "")
    return p, new_off


def post(url, payload):
    req = urllib.request.Request(url.rstrip("/") + "/hb", json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("MONITOR_TOKEN", "")})
    urllib.request.urlopen(req, timeout=5).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B", "C", "all"], help="screen pod: implies --tag _screen_TIER")
    ap.add_argument("--tag", help="log suffix, e.g. _full_cosmos2b_3")
    ap.add_argument("--hard-at", required=True, help="UTC, finalize's --hard-at")
    ap.add_argument("--grace-min", type=float, default=30)
    ap.add_argument("--monitor", default=os.environ.get("MONITOR_URL", ""), help="heartbeat server URL; empty = none")
    ap.add_argument("--label", default=os.environ.get("VAST_CONTAINERLABEL", ""))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.tier or a.tag):
        ap.error("one of --tier / --tag is required")
    tag = a.tag or f"_screen_{a.tier}"
    if a.tier:
        os.environ["SCREEN_TIER"] = a.tier           # before importing sync (via finalize): it sets the push patterns
    else:
        os.environ["FULL_TAG"] = tag
    sys.path.insert(0, str(ROOT / "scripts"))
    import finalize
    finalize.LOG = ROOT / f"data/logs/guard{tag}.log"
    deadline = finalize.ts(a.hard_at) + a.grace_min * 60
    label = a.label if a.label and not re.match(r"C\.\d+$", a.label) else "kvs-" + tag.strip("_").replace("_", "-")
    st = dict(tag=tag, label=label, off=None, last=None)

    def beat(status, prev):
        """Collect and POST one heartbeat; never raises. Returns the sample to diff against next time."""
        cur = sample()
        try:
            payload, off = collect(st, status, prev, cur)
            if a.dry_run:
                print(json.dumps(payload, indent=1))
            if a.monitor:
                post(a.monitor, payload)
                st["off"] = off
            res = "ok"
        except Exception as e:
            res = f"failed: HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else f"failed: {type(e).__name__}"
        if a.monitor and res != st["last"]:
            msg = {"ok": "monitor beats ok" if st["last"] is None else "beat ok again"}.get(res, f"beat {res}")
            if a.dry_run:
                print(msg)
            else:
                finalize.log(msg)
            st["last"] = res
        return cur

    if a.dry_run:
        print(f"alive: {alive()}; deadline {datetime.fromtimestamp(deadline, timezone.utc):%Y-%m-%d %H:%M}Z")
        first = sample()
        time.sleep(1)
        beat("dry-run", first)
        return
    if not (os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID")):   # pushes to the shared HF dataset:
        sys.exit("run on a RunPod pod or vast.ai instance (RUNPOD_POD_ID / CONTAINER_ID unset)")   # never from a laptop
    finalize.log(f"guard armed: tag {tag}, deadline {datetime.fromtimestamp(deadline, timezone.utc):%H:%M}Z"
                 f"{', beats to ' + a.monitor if a.monitor else ''}")
    prev = sample()
    while (procs := alive()) and time.time() < deadline:
        if a.monitor:
            names = " ".join(procs)
            prev = beat("running" if "run_pilot.py" in names else "finalize" if "finalize.py" in names else   # finalize
                        "setup" if "setup" in names and len(procs) == 1 else "running", prev)             # waits alongside
        time.sleep(30)
    finalize.log(f"guard triggered: {'deadline passed' if procs else 'no pod process alive'} (alive: {procs})")
    if a.monitor:
        beat("triggered", prev)
    for p in procs:                                   # deadline case: stop the hung processes before pushing
        subprocess.run(["pkill", "-f", p])
    finalize.run([sys.executable, "scripts/sync.py", "push"], 1200, "guard upload")
    check = {}
    try:
        check = finalize.verify()
        finalize.log(f"guard verify: {check}")
    except Exception as e:
        finalize.log(f"guard verify failed: {type(e).__name__}: {e}")
    finalize.stop_pod(check.get("missing") == 0)


if __name__ == "__main__":
    main()
