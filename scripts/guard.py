"""Crash guard for a pod: if the pod's work stops without stopping the pod, push everything to HF and stop the pod.

    setsid nohup python scripts/guard.py --tier A --hard-at 2026-10-22T12:36Z > /workspace/guard.out 2>&1 < /dev/null &
    python scripts/guard.py --tier A --hard-at ... --dry-run      # print what the guard sees, do nothing

Every 60 s: when none of the pod's processes is alive (setup, screen_pod.sh, run_pilot.py, finalize.py, restart2.sh),
or the time is 30 min past --hard-at (finalize hung), it pushes (sync.py push, at most 20 min), verifies on HF, and
stops the pod with finalize.stop_pod (vast.ai: destroyed only if verified, else stopped). RunPod: deleting the
stopped pod needs the account key (laptop side).
A normal finish never triggers it: finalize stops the pod first, which also ends the guard.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIVE = ["setup_runpod.sh", "setup_ext.sh", "screen_pod.sh", "run_pilot.py", "finalize.py", "restart2.sh"]


def alive():
    return [p for p in ALIVE if subprocess.run(["pgrep", "-f", p], capture_output=True).returncode == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B", "C", "all"], required=True)
    ap.add_argument("--hard-at", required=True, help="UTC, finalize's --hard-at")
    ap.add_argument("--grace-min", type=float, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID")):   # pushes to the shared HF dataset:
        sys.exit("run on a RunPod pod or vast.ai instance (RUNPOD_POD_ID / CONTAINER_ID unset)")   # never from a laptop
    os.environ["SCREEN_TIER"] = a.tier               # before importing sync (via finalize): it sets the push patterns
    sys.path.insert(0, str(ROOT / "scripts"))
    import finalize
    finalize.LOG = ROOT / f"data/logs/guard_screen_{a.tier}.log"
    deadline = finalize.ts(a.hard_at) + a.grace_min * 60
    if a.dry_run:
        print(f"alive: {alive()}; deadline {datetime.fromtimestamp(deadline, timezone.utc):%Y-%m-%d %H:%M}Z")
        return
    finalize.log(f"guard armed: tier {a.tier}, deadline {datetime.fromtimestamp(deadline, timezone.utc):%H:%M}Z")
    while (procs := alive()) and time.time() < deadline:
        time.sleep(60)
    finalize.log(f"guard triggered: {'deadline passed' if procs else 'no pod process alive'} (alive: {procs})")
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
