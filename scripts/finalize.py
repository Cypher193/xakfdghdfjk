"""Deadline guard for the pilot pod: stop work, upload everything to HF, verify, then stop the pod.

    python scripts/finalize.py --stop-at 2026-10-18T00:40Z --hard-at 2026-10-18T01:25Z
    python scripts/finalize.py --stop-at ... --hard-at ... --plan screen --tier A      # screen pod (screen_pod.sh)
    python scripts/finalize.py --stop-at ... --hard-at ... --plan ltx --job gen1         # LTX job pod (ltx_pod.sh)

1. Waits until --stop-at, or until the pilot finishes by itself, whichever is first.
2. Stops the orchestrator and every worker (generation, scoring, probes, timing).
3. Uploads; scores the remaining clips; uploads; runs the final probes; uploads (each step time-boxed).
4. Verifies every local data file is on HF (data/logs/final_check.json), re-uploading until --hard-at.
5. Stops the pod. RunPod: ends /start.sh's `sleep infinity`, so the container exits and GPU billing stops; the pod
   then shows as Exited and is deleted with the account key (laptop side). vast.ai: destroys the instance with its own
   CONTAINER_API_KEY when HF has every file (a stopped instance still bills for its disk), else only stops it.
Runs even if verification is incomplete at --hard-at: the deadline wins, the check file says what is missing.
Screen: scores only the tier's models into its own table, no probes, log and check files suffixed _screen_{tier}.
"""
import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sync  # noqa: E402  (same patterns and repo as the regular uploads)

PY = sys.executable
WORKERS = ["restart2.sh", "restart.sh", "run_pilot.py", "pilot.generate", "pilot.steer", "pilot.invert",
           "pilot.likelihood", "pilot.score", "pilot.probe", "pilot.analysis", "pilot.track", "time_extra.sh", "scripts/ext/",
           "sync.py push --every"]                                   # the background uploader: finalize uploads now
LOG = ROOT / "data/logs/finalize.log"                               # main() adds the screen suffix


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def left(hard):
    return hard - time.time()


def run(cmd, budget_s, label):
    if budget_s < 60:
        log(f"skip {label}: {budget_s:.0f} s left")
        return
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=ROOT, timeout=budget_s, stdout=open(LOG, "a"), stderr=subprocess.STDOUT).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    log(f"{label}: rc={rc} in {time.time() - t0:.0f} s")


def local_files():
    out = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not rel.startswith(".cache") and any(fnmatch.fnmatch(rel, pat) for pat in sync.PUSH_PATTERNS) \
                and not any(fnmatch.fnmatch(rel, pat) for pat in sync.IGNORE):
            out.append(rel)
    return out


def verify():
    from huggingface_hub import HfApi
    remote = set(HfApi().list_repo_files(sync.repo_id(), repo_type="dataset"))
    loc = local_files()
    missing = [f for f in loc if f not in remote]
    return dict(local=len(loc), remote=len(remote), missing=len(missing), missing_sample=missing[:20])


def stop_pod(verified=False):
    """vast.ai: destroy the instance if every file is on HF, else stop it (its disk, and the data, are kept).
    RunPod: end /start.sh's final `sleep infinity`: start.sh exits, docker-init (PID 1) exits, the container stops."""
    if os.environ.get("CONTAINER_API_KEY"):
        cmd = ["destroy", "instance", os.environ["CONTAINER_ID"], "-y"] if verified else \
              ["stop", "instance", os.environ["CONTAINER_ID"]]
        log(f"stopping instance: vastai {' '.join(cmd)}")
        if not shutil.which("vastai"):                # setup died before installing requirements.txt
            subprocess.run([PY, "-m", "pip", "install", "-q", "vastai"])
        r = subprocess.run(["vastai", *cmd], capture_output=True, text=True,   # key via env: not in argv or the log
                           env=dict(os.environ, VAST_API_KEY=os.environ["CONTAINER_API_KEY"]))
        log(f"vastai rc={r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}")
        return r.returncode == 0
    for _ in range(6):         # some hosts run nested start.sh processes: a second one starts a new sleep, so repeat
        start = subprocess.run(["pgrep", "-f", "/start.sh"], capture_output=True, text=True).stdout.split()
        if not start:
            return True
        for pid in start:
            kids = subprocess.run(["pgrep", "-P", pid, "-f", "sleep infinity"], capture_output=True, text=True).stdout.split()
            for k in kids:
                log(f"stopping pod: kill sleep infinity (pid {k}, child of start.sh {pid})")
                subprocess.run(["kill", k])
        time.sleep(10)
    log("start.sh still running after killing its sleeps: signalling PID 1")
    subprocess.run(["kill", "-TERM", "1"])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-at", required=True, help="UTC, e.g. 2026-10-18T00:40Z")
    ap.add_argument("--hard-at", required=True, help="UTC; the pod is stopped by then whatever happens")
    ap.add_argument("--no-stop-pod", action="store_true", help="testing: do everything except stopping the pod")
    ap.add_argument("--plan", choices=["main", "screen", "ltx"], default="main")
    ap.add_argument("--job", help="plan ltx: gen1..gen4 or probe")
    ap.add_argument("--tier", choices=["A", "B", "C", "all"], default="all")
    a = ap.parse_args()
    if not (os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID")):   # pushes to the shared HF dataset:
        sys.exit("run on a RunPod pod or vast.ai instance (RUNPOD_POD_ID / CONTAINER_ID unset)")   # never from a laptop
    from run_pilot import FINAL_PROBES, score_args, tag      # same probe list and scorer scope as the orchestrator
    global LOG
    t = tag(a.plan, a.job if a.plan == "ltx" else a.tier)
    LOG = ROOT / f"data/logs/finalize{t}.log"
    stop, hard = ts(a.stop_at), ts(a.hard_at)
    log(f"armed: stop work at {a.stop_at}, pod stopped by {a.hard_at}")
    while time.time() < stop and (running("run_pilot.py") or running("restart2.sh")):
        time.sleep(30)
    log("stopping pilot workers" if running("run_pilot.py") else "pilot already finished")
    for w in WORKERS:
        subprocess.run(["pkill", "-f", w])
    time.sleep(10)
    run([PY, "scripts/sync.py", "push"], min(900, left(hard) - 300), "upload 1")
    if a.plan != "ltx":                                       # ltx: the job's last step scores (a stopped job: score by hand)
        run([PY, "-m", "pilot.score", *score_args(a.plan, a.tier)], min(900, left(hard) - 900), "final scoring pass")
    run([PY, "scripts/sync.py", "push"], min(600, left(hard) - 300), "upload 2")
    for model, exps, target in FINAL_PROBES if a.plan == "main" else []:
        run([PY, "-m", "pilot.probe", "--model", model, "--exps", *exps, "--target", target],
            min(300, left(hard) - 900), f"probe {model} {target}")
    check = {}
    while left(hard) > 120:
        run([PY, "scripts/sync.py", "push"], left(hard) - 120, "upload")
        try:
            check = verify()
        except Exception as e:                                # network hiccup: retry while time remains
            check = dict(error=f"{type(e).__name__}: {e}")
        log(f"verify: {check}")
        if check.get("missing") == 0:
            break
        # upload_large_folder keeps per-file state in .cache/huggingface/upload; a push killed between upload and
        # commit leaves files marked done that never reached the repo (gen4: 153 "missing" for 1.8 h). Forget the state.
        shutil.rmtree(ROOT / ".cache/huggingface/upload", ignore_errors=True)
        time.sleep(30)
    check.update(verified=check.get("missing") == 0, at=f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    (ROOT / f"data/logs/final_check{t}.json").write_text(json.dumps(check, indent=1))
    try:                                                      # the check file itself, then the log, last
        from huggingface_hub import HfApi
        for f in (f"data/logs/final_check{t}.json", f"data/logs/finalize{t}.log"):
            HfApi().upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=sync.repo_id(),
                                repo_type="dataset", commit_message="final check")
    except Exception as e:
        log(f"final check upload failed: {e}")
    if a.no_stop_pod:
        log("--no-stop-pod: leaving the pod running")
        return
    stop_pod(check.get("verified", False))


if __name__ == "__main__":
    main()
