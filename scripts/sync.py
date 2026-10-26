"""Keep pilot data on a private Hugging Face dataset repo, so any pod (any datacenter) can be thrown away.

    python scripts/sync.py pull                 # at pod start: renders, scenes, runs, results, logs
    python scripts/sync.py push                 # once
    python scripts/sync.py push --every 900     # loop: push every 15 min (run in the background)

Model weights are not synced (re-download from the Hub). Uses HF_TOKEN. Pushes are resumable and only
upload changed files. Repo: HF_DATA_REPO, default <your HF user>/cvpr-vail-pilot.
LTX pods (LTX_JOB=gen1..gen4|probe|probe2|sKofN, scripts/ltx_pod.sh): PATTERNS (pull) = inputs + the job's runs, logs and
results (probe, probe2: also every runs/*/ltx2b; shards sKofN: only runs/*/ltx2b/*/meta.json and score.json, so
PATTERNS is not a superset of PUSH_PATTERNS); PUSH_PATTERNS = only the job's own files.
Screen pods (SCREEN_TIER=A|B|all, set by scripts/screen_pod.sh) pull the inputs plus their own runs, logs and table
and push only those: two pods share the repo, and a pod must never re-upload a stale pulled copy of the other's files.
"""
import argparse
import os
import re
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, whoami

PATTERNS = ["data/scenes/**", "data/renders/**", "data/logs/**", "data/real/**", "runs/**", "results/**"]


def repo_id():
    return os.environ.get("HF_DATA_REPO") or f"{whoami()['name']}/cvpr-vail-pilot"


IGNORE = ["**/*.partial.mp4", "**/*.tmp"]      # clips still being written
ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("SCREEN_TIER"):
    from run_pilot import screen_models, tag
    t = tag("screen", os.environ["SCREEN_TIER"])
    PATTERNS = ["data/scenes/**", "data/renders/**", "data/real/**", f"data/logs/*{t}*", f"results/*{t}*"] + \
        [f"runs/*/{m}/**" for m in screen_models(os.environ["SCREEN_TIER"])]
elif os.environ.get("LTX_JOB"):
    from run_pilot import LTX_PULL_EXTRA, LTX_RUNS, tag
    j = os.environ["LTX_JOB"]
    base = ["data/scenes/**", "data/renders/**", "data/real/**"]
    mine = [f"data/logs/*{tag('ltx', j)}*", f"results/*{tag('ltx', j)}*"]
    PUSH_PATTERNS = base + mine + [f"{r}/**" for r in LTX_RUNS[j]] +         (["results/p6_ltx2b*"] if j.startswith("probe") else [])    # pilot.probe writes results/p6_{model}_{target}_{pool}.csv
    PATTERNS = PUSH_PATTERNS + ([f"{r}/**" for r in LTX_PULL_EXTRA] if j.startswith("probe") else [])   # pull: a superset
    if re.fullmatch(r"s\d+of\d+", j):     # phase 2 shard: only the old clips' meta/score (skip check, scorer), no videos
        PATTERNS = base + mine + [f"{r}/*/{f}" for r in LTX_RUNS[j] for f in ("meta.json", "score.json")]
        PATTERNS.append("assets_cache/**")  # Poly Haven lists + maps from the laptop: same choices as the pilot, no downloads
    if re.fullmatch(r"c9b?s\d+of\d+", j):   # phase 3 C9 / C9b: the B1 fit data (acts + meta, no videos), input renders
        PATTERNS = ["data/scenes/**", "data/renders/kmain/*_input.mp4", "runs/b1/ltx2b/*/acts.npz",
                    "runs/b1/ltx2b/*/meta.json", "results/p6_ltx2b_hspeed_obj.csv"] + mine
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]]
        # a relaunched shard: every shard's meta.json (cheap), and its own clips' video + score, so the job's score step
        # scores clips an earlier pod generated but never scored (it timed out first)
        from run_pilot import LTX_JOBS
        PATTERNS += [f"{r}/*/meta.json" for r in LTX_RUNS[j]] + \
            [f"{r}/{i}_s0/{f}" for r in LTX_RUNS[j] for i in LTX_JOBS[j][0]["ids"] for f in ("video.mp4", "score.json")]
    if j == "score2":                     # every LTX clip with meta.json but no score.json on HF, video + meta only
        BAD = {"runs/b1/ltx2b/kmain_cv_41_s2"}       # HF holds a size-inconsistent video.mp4 there: unreadable
        fs = HfApi().list_repo_files(repo_id(), repo_type="dataset")
        meta = {f.rsplit("/", 1)[0] for f in fs if re.fullmatch(r"runs/(b1|b2nl|b3)/ltx2b/[^/]+/meta\.json", f)}
        todo = sorted(meta - {f.rsplit("/", 1)[0] for f in fs if f.endswith("/score.json")} - BAD)
        PATTERNS = ["data/scenes/**"] + mine + [f"{d}/{x}" for d in todo for x in ("video.mp4", "meta.json")]
        PUSH_PATTERNS = mine + [f"{d}/score.json" for d in todo]
    if j == "bratio2":                    # renders its own inputs: scenes + Poly Haven cache in; renders + runs out
        PATTERNS = ["data/scenes/kratio2/**", "assets_cache/**"] + mine + [f"{r}/**" for r in LTX_RUNS[j]]
        PUSH_PATTERNS = ["data/renders/kratio2/**"] + mine + [f"{r}/**" for r in LTX_RUNS[j]]
    if j == "probe2":                    # pilot.probe reads acts.npz + meta.json + the scene specs only, no videos/renders
        PATTERNS = ["data/scenes/**"] + mine + [f"{r}/*/{f}" for r in LTX_PULL_EXTRA for f in ("acts.npz", "meta.json")]
PUSH_PATTERNS = globals().get("PUSH_PATTERNS", PATTERNS)       # what push uploads and finalize verifies


def pull():
    for attempt in range(4):    # xet transfers fail now and then ("File size mismatch", probe2 setup); resumes on retry
        try:
            snapshot_download(repo_id(), repo_type="dataset", local_dir=ROOT, allow_patterns=PATTERNS)
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"pull failed ({type(e).__name__}: {str(e)[:120]}), retrying", flush=True)
            time.sleep(30)
    print("pulled", repo_id(), flush=True)


def push():
    if not any((ROOT / p.split("/**")[0]).exists() for p in PUSH_PATTERNS):
        print("nothing to push yet", flush=True)
        return
    HfApi().upload_large_folder(repo_id=repo_id(), repo_type="dataset", folder_path=ROOT,
                                allow_patterns=PUSH_PATTERNS, ignore_patterns=IGNORE, private=True, print_report=False)
    print(time.strftime("%H:%M:%S"), "pushed", repo_id(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "push"])
    ap.add_argument("--every", type=int, default=0, help="seconds between pushes (loop)")
    a = ap.parse_args()
    if a.cmd == "pull":
        pull()
    else:
        while True:
            try:
                push()
            except Exception as e:          # a failed push must not kill the loop; the next one retries
                print("push failed:", e, flush=True)
            if not a.every:
                break
            time.sleep(a.every)
