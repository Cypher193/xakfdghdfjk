"""Backstop monitor for job pods while the laptop is off (runs on the cheapest vast instance, label kvs-rpmon).

Every 5 min:
- RunPod (team account, RUNPOD_MONITOR_KEY: a separate key the user revokes after the run): every pod that is named
  kvs-* AND listed in data/logs/runpod_caps.json on HF (written by scripts/runpod_launch.py) is terminated when
  (a) its final_check_ltx_{job}.json on HF is verified and was first seen >= 15 min ago, or (b) its hard cap passed.
  A stopped (EXITED) pod without a verified check is left alone: its disk holds data that never reached HF.
- vast (MONITOR_VAST_KEY, the team env var): every kvs-* worker whose final check is verified, first seen >= 15 min
  ago, and still listed is destroyed (vastai exits 0 on a failed in-pod destroy: 2026-11-04).
- Status to HF every 30 min (data/logs/rpmon_status.json: pods, actions; no secrets). With no kvs-* pod anywhere for
  2 h, it destroys its own instance (CONTAINER_ID / CONTAINER_API_KEY).
"""
import datetime as dt
import json
import os
import re
import subprocess
import time

import requests
from huggingface_hub import HfApi, hf_hub_download

API = "https://rest.runpod.io/v1"
HF_REPO = os.environ.get("HF_DATA_REPO", "CypherNotHere/cvpr-vail-pilot")
hf = HfApi()
first_seen, actions, last_status, last_busy = {}, [], 0.0, time.time()


def now():
    return dt.datetime.now(dt.timezone.utc)


def log(msg):
    line = f"{now():%Y-%m-%d %H:%M:%S}Z {msg}"
    print(line, flush=True)
    actions.append(line)


def verified(job, files):
    f = f"data/logs/final_check_ltx_{job}.json"
    if f not in files:
        return False
    try:
        c = json.load(open(hf_hub_download(HF_REPO, f, repo_type="dataset", force_download=True)))
        return bool(c.get("verified")) and c.get("missing") == 0
    except Exception:
        return False


def due(key):
    """True once a verified check has been seen for 15 min (time for the pod to end itself first)."""
    first_seen.setdefault(key, time.time())
    return time.time() - first_seen[key] >= 900


def runpod(files):
    hdr = {"Authorization": f"Bearer {os.environ['RUNPOD_MONITOR_KEY']}"}
    caps = {}
    if "data/logs/runpod_caps.json" in files:
        caps = json.load(open(hf_hub_download(HF_REPO, "data/logs/runpod_caps.json", repo_type="dataset", force_download=True)))
    pods = requests.get(f"{API}/pods", headers=hdr, timeout=60).json()
    ours = [p for p in pods if str(p.get("name", "")).startswith("kvs-") and p["id"] in caps]
    for p in ours:
        cap = caps[p["id"]]
        why = None
        if verified(cap["job"], files) and due(p["id"]):
            why = "final check verified 15 min ago"
        elif now() > dt.datetime.fromisoformat(cap["terminate_after"]):
            why = f"hard cap {cap['terminate_after']} passed"
        if why:
            r = requests.post("https://api.runpod.io/graphql", headers=hdr, timeout=60,   # GraphQL: proven on the smoke pod
                              json={"query": f'mutation {{ podTerminate(input: {{podId: "{p["id"]}"}}) }}'})
            log(f"runpod terminate {p['id']} {p.get('name')} ({why}): HTTP {r.status_code} {r.text[:80]}")
    return [dict(id=p["id"], name=p.get("name"), status=p.get("desiredStatus"), cost=p.get("costPerHr")) for p in ours]


def vast_job(label):
    m = re.fullmatch(r"kvs-(?:ltx)?(.+?)-(\d+)of(\d+)", label)
    if not m:
        return None
    return m[1] if re.search(r"\d+of\d+$", m[1]) or (m[2], m[3]) == ("1", "1") else f"{m[1]}{m[2]}of{m[3]}"


def vast(files):
    env = dict(os.environ, VAST_API_KEY=os.environ["MONITOR_VAST_KEY"])
    out = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True, env=env, timeout=120).stdout
    insts = [i for i in json.loads(out or "[]") if str(i.get("label", "")).startswith("kvs-")
             and str(i.get("id")) != os.environ.get("CONTAINER_ID")]
    for i in insts:
        job = vast_job(i["label"])
        if job and verified(job, files) and due(f"vast{i['id']}"):
            r = subprocess.run(["vastai", "destroy", "instance", str(i["id"]), "-y", "--raw"], capture_output=True,
                               text=True, env=env, timeout=120)
            log(f"vast destroy {i['id']} {i['label']} (final check verified 15 min ago): {(r.stdout + r.stderr).strip()[:120]}")
    return [dict(id=i["id"], label=i["label"], status=i.get("actual_status"), cost=i.get("dph_total")) for i in insts]


def self_destroy():
    env = dict(os.environ, VAST_API_KEY=os.environ["CONTAINER_API_KEY"])
    log("no kvs-* pods for 2 h: destroying this monitor")
    subprocess.run(["vastai", "destroy", "instance", os.environ["CONTAINER_ID"], "-y"], env=env, timeout=120)


while True:
    rp, vs = [], []
    try:
        files = set(hf.list_repo_files(HF_REPO, repo_type="dataset"))
        rp, vs = runpod(files), vast(files)
    except Exception as e:
        log(f"check failed: {type(e).__name__}: {str(e)[:200]}")
    if rp or vs:
        last_busy = time.time()
    if time.time() - last_status >= 1800:
        try:
            hf.upload_file(path_or_fileobj=json.dumps(dict(at=f"{now():%Y-%m-%dT%H:%M}Z", runpod=rp, vast=vs,
                                                           actions=actions[-50:]), indent=1).encode(),
                           path_in_repo="data/logs/rpmon_status.json", repo_id=HF_REPO, repo_type="dataset",
                           commit_message="rpmon status")
            last_status = time.time()
        except Exception as e:
            print(f"status upload failed: {e}", flush=True)
    if time.time() - last_busy > 7200 and os.environ.get("CONTAINER_API_KEY"):
        self_destroy()
    time.sleep(300)
