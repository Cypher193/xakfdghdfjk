"""Laptop-side launcher for RunPod job pods (team account; key in ~/.runpod_key, never printed or passed in argv).

    python scripts/runpod_launch.py create cosmos3_nanosmoke --gpu "NVIDIA A40" --budget-h 1.5
    python scripts/runpod_launch.py create cosmos14bprobe --cpu cpu3g --vcpu 16 --budget-h 6 --max-price 1.0
    python scripts/runpod_launch.py ls
    python scripts/runpod_launch.py terminate POD_ID -y

- A pod runs scripts/ltx_pod.sh JOB BUDGET_H, as a vast worker does, from a start command (the image has no RunPod
  start.sh): sshd with the laptop's public key, one clone, the job, then `sleep infinity`. A marker on the container
  disk keeps a restarted container from running the job twice.
- HF_TOKEN comes from the account's RunPod secret HF_TOKEN. The pod ends itself with its pod-scoped RUNPOD_API_KEY
  (finalize.runpod_stop); scripts/rpmon.py is the backstop. Hard caps per pod go to data/logs/runpod_pods.json and to
  HF (data/logs/runpod_caps.json) so the monitor can enforce them.
- Only pods named kvs-* AND recorded in data/logs/runpod_pods.json are ever terminated: the account is shared.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import requests

API = "https://rest.runpod.io/v1"
KEY = Path.home() / ".runpod_key"
ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data/logs/runpod_pods.json"
IMAGE = "ghcr.io/cypher193/cvpr_vail:latest"
REPO = "https://github.com/Cypher193/xakfdghdfjk.git"
HF_REPO = "CypherNotHere/cvpr-vail-pilot"
START = ('mkdir -p ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys; '
         'ssh-keygen -A >/dev/null 2>&1; (service ssh start || /usr/sbin/sshd) >/dev/null 2>&1; '
         'if [ ! -f /workspace/.kvs_started ]; then mkdir -p /workspace && touch /workspace/.kvs_started && cd /workspace && '
         'git clone {repo} cvpr_vail && cd cvpr_vail && setsid nohup bash scripts/ltx_pod.sh {job} {budget} '
         '> /workspace/ltx.out 2>&1 < /dev/null & fi; sleep infinity')


def hdr():
    return {"Authorization": f"Bearer {KEY.read_text().strip()}"}


def records():
    return json.loads(RECORD.read_text()) if RECORD.exists() else []


def push_caps(rows):
    """Hard caps for the monitor: {pod id: {job, name, terminate_after}} on HF, no secrets."""
    from huggingface_hub import HfApi
    caps = {r["id"]: dict(job=r["job"], name=r["name"], terminate_after=r["terminate_after"]) for r in rows}
    HfApi().upload_file(path_or_fileobj=json.dumps(caps, indent=1).encode(), path_in_repo="data/logs/runpod_caps.json",
                        repo_id=HF_REPO, repo_type="dataset", commit_message="runpod pod caps")


def create(a):
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_pilot import LTX_JOBS
    if a.job not in LTX_JOBS:
        sys.exit(f"unknown job {a.job}")
    pub = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    body = dict(name=f"kvs-{a.job}", imageName=IMAGE, gpuTypeIds=[a.gpu], gpuCount=1, cloudType=a.cloud,
                containerDiskInGb=a.disk, volumeInGb=0, ports=["22/tcp"], minRAMPerGPU=a.ram,
                dockerStartCmd=["bash", "-c", START.format(repo=REPO, job=a.job, budget=a.budget_h)],
                env={"HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}", "HF_DATA_REPO": HF_REPO, "PUBLIC_KEY": pub})
    if a.cpu:                              # CPU-only pod (probes): no GPU fields; RAM comes with the flavor and vCPU count
        for k in ("gpuTypeIds", "gpuCount", "minRAMPerGPU"):
            body.pop(k)
        body |= dict(computeType="CPU", cpuFlavorIds=[a.cpu], vcpuCount=a.vcpu)
        a.gpu = f"CPU {a.cpu} x{a.vcpu}"
    if a.dry_run:
        print(json.dumps(body | {"env": {k: (v if k != "PUBLIC_KEY" else "<laptop public key>") for k, v in body["env"].items()}}, indent=1))
        return
    r = requests.post(f"{API}/pods", json=body, headers=hdr(), timeout=120)
    if not r.ok:
        sys.exit(f"create failed: HTTP {r.status_code}: {r.text[:300]}")
    p = r.json()
    cost = float(p.get("costPerHr") or 0)
    cap = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=a.budget_h + 3)    # ltx_pod.sh guard fires at +2 h
    rec = dict(id=p["id"], name=p.get("name"), job=a.job, gpu=a.gpu, cost_per_hr=cost,
               created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               terminate_after=cap.isoformat(timespec="seconds"))
    rows = records() + [rec]
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rows, indent=1))
    if cost > a.max_price:
        terminate_one(p["id"], f"price ${cost:.3f}/h above --max-price {a.max_price}")
    try:                                   # an HF limit here must not read as a failed create (a caller then retried
        live = {q["id"] for q in pods()}   # on another GPU and made a duplicate pod, 2026-11-06)
        push_caps([x for x in rows if x["id"] in live])
    except Exception as e:
        print(f"caps not pushed ({type(e).__name__}: {str(e)[:80]}); push them later", flush=True)
    print(f"created {p['id']} {rec['name']} {a.gpu} ${cost:.3f}/h, hard cap {rec['terminate_after']}")   # last line


def pods():
    r = requests.get(f"{API}/pods", headers=hdr(), timeout=60)
    r.raise_for_status()
    return r.json()


def terminate_one(pid, why):
    mine = {r["id"] for r in records()}
    p = next((x for x in pods() if x["id"] == pid), None)
    if p is None:
        print(f"{pid}: not on the account")
        return
    if pid not in mine or not str(p.get("name", "")).startswith("kvs-"):
        sys.exit(f"{pid} ({p.get('name')}) was not created by this launcher: refusing")
    r = requests.post("https://api.runpod.io/graphql", headers=hdr(), timeout=60,      # GraphQL: proven on the smoke pod
                      json={"query": f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}'})
    print(f"terminate {pid} {p.get('name')} ({why}): HTTP {r.status_code} {r.text[:120]}")


def ls(a):
    mine = {r["id"] for r in records()}
    for p in pods():
        tag = "ours" if p["id"] in mine and str(p.get("name", "")).startswith("kvs-") else "NOT OURS"
        print(f"{p['id']}  {p.get('name', ''):32s} {p.get('desiredStatus', ''):9s} ${float(p.get('costPerHr') or 0):.3f}/h  {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("job")
    c.add_argument("--gpu", default="NVIDIA A40")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    c.add_argument("--disk", type=int, default=150)
    c.add_argument("--ram", type=int, default=48, help="min RAM per GPU, GB")
    c.add_argument("--cpu", help="CPU-only pod of this flavor (e.g. cpu3g, cpu3m, cpu5c); --gpu and --ram are ignored")
    c.add_argument("--vcpu", type=int, default=16, help="vCPUs of a --cpu pod")
    c.add_argument("--budget-h", type=float, required=True)
    c.add_argument("--max-price", type=float, default=0.80, help="terminate at once above this $/h")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(f=create)
    sub.add_parser("ls").set_defaults(f=ls)
    t = sub.add_parser("terminate")
    t.add_argument("pod_id")
    t.add_argument("-y", action="store_true", required=True)
    t.set_defaults(f=lambda a: terminate_one(a.pod_id, "requested"))
    a = ap.parse_args()
    a.f(a)
