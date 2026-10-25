"""Laptop-side vast.ai launcher for the kvs-* instances (docs/monitor_design.md section 4, decisions 11-12).

    python scripts/vast_launch.py monitor --dry-run   # show the 3 cheapest offers + the exact create command, create nothing
    python scripts/vast_launch.py monitor             # create kvs-monitor, wait until it answers, print http://IP:PORT/?t=TOKEN
    python scripts/vast_launch.py worker JOB [--n 4] [--dry-run]   # GPU workers kvs-ltx-{k}of{n} running POD_SCRIPT JOB
    python scripts/vast_launch.py ls                  # kvs-* instances: status, GPU, $/h, IP, mapped port, minutes up
    python scripts/vast_launch.py destroy ID [-y]     # destroy one kvs-* instance (refuses other labels)

Laptop only. Account key from VAST_API_KEY or ~/.vast_key (put in the subprocess env, never printed or put on a command line).
The command line carries no secrets: the monitor instance gets MONITOR_TOKEN, MONITOR_VAST_KEY and optional MONITOR_NTFY from
vast.ai ACCOUNT-LEVEL environment variables (console > Account > Environment Variables). This script only reads MONITOR_TOKEN
from your shell to build the URL it prints.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VASTAI = shutil.which("vastai") or str(Path.home() / "AppData/Roaming/Python/Python313/Scripts/vastai.exe")
QUERY = ("rentable=true verified=true num_gpus=1 reliability>0.98 direct_port_count>=1 static_ip=true "
         "inet_down>=100 disk_space>=10")
SRC = "https://raw.githubusercontent.com/Cypher193/xakfdghdfjk/main/scripts/monitor.py"
# python:3.11-slim has no curl but has urllib; env vars from the account reach the container env, copy them for ssh shells too
ONSTART = ("pip install -q vastai; python -c \"import urllib.request as u;u.urlretrieve('{src}','/monitor.py')\" && "
           "{{ env | grep -E '^(MONITOR_TOKEN|MONITOR_VAST_KEY|MONITOR_NTFY)=' >> /etc/environment; "
           "cd / && nohup python /monitor.py --port {port} > /monitor.out 2>&1 & }}")
TEMPLATE = "d1b2fe045f890349b734850a9d312072"
POD_SCRIPT = "scripts/ltx_pod.sh"      # swap for scripts/full_pod.sh later
REPO = "https://github.com/Cypher193/xakfdghdfjk.git"
INSTANCES_JSON = Path(__file__).resolve().parent.parent / "data/logs/vast_instances.json"


def vast(*args):
    """Run the vastai CLI with --raw, return parsed JSON (None if empty). Key goes in the env only."""
    env = dict(os.environ)
    if "VAST_API_KEY" not in env and (Path.home() / ".vast_key").exists():
        env["VAST_API_KEY"] = (Path.home() / ".vast_key").read_text().strip()
    r = subprocess.run([VASTAI, *map(str, args), "--raw"], capture_output=True, text=True, encoding="utf-8-sig", env=env)
    out = r.stdout.lstrip("﻿").strip()
    try:
        return json.loads(out) if out else None
    except ValueError:
        sys.exit(f"vastai {args[0]} {args[1]}: unparseable output: {(out or r.stderr)[:300]}")


def attach_ssh(iid):
    """Team accounts cannot hold account-level SSH keys (400 "only in personal context"): attach this laptop's key per instance."""
    pub = Path.home() / ".ssh/id_ed25519.pub"
    if pub.exists():
        try:
            vast("attach", "ssh", iid, pub.read_text().strip())
        except SystemExit:                    # the CLI prints a Python repr here, not JSON; the key is attached anyway
            pass


def cheapest(n=3):
    return (vast("search", "offers", QUERY, "-o", "dph_total") or [])[:n]


def create_cmd(offer_id, a):
    return [str(x) for x in (VASTAI, "create", "instance", offer_id, "--image", a.image, "--disk", a.disk, "--label",
            "kvs-monitor", "--ssh", "--direct", "--env", f"-p {a.port}:{a.port}", "--onstart-cmd",
            ONSTART.format(src=SRC, port=a.port), "--raw")]


def instances():
    d = vast("show", "instances") or []
    return d["instances"] if isinstance(d, dict) else d


def port_of(inst, port):
    try:
        return inst["ports"][f"{port}/tcp"][0]["HostPort"]
    except (KeyError, IndexError, TypeError):
        return None


def wait_up(iid, a, token):
    """Poll show instance until running + port mapped (10 min), then GET / with the bearer until 200 (3 min). URL or None."""
    t0, inst = time.time(), {}
    while time.time() - t0 < 600:
        inst = vast("show", "instance", iid) or {}
        inst = inst.get("instances", inst) if isinstance(inst, dict) else {}
        if inst.get("actual_status") in ("exited", "offline"):
            return None
        hp = port_of(inst, a.port)
        if inst.get("actual_status") == "running" and hp:
            break
        time.sleep(10)
    else:
        return None
    base = f"http://{inst['public_ipaddr'].strip()}:{hp}/"
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            req = urllib.request.Request(base, headers={"Authorization": f"Bearer {token}"})
            if urllib.request.urlopen(req, timeout=10).status == 200:
                return f"{base}?t={token}"
        except Exception:
            pass
        time.sleep(10)
    return None


def monitor(a):
    token = os.environ.get("MONITOR_TOKEN")
    if not token:
        sys.exit("MONITOR_TOKEN is not set in this shell (needed to print the URL; the instance gets its own copy from the "
                 "vast.ai account-level environment variables). Set it, e.g. $env:MONITOR_TOKEN='...'")
    offers = cheapest()
    if not offers:
        sys.exit("no offers match the query")
    for o in offers:
        print(f"offer {o['id']}  {o['gpu_name']}  ${o['dph_total']:.3f}/h  {o.get('geolocation')}  "
              f"{o['inet_down']:.0f} Mb/s down  {o['disk_space']:.0f} GB disk")
    if a.dry_run:
        print("\n" + " ".join(shlex.quote(c) for c in create_cmd(offers[0]["id"], a)))
        return
    for o in offers:
        r = vast(*create_cmd(o["id"], a)[1:-1])
        iid = (r or {}).get("new_contract")
        if not iid:
            print(f"offer {o['id']}: create failed: {r}")
            continue
        print(f"created #{iid} on offer {o['id']}, waiting ...")
        attach_ssh(iid)
        url = wait_up(iid, a, token)
        if url:
            print(f"\nmonitor #{iid}: {url}")
            return
        print(f"#{iid} did not come up, destroying")
        vast("destroy", "instance", iid, "-y")
    sys.exit(1)


def worker_label(a, k):
    m = re.fullmatch(r"s(\d+)of(\d+)", a.job)      # phase 2 shard jobs: one monitor block "ltx2"
    return f"kvs-ltx2-{m[1]}of{m[2]}" if m else f"kvs-ltx{a.job}-{k}of{a.n}"


def worker_offers(a):
    q = (f"rentable=true verified=true num_gpus=1 gpu_name={a.gpu.replace(' ', '_')} gpu_ram>={a.vram} cpu_ram>={a.ram} "
         f"disk_space>={a.disk} reliability>0.98 inet_down>=400 duration>1 cuda_vers>=12.8 direct_port_count>=1 "
         + (f"geolocation={a.geo}" if a.geo else "geolocation!=CN"))   # CN hosts: HF downloads at ~150 s per file
    seen, out = set(a.not_machine), []
    for o in vast("search", "offers", q, "-o", "dph_total") or []:
        if o["machine_id"] not in seen:          # one offer per host so workers never share a machine
            seen.add(o["machine_id"])
            out.append(o)
    return out


def worker_create_cmd(offer_id, label, a, url):
    onstart = (f"cd /workspace && git clone {REPO} cvpr_vail && cd cvpr_vail && setsid nohup bash {POD_SCRIPT} "
               f"{a.job} {a.budget_h} > /workspace/ltx.out 2>&1 < /dev/null &")
    cmd = [VASTAI, "create", "instance", offer_id, "--template_hash", TEMPLATE, "--disk", a.disk, "--label", label]
    if url:
        cmd += ["--env", f"-e MONITOR_URL={url}"]
    return [str(x) for x in cmd + ["--onstart-cmd", onstart, "--raw"]]


def log_instance(rec):
    try:
        rows = json.loads(INSTANCES_JSON.read_text())
    except (OSError, ValueError):
        rows = []
    INSTANCES_JSON.parent.mkdir(parents=True, exist_ok=True)
    INSTANCES_JSON.write_text(json.dumps(rows + [rec], indent=1))


def worker(a):
    url = a.monitor_url or os.environ.get("MONITOR_URL")
    if not url:
        print("WARNING: no --monitor-url / MONITOR_URL, workers will not report to the monitor")
    offers = worker_offers(a)
    print(f"{len(offers)} distinct machines match; top:")
    for o in offers[:max(5, a.n)]:
        print(f"offer {o['id']}  m{o['machine_id']}  {o['gpu_name']}  ${o['dph_total']:.3f}/h  {o.get('geolocation')}  "
              f"{o['inet_down']:.0f} Mb/s down  {o['disk_space']:.0f} GB disk  rel {o.get('reliability2', 0):.3f}")
    if len(offers) < a.n:
        print(f"WARNING: only {len(offers)} distinct machines for n={a.n}")
    if a.dry_run:
        for k, o in enumerate(offers[:a.n], 1):
            print("\n" + " ".join(shlex.quote(c) for c in worker_create_cmd(o["id"], worker_label(a, k), a, url)))
        return
    pool = iter(offers)
    for k in range(1, a.n + 1):
        label, ok = worker_label(a, k), False       # block "ltxgen1": one block per job on the monitor
        for o in [x for _, x in zip(range(3), pool)]:
            r = vast(*worker_create_cmd(o["id"], label, a, url)[1:-1])
            iid = (r or {}).get("new_contract")
            if not iid:
                print(f"offer {o['id']}: create failed: {r}")
                continue
            print(f"{label}: created #{iid} on offer {o['id']}, waiting ...")
            attach_ssh(iid)
            t0, inst = time.time(), {}
            while time.time() - t0 < 1500:                 # the 20 GB pytorch image took > 10 min to pull on 3090 hosts
                inst = vast("show", "instance", iid) or {}
                inst = inst.get("instances", inst) if isinstance(inst, dict) else {}
                if inst.get("actual_status") in ("running", "exited", "offline"):
                    break
                time.sleep(10)
            if inst.get("actual_status") != "running":
                print(f"#{iid} did not come up ({inst.get('actual_status')}), destroying")
                vast("destroy", "instance", iid, "-y")
                continue
            hp = port_of(inst, 22)
            print(f"{label}: #{iid} {inst.get('gpu_name')} {inst.get('geolocation')} ${inst.get('dph_total') or 0:.3f}/h\n"
                  f"  ssh -p {inst.get('ssh_port')} root@{inst.get('ssh_host')}\n"
                  f"  ssh -p {hp} root@{(inst.get('public_ipaddr') or '').strip()}")
            log_instance({"iid": iid, "label": label, "job": a.job, "offer": o["id"], "dph": o["dph_total"],
                          "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
            ok = True
            break
        if not ok:
            print(f"{label}: no instance after 3 attempts")


def ls(_):
    allx = instances()
    print(f"{len(allx)} instances on the account")
    print(f"{'id':>9} {'label':16} {'status':10} {'gpu':16} {'$/h':>6} {'geo':16} {'ip':16} {'port':>6} {'min':>5}")
    for i in allx:
        if not (i.get("label") or "").startswith("kvs-"):
            continue
        up = (time.time() - i["start_date"]) / 60 if i.get("start_date") else 0
        print(f"{i['id']:>9} {i['label']:16} {str(i.get('actual_status')):10} {str(i.get('gpu_name')):16} "
              f"{i.get('dph_total') or 0:6.3f} {str(i.get('geolocation'))[:16]:16} {(i.get('public_ipaddr') or '').strip():16} "
              f"{port_of(i, 8787) or '-':>6} {up:5.0f}")


def destroy(a):
    inst = next((i for i in instances() if i["id"] == a.id), None)
    if not inst:
        sys.exit(f"no instance {a.id}")
    label = inst.get("label") or ""
    if not label.startswith("kvs-"):
        sys.exit(f"refusing: #{a.id} has label {label!r}, not kvs-*")
    if not a.yes and input(f"destroy #{a.id} ({label}, {inst.get('gpu_name')}, {inst.get('actual_status')})? [y/N] ").lower() != "y":
        sys.exit("aborted")
    print(vast("destroy", "instance", a.id, "-y"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--port", type=int, default=8787)
    m.add_argument("--image", default="python:3.11-slim")
    m.add_argument("--disk", type=int, default=10)
    m.set_defaults(f=monitor)
    w = sub.add_parser("worker")
    w.add_argument("job")
    w.add_argument("--n", type=int, default=1)
    w.add_argument("--gpu", default="RTX 3090")
    w.add_argument("--vram", type=int, default=24)
    w.add_argument("--ram", type=int, default=32)
    w.add_argument("--disk", type=int, default=120)
    w.add_argument("--geo", help="2-letter country, e.g. US (default: anywhere but CN)")
    w.add_argument("--not-machine", type=int, nargs="*", default=[], help="machine_ids to skip (e.g. a host that never finished pulling the image)")
    w.add_argument("--budget-h", type=float, default=6)
    w.add_argument("--monitor-url")
    w.add_argument("--dry-run", action="store_true")
    w.set_defaults(f=worker)
    sub.add_parser("ls").set_defaults(f=ls)
    d = sub.add_parser("destroy")
    d.add_argument("id", type=int)
    d.add_argument("-y", "--yes", action="store_true")
    d.set_defaults(f=destroy)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
