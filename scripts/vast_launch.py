"""Laptop-side vast.ai launcher for the kvs-* instances (docs/monitor_design.md section 4, decisions 11-12).

    python scripts/vast_launch.py monitor --dry-run   # show the 3 cheapest offers + the exact create command, create nothing
    python scripts/vast_launch.py monitor             # create kvs-monitor, wait until it answers, print http://IP:PORT/?t=TOKEN
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
        vast("attach", "ssh", iid, pub.read_text().strip())


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
        vast("destroy", "instance", iid)
    sys.exit(1)


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
    print(vast("destroy", "instance", a.id))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("monitor")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--port", type=int, default=8787)
    m.add_argument("--image", default="python:3.11-slim")
    m.add_argument("--disk", type=int, default=10)
    m.set_defaults(f=monitor)
    sub.add_parser("ls").set_defaults(f=ls)
    d = sub.add_parser("destroy")
    d.add_argument("id", type=int)
    d.add_argument("-y", "--yes", action="store_true")
    d.set_defaults(f=destroy)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
