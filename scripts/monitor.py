#!/usr/bin/env python3
"""Run monitor: heartbeat collector, status page and reaper for kvs-* vast.ai instances (docs/monitor_design.md).

    MONITOR_TOKEN=... [MONITOR_VAST_KEY=...] [MONITOR_NTFY=topic] python scripts/monitor.py --port 8787
    python scripts/monitor.py --dry-run     # reaper logs what it would destroy, never destroys
    python scripts/monitor.py --selftest

Pods POST beats to /hb (Bearer token). The page is open with ?t=TOKEN once (sets a cookie).
"""
import argparse
import hmac
import http.client
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VASTAI_FALLBACK = Path.home() / "AppData/Roaming/Python/Python313/Scripts/vastai.exe"
SCHEMA = """
CREATE TABLE IF NOT EXISTS beats(ts REAL, inst INTEGER, json TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, inst INTEGER, label TEXT, level TEXT, line TEXT);
CREATE INDEX IF NOT EXISTS beats_i ON beats(inst, ts);
CREATE INDEX IF NOT EXISTS events_t ON events(ts);
CREATE INDEX IF NOT EXISTS events_i ON events(inst, ts);
"""
LEVELS = ["info", "warn", "error"]


class App:
    def __init__(self, db, token, vast_key="", ntfy="", dry_run=False, stale_min=10):
        self.db = sqlite3.connect(db, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.token, self.vast_key, self.ntfy = token, vast_key, ntfy
        self.dry_run, self.stale_s = dry_run, stale_min * 60
        self.vast = {}  # inst id -> vast row, refreshed by the reaper
        self.reaper = {"enabled": bool(vast_key), "dry_run": dry_run, "last_ok": None, "last_err": None}
        self.dead_since, self.stale_sent, self.dry_seen = {}, {}, set()
        self.backoff, self.last_summary, self.started = False, None, time.time()

    def q(self, sql, args=()):
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
            self.db.commit()
        return rows

    def event(self, inst, label, line, level=None, ts=None):
        self.q("INSERT INTO events(ts, inst, label, level, line) VALUES(?,?,?,?,?)",
               (ts or time.time(), inst, label, level or level_of(line), line))


def level_of(line):
    if line.startswith("KILL ") or re.search(r"\bEND \S+: rc=(?!0,)", line):
        return "error"
    return "warn" if re.search(r"\bSKIP \S+:", line) else "info"


def parse_label(label):
    label = label or ""
    if label == "kvs-monitor":
        return "monitor", ""
    if m := re.fullmatch(r"kvs-screen-(A|B|C|all)", label):
        return "screen", m[1]
    if m := re.fullmatch(r"kvs-([a-z0-9_]+)-(\d+)of(\d+)", label):
        return m[1], f"{m[2]}/{m[3]}"
    return "other", label


def latest_beats(app):
    return {i: (json.loads(j), ts) for i, j, ts in app.q("SELECT inst, json, max(ts) FROM beats GROUP BY inst")}


def eta_h(done, total, used_h):
    if not total or done is None or done < 3 or not used_h:
        return None
    return max(total - done, 0) / (done / used_h)


def shard_row(inst, b, ts, v, now):
    b = b or {}
    label = (v or {}).get("label") or ("" if re.fullmatch(r"C\.\d+", b.get("label") or "") else b.get("label") or "")
    block, shard = parse_label(label)
    age = now - ts if ts else None
    up_min = (now - v["start_date"]) / 60 if v and v.get("start_date") else None
    if v and v.get("actual_status") in ("exited", "offline"):
        dot = "grey"
    elif age is None:
        dot = "red" if up_min and up_min > 60 else "yellow" if up_min and up_min > 30 else "loading"
    else:
        dot = "green" if age <= 60 else "yellow" if age <= 120 else "red"
    dph = (v or {}).get("dph_total")
    clips_total = b.get("clips_total") or None
    host = b.get("host")
    if v and v.get("machine_id"):
        host = f"machine {v['machine_id']}, {v.get('geolocation') or ''}"
    return {"inst": inst, "label": label, "block": block, "shard": shard, "dot": dot, "age_s": age, "up_min": up_min,
            "status": b.get("status") or "loading", "actual_status": (v or {}).get("actual_status"),
            "gpu": (v or {}).get("gpu_name") or b.get("gpu"), "host": host, "step": b.get("step"),
            "clips_done": b.get("clips_done"), "clips_total": clips_total, "used_h": b.get("used_h"),
            "budget_h": b.get("budget_h"), "gpu_util": b.get("gpu_util"), "vram_gb": b.get("vram_gb"),
            "vram_total_gb": b.get("vram_total_gb"), "dph": dph,
            "spent": (now - v["start_date"]) / 3600 * dph if v and v.get("start_date") and dph is not None else None,
            "eta_h": eta_h(b.get("clips_done"), clips_total, b.get("used_h"))}


def stale_ref(r, ts, started, stale_s):
    """Reference time for the no-beat rule: last beat, setup grace, and the monitor's own warm-up after a restart."""
    return max(ts or 0, (r.get("start_date") or 0) + 1800, started + stale_s)


def state(app):
    now = time.time()
    st = classify(dict(app.vast), latest_beats(app), now, bool(app.reaper["last_ok"]))
    kills = app.q("SELECT ts, inst, label, line FROM events WHERE line LIKE 'KILL %' AND ts > ? ORDER BY ts DESC",
                  (now - 86400,))
    return dict(st, now=now, reaper=app.reaper, kills=[dict(zip(("ts", "inst", "label", "line"), k)) for k in kills])


def classify(vast, beats, now, reaper_ok=False):
    """Pure: vast rows {inst: row} + latest beats {inst: (beat, ts)} -> totals, monitor, blocks.

    reaper_ok: the reaper has seen vast; a beat-only instance is then "gone" (finished or destroyed), not dead."""
    blocks, mon, gpus = {}, None, {}
    tot = {"running": 0, "loading": 0, "dead": 0, "dph_now": 0.0, "spent": 0.0, "clips_done": 0, "clips_total": 0}
    for inst in sorted(set(beats) | set(vast)):
        b, ts = beats.get(inst, (None, None))
        v = vast.get(inst)
        s = shard_row(inst, b, ts, v, now)
        if s["label"] == "kvs-monitor":
            mon = mon or {"inst": inst, "ip": (v or {}).get("public_ipaddr"), "dph": s["dph"], "up_min": s["up_min"]}
            continue
        blk = blocks.setdefault(s["block"], {"block": s["block"], "clips_done": 0, "clips_total": 0, "spent": 0.0,
                                             "eta_h": None, "shards": []})
        blk["shards"].append(s)
        if v and not s["label"].startswith("kvs-"):  # foreign instance on the account: listed, never counted
            s.update(dot="grey", status="not ours")
            continue
        gone = reaper_ok and v is None and b is not None
        if gone:
            s.update(dot="grey", status="gone", eta_h=None)
        blk["spent"] += s["spent"] or 0
        if s["clips_total"]:
            blk["clips_done"] += s["clips_done"] or 0
            blk["clips_total"] += s["clips_total"]
        if s["eta_h"] is not None:
            blk["eta_h"] = max(blk["eta_h"] or 0, s["eta_h"])
        if gone:
            continue
        tot["running" if s["dot"] in ("green", "yellow") else "loading" if s["dot"] == "loading" else "dead"] += 1
        if s["actual_status"] and s["actual_status"] not in ("exited", "offline"):  # vast bills loading too
            tot["dph_now"] += s["dph"] or 0
        if s["gpu"]:
            gpus[s["gpu"]] = gpus.get(s["gpu"], 0) + 1
    bl = sorted(blocks.values(), key=lambda x: x["block"])
    for k in ("spent", "clips_done", "clips_total"):
        tot[k] = sum(x[k] for x in bl)
    etas = [x["eta_h"] for x in bl if x["eta_h"] is not None]
    return {"totals": dict(tot, gpus=gpus, eta_h=max(etas) if etas else None), "monitor": mon or {}, "blocks": bl}


def inst_detail(app, inst):
    rows = app.q("SELECT ts, json FROM beats WHERE inst=? AND ts > ? ORDER BY ts", (inst, time.time() - 7200))
    bs = [(ts, json.loads(j)) for ts, j in rows]
    col = lambda k: [b.get(k) for _, b in bs]
    net = [None if b.get("net_rx_mbps") is None else b["net_rx_mbps"] + (b.get("net_tx_mbps") or 0) for _, b in bs]
    latest = dict(bs[-1][1]) if bs else {}
    v = app.vast.get(inst)
    ev = app.q("SELECT ts, level, line FROM events WHERE inst=? ORDER BY ts DESC, id DESC LIMIT 200", (inst,))
    return {"inst": inst, "label": (v or {}).get("label") or latest.get("label"), "latest": latest, "vast": v,
            "tail": latest.pop("tail", ""),
            "series": {"ts": [ts for ts, _ in bs], "gpu_util": col("gpu_util"), "vram_gb": col("vram_gb"),
                       "cpu_pct": col("cpu_pct"), "ram_gb": col("ram_gb"), "net_mbps": net},
            "events": [dict(zip(("ts", "level", "line"), e)) for e in ev]}


def events_list(app, level="", block="", limit=200):
    mn = LEVELS.index(level) if level in LEVELS else 0  # level= means "at least"
    out = []
    for ts, inst, label, lv, line in app.q("SELECT ts, inst, label, level, line FROM events ORDER BY ts DESC, id DESC"):
        if LEVELS.index(lv) >= mn and (not block or parse_label(label)[0] == block):
            out.append({"ts": ts, "inst": inst, "label": label, "block": parse_label(label)[0], "level": lv, "line": line})
            if len(out) >= limit:
                break
    return out


def label_of(app, inst, b):
    return (app.vast.get(inst) or {}).get("label") or ("" if re.fullmatch(r"C\.\d+", b.get("label") or "") else b.get("label") or "")


def ingest(app, b):
    inst, now = int(b["inst"]), time.time()
    prev = app.q("SELECT json FROM beats WHERE inst=? ORDER BY ts DESC LIMIT 1", (inst,))
    app.q("INSERT INTO beats VALUES(?,?,?)", (now, inst, json.dumps(b)))
    label = label_of(app, inst, b)
    if b.get("status") == "triggered" and (not prev or json.loads(prev[0][0]).get("status") != "triggered"):
        app.event(inst, label, f"guard triggered on {label}", "error", now)
    for line in b.get("events") or []:
        app.event(inst, label, str(line), None, now)


# --- reaper -----------------------------------------------------------------------------------------------------

def vastai(app, *args):
    exe = shutil.which("vastai") or str(VASTAI_FALLBACK)
    env = dict(os.environ, VAST_API_KEY=app.vast_key)
    try:
        p = subprocess.run([exe, *args], capture_output=True, env=env, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, "", type(e).__name__
    dec = lambda x: x.decode("utf-8-sig", "replace").lstrip("﻿")
    return p.returncode, dec(p.stdout), dec(p.stderr)


def notify(app, title, body, prio):
    if not app.ntfy:
        return
    req = urllib.request.Request(f"https://ntfy.sh/{app.ntfy}", body.encode(),
                                 {"Title": title, "Priority": str(prio)})
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:
        pass


def kill(app, inst, label, reason, last_ts, b):
    """Destroy (or, in dry-run, log) one instance and leave a trace everywhere; returns True if destroyed."""
    pre = "[dry-run] " if app.dry_run else ""
    m = f"{(time.time() - last_ts) / 60:.0f}" if last_ts else "?"
    spent = (b or {}).get("spent")
    body = (f"{label} #{inst} destroyed: {reason}; last beat {m} min ago, {(b or {}).get('used_h')}/"
            f"{(b or {}).get('budget_h')} h, ${spent if spent is not None else '?'}")
    if app.dry_run:
        if (inst, reason.split()[0]) in app.dry_seen:
            return False
        app.dry_seen.add((inst, reason.split()[0]))
    else:
        rc, out, err = vastai(app, "destroy", "instance", str(inst), "-y")     # without -y the CLI prompts [y/N]
        if rc:
            app.event(inst, label, f"reaper: destroy #{inst} failed rc={rc}", "warn")
            print(f"destroy {inst} failed rc={rc}", flush=True)
            return False
        app.vast.pop(inst, None)
    app.event(inst, label, f"KILL {pre}{label} #{inst}: {reason}")
    notify(app, f"kvs kill {label}", pre + body, 4)
    print(f"{pre}kill {label} #{inst}: {reason}", flush=True)
    return not app.dry_run


def fail(app, msg):
    if app.reaper["last_err"] != msg:
        app.event(0, "reaper", f"reaper: {msg}", "warn")
        print(f"reaper: {msg}", flush=True)
    app.reaper["last_err"] = msg


def reap_once(app):
    if app.backoff:
        app.backoff = False
        return
    rc, out, err = vastai(app, "show", "instances", "--raw")
    if rc or "too frequent" in (out + err).lower():
        app.backoff = True
        return fail(app, f"vastai rc={rc}: {(err or out).strip()[:120]}")
    try:
        rows = json.loads(out.lstrip("﻿"))
    except ValueError:
        return fail(app, "vastai output is not JSON")
    app.vast = {r["id"]: r for r in rows}
    app.reaper.update(last_ok=time.time(), last_err=None)
    beats, now, mine = latest_beats(app), time.time(), 0
    for r in rows:
        label, inst = r.get("label") or "", r["id"]
        if not label.startswith("kvs-") or label == "kvs-monitor":
            continue
        mine += 1
        b, ts = beats.get(inst, ({}, None))
        if ts != app.stale_sent.get(inst, ts):  # a newer beat ends the dead episode
            app.stale_sent.pop(inst)
        reason = None
        if r.get("actual_status") in ("exited", "offline"):
            since = app.dead_since.setdefault(inst, now)
            if now - since > app.stale_s:
                reason = f"{r['actual_status']} for {(now - since) / 60:.0f} min"
        else:
            app.dead_since.pop(inst, None)
        if r.get("actual_status") == "running":
            ref = stale_ref(r, ts, app.started, app.stale_s)
            if now - ref > app.stale_s:
                reason = f"no beat for {(now - ref) / 60:.0f} min"
            elif now - ref > 120 and inst not in app.stale_sent:
                app.stale_sent[inst] = ts
                m = f"{(now - ref) / 60:.0f}"
                app.event(inst, label, f"stale: no beat {m} min, vast {r['actual_status']}", "warn")
                notify(app, f"kvs stale {label}", f"{label} #{inst}: no beat {m} min, vast {r['actual_status']}; "
                       f"reaped at {app.stale_s // 60} min", 3)
        if b.get("used_h") is not None and b.get("budget_h") and b["used_h"] > b["budget_h"] + 1:
            reason = f"used {b['used_h']} h > budget {b['budget_h']} h + 1"
        if reason:
            kill(app, inst, label, reason, ts, dict(b, spent=shard_row(inst, b, ts, r, now)["spent"]))
    summary = f"{mine} kvs instances"
    if summary != app.last_summary:
        print(summary, flush=True)
        app.last_summary = summary


def reaper_loop(app):
    if not app.vast_key:
        print("reaper disabled (MONITOR_VAST_KEY unset)", flush=True)
    n = 0
    while True:
        try:
            if app.vast_key:
                reap_once(app)
            if n % 60 == 0:
                app.q("DELETE FROM beats WHERE ts < ?", (time.time() - 86400,))
        except Exception as e:
            fail(app, f"cycle error {type(e).__name__}")
        n += 1
        time.sleep(60)


# --- http -------------------------------------------------------------------------------------------------------

def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # the request line can carry ?t=TOKEN
            pass

        def send(self, code, body=b"", ctype="application/json", headers=()):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def authed(self, tok=None):
            got = tok if tok is not None else (self.headers.get("Authorization") or "")[7:] \
                if (self.headers.get("Authorization") or "").startswith("Bearer ") else ""
            cookie = re.search(r"(?:^|;\s*)kvs_t=([^;]*)", self.headers.get("Cookie") or "")
            ok = lambda t: hmac.compare_digest(t.encode(), app.token.encode())
            return ok(got) or bool(cookie and ok(urllib.parse.unquote(cookie[1])))

        def route(self, method):
            u = urllib.parse.urlsplit(self.path)
            qs = urllib.parse.parse_qs(u.query)
            if method == "GET" and "t" in qs and self.authed(qs["t"][0]):
                rest = urllib.parse.urlencode([(k, v) for k, vs in qs.items() if k != "t" for v in vs])
                return self.send(302, b"", headers=[
                    ("Location", u.path + ("?" + rest if rest else "")),
                    ("Set-Cookie", f"kvs_t={urllib.parse.quote(app.token)}; HttpOnly; SameSite=Strict; Path=/; Max-Age=604800")])
            if not self.authed():
                return self.send(401)
            p = u.path
            if method == "POST" and p == "/hb":
                n = int(self.headers.get("Content-Length") or 0)
                if n > 1 << 20:
                    return self.send(413)
                try:
                    ingest(app, json.loads(self.rfile.read(n)))
                except (ValueError, KeyError, TypeError):
                    return self.send(400)
                return self.send(200, {"ok": True})
            if method == "POST" and (m := re.fullmatch(r"/api/kill/(\d+)", p)):
                return self.manual_kill(int(m[1]))
            if method != "GET":
                return self.send(404)
            if p == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "/api/state":
                return self.send(200, state(app))
            if m := re.fullmatch(r"/api/inst/(\d+)", p):
                return self.send(200, inst_detail(app, int(m[1])))
            if p == "/api/events":
                g = lambda k: qs.get(k, [""])[0]
                return self.send(200, events_list(app, g("level"), g("block"), int(g("limit") or 200)))
            self.send(404)

        def manual_kill(self, inst):
            v = app.vast.get(inst)
            if not app.vast_key:
                return self.send(503, {"error": "reaper disabled, no vast key"})
            if not v or not (v.get("label") or "").startswith("kvs-") or v["label"] == "kvs-monitor":
                return self.send(403, {"error": "not a kvs worker"})
            beat = latest_beats(app).get(inst, ({}, None))
            done = kill(app, inst, v["label"], "manual kill from page", beat[1], beat[0])
            self.send(200, {"destroyed": done, "dry_run": app.dry_run})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

    return H


# --- page -------------------------------------------------------------------------------------------------------

PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>kvs monitor</title><style>
:root{--bg:#f3f4f6;--sf:#fff;--bd:#d4d8de;--tx:#16181d;--mu:#596070;--gn:#17803a;--ye:#8a5d00;--rd:#c42b1c;--gy:#636b78;--ac:#2b5fd9;--dg:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#0e1013;--sf:#181b20;--bd:#2d323b;--tx:#e7e9ec;--mu:#9ba2ae;--gn:#4fc46d;--ye:#e2b33c;--rd:#ff6e64;--gy:#8f96a3;--ac:#7ea6ff;--dg:#c4291c}}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.35 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums}
body.lock{overflow:hidden}
button{font:inherit;color:inherit;cursor:pointer}
header{position:sticky;top:0;z-index:5;height:48px;background:var(--sf);border-bottom:1px solid var(--bd)}
.in{max-width:760px;margin:0 auto;padding:0 16px}
header .in{height:100%;display:flex;align-items:center;gap:8px}
header b{font-size:14px;font-weight:600}
#upd{margin-left:auto}
.chip{border:1px solid currentColor;border-radius:12px;padding:0 8px;font-size:12px;white-space:nowrap}
main{padding-top:16px;padding-bottom:24px}
#pg{display:flex;flex-direction:column;gap:24px}
.s12{font-size:12px}
.mu{color:var(--mu)}
.c-gn{color:var(--gn)}.c-ye{color:var(--ye)}.c-rd{color:var(--rd)}.c-gy{color:var(--gy)}
#ban{display:flex;flex-direction:column;gap:8px;margin-bottom:16px}
#ban:empty{display:none}
.bn{display:block;width:100%;text-align:left;background:var(--dg);color:#fff;border:0;border-radius:8px;padding:8px 12px;font-size:14px}
div.bn{cursor:default}
.yb{background:var(--sf);border:1px solid var(--bd);border-left:4px solid var(--ye);border-radius:8px;padding:8px 12px}
#tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.tile{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:8px 12px}
.tile .val{font-size:20px;font-weight:600;line-height:1.3}
.tile .s12:last-child{min-height:16px}
#blocks{display:grid;gap:12px}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:12px}
.bh{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 8px}
.bh b{font-size:14px;font-weight:600}
.bh .tot{margin-left:auto;font-weight:600}
.l2{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;margin-top:4px;font-size:12px;color:var(--mu)}
.bar{flex:1;min-width:48px;height:6px;background:var(--bd);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ac)}
.row{min-height:48px;padding:8px 0 8px 8px;border-top:1px solid var(--bd);border-left:3px solid transparent;cursor:pointer;margin-top:8px}
.row[aria-current]{border-left-color:var(--ac)}
.row.mu .l1 .stp{opacity:.8}
.l1{display:flex;align-items:center;gap:8px}
.l1 b{font-weight:600}
.w{min-width:64px}
.stp{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mu)}
.dot{flex:none;width:10px;height:10px;border-radius:50%;background:var(--gy)}
.dot.green{background:var(--gn)}.dot.yellow{background:var(--ye)}.dot.red{background:var(--rd)}
.dot.loading{background:none;border:2px solid var(--gy)}
.det{position:relative}
.top{display:flex;align-items:center;gap:8px;min-height:40px;background:var(--sf);margin:0 0 8px}
.top b{font-weight:600}
.xb{margin-left:auto;width:40px;height:40px;border:0;background:none;font-size:24px;line-height:1}
.det h3{margin:16px 0 4px;font-size:12px;font-weight:400;color:var(--mu)}
.sps{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.sps .net{grid-column:1/3}
.sph{display:flex;justify-content:space-between;align-items:baseline}
.sph b{font-size:14px;font-weight:600}
.sps svg{display:block;width:100%;height:32px;color:var(--ac)}
pre{margin:0;max-height:240px;overflow:auto;font:11px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:8px;white-space:pre-wrap;overflow-wrap:anywhere}
.kb{display:block;width:100%;height:44px;margin-top:16px;border:0;border-radius:8px;background:var(--dg);color:#fff;font-weight:600}
.kb:disabled{background:var(--gy);cursor:not-allowed}
#bd{position:fixed;inset:0;z-index:15;background:rgba(0,0,0,.45)}
#evs h2{margin:0 0 8px;font-size:14px;font-weight:600}
.chs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.ch{height:40px;padding:0 12px;border-radius:20px;border:1px solid var(--bd);background:var(--bg);color:var(--mu)}
.ch[aria-pressed=true]{background:var(--sf);color:var(--tx);border:2px solid var(--ac)}
.er{display:flex;gap:8px;padding:4px 0;font-size:12px}
.lb{flex:none;width:3px;align-self:stretch;background:transparent}
.er.error .lb{background:var(--rd)}.er.warn .lb{background:var(--ye)}
.er time{flex:none;color:var(--mu)}
.bdg{flex:none;max-width:40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;border:1px solid var(--bd);border-radius:4px;padding:0 4px}
.et{min-width:0;overflow-wrap:anywhere}
#ft{font-size:12px;color:var(--mu)}
@media(min-width:720px){#tiles{grid-template-columns:repeat(6,1fr)}#bd{display:none}}
@media(max-width:719px){.det{position:fixed;left:0;right:0;bottom:0;height:80vh;overflow:auto;z-index:20;border-radius:12px 12px 0 0;border-bottom:0;padding:16px}
.top{position:sticky;top:-16px;z-index:2;margin:-16px -16px 8px;padding:16px 16px 0}}
</style></head><body>
<header><div class=in><b>kvs monitor</b><span id=upd class="s12 mu"></span><span id=chip class=chip></span></div></header>
<main class=in><div id=ban></div><div id=pg><div id=tiles></div><div id=blocks></div>
<section id=evs><h2>Events</h2><div id=chips></div><div id=ev></div></section><div id=ft></div></div></main>
<div id=bd hidden></div><div id=det class="card det" hidden></div>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=(v,d=1)=>v==null?'—':(+v).toFixed(d),mq=matchMedia('(min-width:720px)');
const eta=h=>h==null?'—':h>=48?f(h/24)+' d':f(h)+' h',ago=s=>s<60?Math.round(s)+' s':Math.round(s/60)+' min';
const upt=m=>m==null?'—':m<60?Math.round(m)+' min':f(m/60,m<600?1:0)+' h',hm=ts=>new Date(ts*1e3).toTimeString().slice(0,5);
let S,D,open=null,ok=0,err=null,dErr=null,kmsg='',lv='',bk='',EV=[];
async function api(p,o){const r=await fetch(p,o);if(!r.ok)throw{s:r.status,b:await r.text().catch(()=>'')};return r.json()}
const msg=e=>e&&e.s==401?'Not signed in: open the link with ?t=TOKEN once.':`Monitor unreachable (${e&&e.s?'HTTP '+e.s:'network'}). Retrying every 15 s.`;
const flat=()=>S?S.blocks.flatMap(b=>b.shards):[],ld=s=>s.age_s==null&&s.dot!='grey';
function word(s){
  if(s.status=='not ours')return['not ours','gy'];
  if(s.step=='done')return['done','gn'];
  if(s.dot=='grey')return[s.status=='gone'?'gone':s.actual_status||'gone','gy'];
  if(ld(s))return['loading',{red:'rd',yellow:'ye'}[s.dot]||'gy'];
  return{green:['live','gn'],yellow:['late','ye'],red:['stale','rd']}[s.dot]}
function row(s){
  const [w,c]=word(s),l=ld(s),no=s.status=='not ours',T=s.clips_total,full=s.step=='done';
  let stp=no?'':l?s.actual_status:s.step;
  if(!l&&!no&&s.status!='running'&&s.status!='gone')stp=s.status+(s.step?' '+s.step:'');
  const l2=l?`${f(s.up_min,0)} min since start · $${f(s.dph,2)}/h`:no?`${esc(s.gpu)} · $${f(s.dph,2)}/h`
    :(T?`<span class=bar><i style="width:${full?100:Math.min(100,100*(s.clips_done||0)/T)}%${full?';background:var(--gn)':''}"></i></span>`:'')
    +`<span>${s.clips_done??'?'}/${T||'?'}</span><span>${f(s.gpu_util,0)}%</span><span>${f(s.vram_gb,0)}/${f(s.vram_total_gb,0)}G</span><span>${f(s.used_h)}/${f(s.budget_h,0)}h</span><span>$${f(s.spent,2)}</span>`;
  return `<div class="row${no?' mu':''}" role=button tabindex=0 data-i=${s.inst}${s.inst===open?' aria-current=true':''}><div class=l1><i class="dot ${l&&s.dot=='loading'?'loading':s.dot}"></i><span class="w c-${c}">${esc(w)}</span><b>${esc(s.shard)}</b><span class=stp>${esc(stp)}</span><span class="s12 mu">${l||no||s.dot=='grey'?'':eta(s.eta_h)}</span></div><div class=l2>${l2}</div></div>`}
function card(b){
  const n={},T=b.clips_total;
  b.shards.forEach(s=>{if(s.status=='not ours'||s.dot=='grey')return;const k=ld(s)?'loading':{green:'live',yellow:'late',red:'stale'}[s.dot];n[k]=(n[k]||0)+1});
  const cn=['live','late','stale','loading'].filter(k=>n[k]).map(k=>`<span class="${k=='stale'?'c-rd':''}">${k} ${n[k]}</span>`).join(' · ');
  const sh=b.shards.slice().sort((a,b)=>(parseInt(a.shard)-parseInt(b.shard))||a.shard.localeCompare(b.shard));
  return `<section class=card><div class=bh><b>${esc(b.block)}</b><span class="s12 mu">${cn}</span><span class=tot>$${f(b.spent,2)}</span></div>
  <div class=l2><span class=bar><i style="width:${T?100*b.clips_done/T:0}%"></i></span><span>${b.clips_done}/${T}</span><span>eta ${eta(b.eta_h)}</span></div>${sh.map(row).join('')}</section>`}
function tiles(){
  const t=S.totals,late=flat().filter(s=>s.dot=='yellow'&&!ld(s)).length;
  const T=(l,v,sub='',c='')=>`<div class=tile><div class="s12 mu ${c}">${l}</div><div class="val ${c}">${v}</div><div class=s12>${sub}</div></div>`;
  return T('Running',t.running,(t.loading?`<span class=mu>+${t.loading} loading</span> `:'')+(late?`<span class=c-ye>${late} late</span>`:''))
    +T('Dead',t.dead,'',t.dead?'c-rd':'')+T('$/h now','$'+f(t.dph_now,2))+T('Spent','$'+f(t.spent,t.spent>=100?0:2))
    +T('Clips',t.clips_total?Math.round(100*t.clips_done/t.clips_total)+' %':'—',t.clips_total?t.clips_done+'/'+t.clips_total:'')+T('ETA',eta(t.eta_h))}
function ban(){
  let h=err?`<div class=bn>${esc(msg(err))}</div>`:'';
  if(!S)return h;
  const red=flat().filter(s=>s.dot=='red'),b=(i,t)=>`<button class=bn data-i=${i}>${t}</button>`;
  red.slice(0,3).forEach(s=>h+=b(s.inst,`#${s.inst} ${esc(s.block)} ${esc(s.shard)} · ${ld(s)?'loading '+f(s.up_min,0):'no beat '+Math.round(s.age_s/60)} min`));
  if(red.length>3)h+=`<div class=bn>+${red.length-3} more</div>`;
  S.kills.filter(k=>k.ts>S.now-3600).forEach(k=>h+=b(k.inst,`${esc(k.line)} · ${ago(S.now-k.ts)} ago`));
  return h+(S.reaper.last_err?`<div class=yb>reaper: ${esc(S.reaper.last_err)}</div>`:'')}
function chip(){
  const r=S.reaper;
  return!r.enabled?['off','gy']:r.last_err?['error','rd']:!r.last_ok||S.now-r.last_ok>180?['lagging','ye']:r.dry_run?['dry-run','ye']:['on','gn']}
function foot(){
  const m=S.monitor,g=Object.entries(S.totals.gpus).map(([k,n])=>n+'×'+k).join(', ');
  return(m.inst?`monitor #${m.inst} · ${esc(m.ip)} · $${f(m.dph,2)}/h · up ${upt(m.up_min)}`:'monitor not in vast list')+(g?' · GPUs '+esc(g):'')}
function snap(){const d=$('det'),p=d.querySelector('pre');return{st:d.scrollTop,ps:p&&p.scrollTop,bot:!p||p.scrollHeight-p.scrollTop-p.clientHeight<4}}
function apply(z){const d=$('det'),p=d.querySelector('pre');if(p)p.scrollTop=z.bot?p.scrollHeight:z.ps;d.scrollTop=z.st}
function place(){
  const d=$('det'),o=open!=null,wide=mq.matches;
  d.hidden=!o;$('bd').hidden=!o||wide;document.body.classList.toggle('lock',o&&!wide);
  if(o&&wide){const r=document.querySelector(`.row[data-i="${open}"]`),c=r&&r.closest('.card');c?c.after(d):document.body.append(d)}}
function render(){
  const z=snap();document.body.append($('det'));
  $('ban').innerHTML=ban();$('pg').style.opacity=err?.55:1;
  if(S){
    const [t,c]=chip();$('chip').textContent='reaper '+t;$('chip').className='chip c-'+c;
    $('tiles').innerHTML=tiles();$('ft').innerHTML=foot();
    const bl=S.blocks.slice().sort((a,b)=>(a.block=='other')-(b.block=='other')||a.block.localeCompare(b.block));
    $('blocks').innerHTML=bl.length?bl.map(card).join(''):`<div class=card>No instances yet. Workers appear after their first beat (30 s) or the next vast listing (60 s).${S.reaper.enabled?'':'<br>Reaper off: only instances that send beats are listed.'}</div>`}
  place();apply(z)}
function tick(){
  if(!ok){$('upd').textContent=err?'unreachable':'loading…';$('upd').className='s12 '+(err?'c-rd':'mu');return}
  const s=(Date.now()-ok)/1e3;
  $('upd').textContent=(err?'unreachable · last ok ':'updated ')+ago(s)+' ago';$('upd').className='s12 '+(err||s>45?'c-rd':'mu')}
function spark(n,ts,a,max,u,d,cls){
  const v=a.filter(x=>x!=null),m=max||Math.max(1,...v),t0=ts[0],span=(ts[ts.length-1]-t0)||1,segs=[];
  let cur=null,lt=0;
  a.forEach((x,i)=>{
    if(x==null){cur=null;return}
    if(!cur||ts[i]-lt>90){cur=[];segs.push(cur)}
    lt=ts[i];cur.push(`${((ts[i]-t0)/span*120).toFixed(1)},${(1+22*(1-Math.min(x/m,1))).toFixed(1)}`)});
  const pl=segs.map(s=>`<polyline points="${(s.length<2?[s[0],s[0]]:s).join(' ')}" fill=none stroke=currentColor stroke-width=1.5 stroke-linecap=round stroke-linejoin=round vector-effect=non-scaling-stroke />`).join('');
  return `<div class=${cls||'sp'}><div class=sph><span class="s12 mu">${n}</span><b>${v.length?f(v[v.length-1],d)+u:'—'}</b></div><svg viewBox="0 0 120 24" preserveAspectRatio=none>${pl}</svg><div class="s12 mu">min ${v.length?f(Math.min(...v),d):'—'} · max ${v.length?f(Math.max(...v),d):'—'}</div></div>`}
const evRow=(e,b)=>`<div class="er ${e.level}"><i class=lb></i><time class=s12 title="${esc(new Date(e.ts*1e3).toLocaleString())}">${hm(e.ts)}</time>${b?`<span class=bdg>${esc(e.block=='other'?e.label:e.block)}</span>`:''}<span class=et>${e.level!='info'?`<b class=c-${e.level=='error'?'rd':'ye'}>${e.level}</b> `:''}${esc(e.line.replace(/^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d /,''))}</span></div>`;
function dtl(){
  if(open==null)return;
  const z=snap(),d=$('det'),s=flat().find(x=>x.inst===open)||{},R=S?S.reaper:{};
  let h=`<div class=top><b>${esc((D&&D.label)||s.label||'')}</b><span class=mu>#${open}</span><button id=x class=xb aria-label=Close>×</button></div>`;
  h+=`<div class="s12 mu">${[s.gpu,s.host,s.dph!=null?'$'+f(s.dph,2)+'/h':'',s.up_min!=null?'up '+upt(s.up_min):''].filter(Boolean).map(esc).join(' · ')}</div>`;
  if(dErr)h+=`<p class=c-rd>${esc(msg(dErr))}</p>`;
  if(D){
    const q=D.series,l=D.latest||{},t=q.ts;
    h+=`<div class=sps>${spark('GPU',t,q.gpu_util,100,'%',0)}${spark('VRAM',t,q.vram_gb,l.vram_total_gb,' GB',1)}${spark('CPU',t,q.cpu_pct,100,'%',0)}${spark('RAM',t,q.ram_gb,l.ram_total_gb,' GB',1)}${spark('Net rx+tx',t,q.net_mbps,0,' Mb/s',1,'sp net')}</div>`;
    h+=`<h3>Log tail</h3><pre>${esc(D.tail)}</pre><h3>Events</h3>${D.events.slice(0,50).map(e=>evRow(e,0)).join('')||'<span class="s12 mu">none</span>'}`}
  if(S&&s.status!='not ours')h+=`<button id=k class=kb${R.enabled?'':' disabled'}>${R.enabled?'Destroy #'+open:'kill needs MONITOR_VAST_KEY'}</button>`;
  if(kmsg)h+=`<p class=s12>${esc(kmsg)}</p>`;
  d.innerHTML=h;apply(z)}
async function loadD(){
  const i=open;if(i==null)return;
  try{const x=await api('/api/inst/'+i);if(i!==open)return;D=x;dErr=null}catch(e){dErr=e}
  dtl()}
function openI(i){open=i;D=null;dErr=null;kmsg='';render();dtl();loadD()}
function close(){open=null;render()}
async function kill(){
  const lb=(D&&D.label)||'';
  if(!confirm(`Destroy ${lb} (#${open})? Clips since the last HF sync (≤ 15 min) are lost.`))return;
  try{const j=await api('/api/kill/'+open,{method:'POST'});kmsg=j.destroyed?'destroyed':j.dry_run?'dry-run: logged, not destroyed':'not destroyed'}catch(e){kmsg=e.b||msg(e)}
  dtl();poll()}
function evs(){
  const ch=(g,v,l,cur)=>`<button class=ch data-g=${g} data-v="${esc(v)}" aria-pressed=${cur===v}>${esc(l)}</button>`;
  $('chips').innerHTML=`<div class=chs>${[['','All'],['warn','Warn+'],['error','Error']].map(x=>ch('lv',x[0],x[1],lv)).join('')}</div><div class=chs>${ch('bk','','All blocks',bk)}${S?S.blocks.map(b=>ch('bk',b.block,b.block,bk)).join(''):''}</div>`;
  $('ev').innerHTML=EV.length?EV.map(e=>evRow(e,1)).join(''):'<p class=mu>No events for this filter.</p>'}
async function loadEv(){try{EV=await api(`/api/events?level=${lv}&block=${encodeURIComponent(bk)}&limit=100`)}catch(e){}evs()}
async function poll(){
  try{S=await api('/api/state');ok=Date.now();err=null}catch(e){err=e}
  render();if(!err)await loadEv();tick()}
document.onclick=e=>{
  const t=e.target,r=t.closest('[data-i]'),c=t.closest('.ch');
  if(t.closest('#x')||t.id=='bd')return close();
  if(t.closest('#k'))return kill();
  if(c){if(c.dataset.g=='lv')lv=c.dataset.v;else bk=c.dataset.v;return loadEv()}
  if(r&&!t.closest('#det'))openI(+r.dataset.i)};
document.onkeydown=e=>{
  if(e.key=='Escape'&&open!=null)close();
  else if((e.key=='Enter'||e.key==' ')&&e.target.matches('.row')){e.preventDefault();openI(+e.target.dataset.i)}};
mq.onchange=()=>render();
poll();setInterval(poll,15e3);setInterval(loadD,3e4);setInterval(tick,1e3);
</script></body></html>"""


# --- selftest ---------------------------------------------------------------------------------------------------

def selftest():
    os.environ.pop("MONITOR_VAST_KEY", None)
    app = App(":memory:", "t")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def call(method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        c.request(method, path, json.dumps(body) if body is not None else None, headers or {})
        r = c.getresponse()
        return r.status, r.read(), r

    bearer = {"Authorization": "Bearer t"}
    beat = {"inst": 1, "label": "kvs-x-1of2", "status": "running", "clips_done": 6, "clips_total": 12, "used_h": 1,
            "budget_h": 10, "events": ["2026-10-26 09:00:00 START b1:x (est 1.00 h, used 0 h)",
                                       "2026-10-26 09:10:00 END b1:x: rc=1, 0.17 h (used 0.17 h)"]}
    assert call("POST", "/hb", beat)[0] == 401
    assert call("POST", "/hb", beat, bearer)[0] == 200
    st = json.loads(call("GET", "/api/state", headers=bearer)[1])
    blk = st["blocks"][0]
    assert blk["block"] == "x" and blk["shards"][0]["shard"] == "1/2" and blk["clips_done"] == 6, st
    assert blk["shards"][0]["eta_h"] == 1.0, st
    ev = json.loads(call("GET", "/api/events", headers=bearer)[1])
    assert sorted(e["level"] for e in ev) == ["error", "info"] and {e["block"] for e in ev} == {"x"}, ev
    code, _, r = call("GET", "/?t=t")
    assert code == 302 and "kvs_t=" in r.getheader("Set-Cookie")
    cookie = r.getheader("Set-Cookie").split(";")[0]
    code, body, _ = call("GET", "/", headers={"Cookie": cookie})
    assert code == 200 and b"<html" in body
    assert parse_label("kvs-screen-A") == ("screen", "A") and parse_label("foo") == ("other", "foo")
    now = time.time()
    rows = {111: {"label": "kvs-x-1of2", "actual_status": "running", "start_date": now - 7200, "dph_total": 0.4},
            222: {"label": "kvs-monitor", "actual_status": "running", "start_date": now - 60, "dph_total": 0.05},
            444: {"label": "someone-else", "actual_status": "running", "start_date": now - 7200, "dph_total": 1.0}}
    assert now - stale_ref(rows[111], None, now, 600) <= 600  # fresh monitor: no no-beat kill for #111
    c = classify(rows, {111: ({"status": "running"}, now)}, now)
    assert (c["totals"]["running"], c["totals"]["dead"], c["totals"]["dph_now"], c["monitor"]["inst"]) == (1, 0, 0.4, 222), c
    assert [(s["inst"], s["dot"], s["status"]) for s in c["blocks"][0]["shards"]] == [(444, "grey", "not ours")], c
    c = classify({}, {555: ({"label": "kvs-x-1of2", "status": "running", "clips_done": 5, "clips_total": 10}, now - 3000)}, now, True)
    s = c["blocks"][0]["shards"][0]
    assert (s["status"], s["dot"], c["totals"]["running"], c["totals"]["dead"], c["blocks"][0]["clips_total"]) == ("gone", "grey", 0, 0, 10), c
    print("selftest ok")


def demo(port):
    """Fake fleet for looking at the page: python scripts/monitor.py --demo (no env needed)."""
    now = time.time()
    app = App(":memory:", "demo", "demo", "", True)
    app.reaper["last_ok"] = now - 40
    vr = lambda label, st, dph, age_h, **kw: dict(dict(gpu_name="RTX 6000Ada", machine_id=8821, geolocation="CZ"), label=label,
                                                 actual_status=st, dph_total=dph, start_date=now - age_h * 3600, **kw)
    app.vast = {1001: vr("kvs-cosmos2b-1of4", "running", .42, 4.2), 1002: vr("kvs-cosmos2b-2of4", "running", .42, 3),
                1003: vr("kvs-cosmos2b-3of4", "running", .42, 2), 1004: vr("kvs-cosmos2b-4of4", "loading", .42, .2),
                2001: vr("kvs-wan5b-1of3", "running", .42, 4), 2002: vr("kvs-wan5b-2of3", "running", .42, .3),
                2003: vr("kvs-wan5b-3of3", "exited", .42, 3),
                3001: vr("jupyter-sarthak", "running", .30, 2, gpu_name="RTX 4090"),
                9001: vr("kvs-monitor", "running", .06, 26, gpu_name="GTX 1650", public_ipaddr="1.2.3.4")}
    # inst: (label, last beat age s, step, status, done, total, used_h, gpu_util)
    fleet = {1001: ("kvs-cosmos2b-1of4", 20, "b1:cosmos2b", "running", 61, 144, 4.2, 97),
             1002: ("kvs-cosmos2b-2of4", 90, "b1:cosmos2b", "running", 40, 144, 3, 91),
             1003: ("kvs-cosmos2b-3of4", 360, "b1:cosmos2b", "running", 22, 144, 2, 0),
             2001: ("kvs-wan5b-1of3", 15, "done", "finalize", 144, 144, 3.9, 3),
             2002: ("kvs-wan5b-2of3", 10, "b1:wan5b", "running", 2, 144, .3, 88),
             2003: ("kvs-wan5b-3of3", 1500, "b1:wan5b", "running", 90, 144, 2.5, 95)}
    rows = []
    for inst, (label, age, step, st, done, total, used, gpu) in fleet.items():
        for k in range(int((7200 - age) / 30)):
            ts = now - age - 30 * k
            if inst == 1001 and now - 2400 < ts < now - 1800:  # the 10 min gap
                continue
            w = math.sin(ts / 90 + inst)
            rows.append((ts, inst, json.dumps({
                "inst": inst, "label": label, "status": st, "step": step, "clips_done": done, "clips_total": total,
                "used_h": used, "budget_h": 10, "gpu": "RTX 6000Ada", "host": "machine 8821, CZ",
                "gpu_util": max(0, min(100, gpu + 3 * w)), "vram_gb": 19.4 + w, "vram_total_gb": 47.5,
                "cpu_pct": 23 + 8 * w, "ram_gb": 21 + w, "ram_total_gb": 63, "net_rx_mbps": 2.1 + w, "net_tx_mbps": 14.8 + 3 * w,
                "events": [], "tail": chr(10).join(f"step {n} loss 0.{n % 97:02d} {label}" for n in range(40))})))
    with app.lock:
        app.db.executemany("INSERT INTO beats VALUES(?,?,?)", rows)
        app.db.commit()
    stamp = lambda ts, s: time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(ts)) + s
    for ago_s, inst, label, line, lv in [
            (7000, 1001, "kvs-cosmos2b-1of4", "START b1:cosmos2b (est 4.00 h, used 0 h)", None),
            (6500, 2001, "kvs-wan5b-1of3", "END b1:wan5b: rc=0, 3.90 h (used 3.90 h)", None),
            (5000, 2002, "kvs-wan5b-2of3", "SKIP b2:wan5b: already done", None),
            (3000, 1002, "kvs-cosmos2b-2of4", "END b1:cosmos2b: rc=1, 0.80 h (used 0.80 h)", None),
            (1500, 2001, "kvs-wan5b-1of3", "guard triggered on kvs-wan5b-1of3", "error"),
            (1200, 2999, "kvs-wan5b-9of9", "KILL kvs-wan5b-9of9 #2999: no beat for 11 min", None),
            (900, 0, "reaper", "reaper: vastai rc=1: too frequent", "warn"),
            (210, 1003, "kvs-cosmos2b-3of4", "stale: no beat 3 min, vast running", "warn")]:
        ts = now - ago_s
        app.event(inst, label, stamp(ts, line) if not line.startswith(("KILL", "guard", "reaper", "stale")) else line, lv, ts)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    print(f"http://localhost:{srv.server_address[1]}/?t=demo", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--db", default=str(Path(__file__).with_name("monitor.db")))
    ap.add_argument("--stale-min", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="serve a fake fleet, token 'demo', no env needed")
    a = ap.parse_args()
    if a.demo:
        return demo(a.port)
    if a.selftest:
        return selftest()
    token = os.environ.get("MONITOR_TOKEN")
    if not token:
        sys.exit("MONITOR_TOKEN is required")
    app = App(a.db, token, os.environ.get("MONITOR_VAST_KEY", ""), os.environ.get("MONITOR_NTFY", ""),
              a.dry_run, a.stale_min)
    threading.Thread(target=reaper_loop, args=(app,), daemon=True).start()
    print(f"monitor on :{a.port}, db {a.db}, dry-run={a.dry_run}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
