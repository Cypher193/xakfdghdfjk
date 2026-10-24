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
    st = classify(dict(app.vast), latest_beats(app), now)
    kills = app.q("SELECT ts, inst, label, line FROM events WHERE line LIKE 'KILL %' AND ts > ? ORDER BY ts DESC",
                  (now - 86400,))
    return dict(st, now=now, reaper=app.reaper, kills=[dict(zip(("ts", "inst", "label", "line"), k)) for k in kills])


def classify(vast, beats, now):
    """Pure: vast rows {inst: row} + latest beats {inst: (beat, ts)} -> totals, monitor, blocks."""
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
        blk["spent"] += s["spent"] or 0
        if s["clips_total"]:
            blk["clips_done"] += s["clips_done"] or 0
            blk["clips_total"] += s["clips_total"]
        if s["eta_h"] is not None:
            blk["eta_h"] = max(blk["eta_h"] or 0, s["eta_h"])
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
            out.append({"ts": ts, "inst": inst, "label": label, "level": lv, "line": line})
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
        rc, out, err = vastai(app, "destroy", "instance", str(inst))
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

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>kvs monitor</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--card:#f6f6f6}
@media(prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--card:#1b1b1b}}
body{margin:0 auto;max-width:640px;padding:8px 16px 40px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin:10px 0}
.row{padding:6px 0;border-top:1px solid var(--line);cursor:pointer;overflow-wrap:anywhere}.mut{color:var(--mut)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:#888}
.green{background:#2a2}.yellow{background:#ca0}.red{background:#d33}.loading{background:none;border:1px solid #888;box-sizing:border-box}
#ban{background:#d33;color:#fff;padding:8px;border-radius:8px;margin:8px 0}
pre{overflow:auto;max-height:260px;font-size:11px;background:var(--card);padding:6px}
progress{width:70px;vertical-align:middle}svg{vertical-align:middle;margin-left:6px}
.error{color:#d33}.warn{color:#b80}button,select{font:inherit;padding:6px}
</style></head><body>
<h3 style="margin:8px 0">kvs monitor</h3><div id=ban hidden></div><div id=hdr></div><div id=det></div><div id=blocks></div>
<h4>Events</h4><select id=lv><option value="">all<option value=warn>warn+<option value=error>error</select>
<select id=bk><option value="">all blocks</select><div id=ev></div>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const f=(v,d=1)=>v==null?'—':(+v).toFixed(d), $=id=>document.getElementById(id);
const hm=ts=>new Date(ts*1000).toLocaleTimeString([],{hour12:false});
let open=null, S=null;
async function api(p,o){const r=await fetch(p,o);if(!r.ok)throw r.status;return r.json()}
function row(s){
  const bar=s.clips_total?`<progress value=${s.clips_done||0} max=${s.clips_total}></progress> ${s.clips_done??'?'}/${s.clips_total}`:`${s.clips_done??'?'}/?`;
  const body=s.dot=='loading'||s.status=='not ours'||s.status=='loading'&&s.age_s==null
    ?`<span class=mut>${esc(s.status)} ·${esc(s.actual_status)} · ${f(s.up_min,0)} min</span>`
    :`${bar} <span class=mut>${esc(s.gpu)} · ${esc(s.host)} · ${f(s.gpu_util,0)}% gpu · ${f(s.vram_gb,0)}/${f(s.vram_total_gb,0)} GB · ${f(s.used_h)}/${f(s.budget_h,0)} h · $${f(s.spent,2)} · ${esc(s.step)} · eta ${f(s.eta_h)} h</span>`;
  return `<div class=row data-i=${s.inst}><span class="dot ${s.dot}"></span><b>${esc(s.shard)}</b> ${body}</div>`}
function render(){
  const t=S.totals, gp=Object.entries(t.gpus).map(([k,n])=>n+'×'+k).join(', ');
  $('hdr').innerHTML=`<div>${t.running} running · ${t.loading} loading · ${t.dead} dead · $${f(t.dph_now,2)}/h · $${f(t.spent,2)} spent</div>
   <div class=mut>${t.clips_done}/${t.clips_total} clips · slowest ETA ${f(t.eta_h)} h · ${esc(gp)}${S.monitor.inst?' · monitor $'+f(S.monitor.dph,2)+'/h':''}
   · reaper ${S.reaper.enabled?(S.reaper.dry_run?'dry-run':'on'):'off'}${S.reaper.last_err?' ('+esc(S.reaper.last_err)+')':''}</div>`;
  const red=S.blocks.some(b=>b.shards.some(s=>s.dot=='red')), k=S.kills.filter(k=>k.ts>S.now-3600);
  $('ban').hidden=!(red||k.length);
  $('ban').textContent=(red?'Instance without heartbeat. ':'')+k.map(k=>k.line).join(' | ');
  $('blocks').innerHTML=S.blocks.map(b=>`<div class=card><b>${esc(b.block)}</b> <span class=mut>${b.clips_done}/${b.clips_total} · $${f(b.spent,2)} · eta ${f(b.eta_h)} h</span>${b.shards.map(row).join('')}</div>`).join('');
  const sel=$('bk'), cur=sel.value;
  sel.innerHTML='<option value="">all blocks'+S.blocks.map(b=>`<option>${esc(b.block)}`).join('');sel.value=cur;
}
function spark(n,a,max){
  const m=max||Math.max(1,...a.filter(x=>x!=null));
  const p=a.map((v,i)=>v==null?'':`${(i/Math.max(a.length-1,1)*120).toFixed(1)},${(24-24*v/m).toFixed(1)}`).filter(Boolean).join(' ');
  return `<div>${n} <b>${f(a[a.length-1])}</b><svg viewBox="0 0 120 24" width=120 height=24><polyline fill=none stroke=currentColor points="${p}"/></svg></div>`}
async function detail(){
  if(open==null){$('det').innerHTML='';return}
  const d=await api('/api/inst/'+open), s=d.series, l=d.latest;
  $('det').innerHTML=`<div class=card><b>${esc(d.label)}</b> #${d.inst} <button id=x>close</button>
   ${spark('gpu %',s.gpu_util,100)}${spark('vram GB',s.vram_gb,l.vram_total_gb)}${spark('cpu %',s.cpu_pct,100)}
   ${spark('ram GB',s.ram_gb,l.ram_total_gb)}${spark('net Mb/s',s.net_mbps)}
   <pre>${esc(d.tail)}</pre><pre>${d.events.map(e=>hm(e.ts)+' '+esc(e.line)).join('\n')}</pre>
   <button id=k>kill instance</button></div>`;
  $('x').onclick=()=>{open=null;detail()};
  $('k').onclick=async()=>{if(confirm('Destroy '+d.label+' #'+d.inst+'?')){await api('/api/kill/'+d.inst,{method:'POST'});poll()}};
}
async function events(){
  const e=await api(`/api/events?level=${$('lv').value}&block=${encodeURIComponent($('bk').value)}`);
  $('ev').innerHTML=e.map(e=>`<div class="${e.level}">${hm(e.ts)} <span class=mut>${esc(e.label)}</span> ${esc(e.line)}</div>`).join('');
}
async function poll(){try{S=await api('/api/state');render();events()}catch(e){$('ban').hidden=false;$('ban').textContent='monitor unreachable '+e}}
$('blocks').onclick=e=>{const r=e.target.closest('.row');if(r){open=+r.dataset.i;detail();$('det').scrollIntoView()}};
$('lv').onchange=$('bk').onchange=events;
poll();setInterval(poll,15000);setInterval(detail,30000);
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
    assert sorted(e["level"] for e in ev) == ["error", "info"], ev
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
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--db", default=str(Path(__file__).with_name("monitor.db")))
    ap.add_argument("--stale-min", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
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
