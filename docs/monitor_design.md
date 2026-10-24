# Run monitor: design

One page, reachable from a phone, that shows every rented GPU, what it is doing, its live system stats and logs, and
kills instances whose work has died. Companion to `docs/vast_plan.md` (waves, shards, launcher).

## What exists already

- Every pod runs `scripts/guard.py`: a 60 s loop that watches the pod's processes and, when they are gone, pushes to
  HF and destroys the instance with its own `CONTAINER_API_KEY` (`finalize.stop_pod`). Auto-termination on a script
  crash is therefore done per pod, with no central piece. What it cannot catch: the whole container dying (host OOM,
  host offline, pre-emption), because then the guard dies with it.
- `run_pilot.py` logs `START / END / SKIP / DONE-EARLIER` lines per step to `data/logs/pilot{tag}.log` and the budget
  state to `data/logs/budget{tag}.json`; `sync.py` pushes both to HF every 15 min. Progress per shard is already on
  disk as `runs/{exp}/{model}/*/meta.json` counts.
- The vast.ai API lists instances with `actual_status`, `gpu_util`, `cpu_util`, `mem_usage`, `machine_id`,
  `geolocation`, `cur_state`.

The monitor adds three things: a heartbeat from each pod (richer and faster than the 15 min HF push), one server that
keeps them and serves the page, and a reaper for containers that stopped heartbeating but still bill.

## Pieces

### 1. Heartbeat: `guard.py --monitor URL` (same loop, +~60 lines)

Every 30 s the guard POSTs one JSON object to `URL/hb` with header `Authorization: Bearer $MONITOR_TOKEN`:

```json
{"inst": 12345678, "label": "kvs-cosmos2b-3of15", "block": "cosmos2b", "shard": "3/15", "host": "machine 8821, CZ",
 "gpu": "RTX 6000Ada 48 GB", "status": "running",
 "step": "b1:cosmos2b", "clips_done": 61, "clips_total": 144, "used_h": 4.2, "budget_h": 10,
 "gpu_util": 97, "vram_gb": 19.4, "vram_total_gb": 47.5, "cpu_pct": 23, "ram_gb": 21.0, "ram_total_gb": 63,
 "disk_gb": 112, "net_rx_mbps": 2.1, "net_tx_mbps": 14.8,
 "events": ["2026-10-26 09:14:02 END b1:cosmos2b: rc=0, 3.95 h"],
 "tail": "last 30 lines of the current step log"}
```

Sources: `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`,
`/proc/stat` and `/proc/net/dev` deltas, `/proc/meminfo`, `shutil.disk_usage`, `budget{tag}.json`, the pilot log's new
lines since the last beat (these become the global event log), and the newest `data/logs/step*.log`. `clips_total`
comes from the shard's plan. A failed POST is ignored: the guard's own job (push + destroy) never depends on the
monitor.

### 2. Server: `scripts/monitor.py` (one file, stdlib only: `http.server`, `sqlite3`, `json`, `threading`)

- `POST /hb` → insert into `beats(ts, inst, json)`, append `events(ts, inst, line)` for each new pilot-log line.
- `GET /` → the page (inline HTML + ~150 lines of vanilla JS; mobile-first single column).
- `GET /api/state` → latest beat per instance, grouped by block, plus block totals (clips done / total, $ so far from
  `used_h × dph`, ETA from the shard's own clips/hour).
- `GET /api/inst/{id}` → last 2 h of beats (for the small util/VRAM/net sparklines), last 200 events, log tail.
- `GET /api/events?level=` → global event log (every START/END/SKIP/fail line from every pod, newest first).
- `POST /api/kill/{id}` → manual destroy (confirm on the page).
- Reaper thread, every 60 s, with a vast.ai key scoped to `instance_read` + destroy only:
  - `vastai show instances --raw`; only instances whose `label` starts with `kvs-` are ever touched.
  - running, no heartbeat for 10 min → log event, send one `ntfy` push, destroy. The last ≤ 15 min of clips are the
    loss (the pod's own sync push), the same as a vast.ai pre-emption.
  - `exited` / `offline` for 10 min → destroy (a dead container still bills its disk).
  - `used_h > budget_h + 1` → destroy (the guard's `--hard-at` should have done it; this is the second fence).
  - Every kill writes `events` and the page's red banner; nothing is silent.
- SQLite file `monitor.db` next to the script; beats older than 24 h are deleted hourly. History beyond that is the
  logs on HF, which the page links to per pod (`data/logs/pilot_full_*.log`).

### 3. Page

Top: counters (instances running / idle / dead, GPUs by class, $/h now, $ spent, clips done / total, ETA of the slowest
block). Then one card per block, each a row per shard:

    cosmos2b  ████████░░ 61/144  RTX 6000Ada  CZ  97% gpu  19/48 GB  4.2/10 h  $2.5  b1:cosmos2b   ●

Tap a row → that instance: 2 h sparklines for GPU %, VRAM, CPU %, RAM, net rx/tx; the log tail; its events; the
kill button. A red dot and a banner for any instance without a beat for > 2 min. Global event log at the bottom,
filterable by block and by level (END rc≠0, SKIP, kill). No charts library, no framework, no build step.

### 4. Where it runs, how the phone reaches it

Run it on its own cheapest instance, not on a worker: a worker destroys itself when its shard is verified, which is
exactly when its monitor would vanish. The cheapest vast.ai GPU (GTX 1650 class, `static_ip=true`,
`direct_port_count>=1`) is $0.05–0.10/h, about $3 for a two-day run, and keeps the monitor's lifetime independent of
every shard. (If you would rather co-locate: the same `monitor.py` runs on any worker; the launcher then has to hand
the next worker the SQLite file, which is the complexity the $3 buys off.)

`vast_launch.py monitor` creates it (`-p 8787:8787` in `env`, label `kvs-monitor`, onstart runs `monitor.py`) and
prints `http://PUBLIC_IP:VAST_TCP_PORT_8787/?t=TOKEN`. Workers get `MONITOR_URL` from the launcher's onstart command;
`MONITOR_TOKEN` is a vast.ai account-level environment variable, like `HF_TOKEN`, never in the template.

The page is plain HTTP with a bearer token: fine on mobile data, weak on shared Wi-Fi. HTTPS without any account or
DNS: `cloudflared tunnel --url http://localhost:8787` on the monitor instance gives a random `https://*.trycloudflare.com`
URL (one binary, logged at start, changes if the monitor restarts). Phone alerts: one `curl -d "$msg" ntfy.sh/$TOPIC`
from the reaper and from `finalize` (shard done / verified / failed); the ntfy app subscribes to the topic, no account.

### 5. Security

- The monitor's vast.ai key is a scoped key (`instance_read`, `instance_write` destroy), created on the laptop, passed as
  an account environment variable; the account key never leaves `~/.vast_key`. The reaper touches `kvs-*` labels only.
- `MONITOR_TOKEN` is required on every endpoint, including `GET /`. Tokens never appear in logs (the guard logs
  "beat failed: HTTP 401", not the header).
- The monitor never holds `HF_TOKEN`: it reads nothing from HF and pushes nothing; data flow is pod → HF as today.

## Failure modes

| What dies | Effect | Recovery |
|---|---|---|
| A worker's script | guard pushes, verifies, destroys (unchanged) | launcher's `vast_status` relaunches the shard |
| A worker's container | no beats → reaper destroys after 10 min; ≤ 15 min of clips lost | same |
| The monitor | workers unaffected (beats fail silently); page down, reaper down | relaunch `monitor` (history since last HF push lost; logs are on HF) |
| vast.ai API 429 | reaper skips a cycle, backs off (no `Retry-After` is sent) | automatic |
| Wrong kill | impossible outside `kvs-*` labels; every kill is an event + ntfy push | relaunch the shard |

## Size and order

1. `guard.py --monitor`: heartbeat collection (nvidia-smi, /proc, budget json, log tail). ~60 lines. Testable on a
   laptop with `--dry-run --monitor http://localhost:8787` (prints the payload).
2. `monitor.py`: server + reaper + page. ~250 lines Python, ~150 lines HTML/JS. Reaper has `--dry-run` (log, never
   destroy).
3. Launcher: `monitor` subcommand, `MONITOR_URL` in the worker onstart, `kvs-{block}-{k}of{N}` labels.
4. Optional: `cloudflared` line in the monitor onstart, `ntfy` topic.

About half a day of work. Do it before wave 0, so the calibration instances are the first thing the page shows.

## Not built on purpose

Grafana/Prometheus, a database server, user accounts, websockets, a mobile app, historical dashboards beyond 24 h,
per-process GPU accounting. All of it is replaceable by the HF logs and `vastai show instances` after the run.
