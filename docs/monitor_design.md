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

## Appendix: implementation decisions

Settled before coding; the code follows these exactly.

1. **Auth.** A request with `?t=TOKEN` matching the token gets `302` to the same path without `?t` plus `Set-Cookie: kvs_t=TOKEN; HttpOnly; SameSite=Strict; Path=/; Max-Age=604800` (no `Secure`: plain HTTP). Every route accepts `Authorization: Bearer` or the `kvs_t` cookie, compared with `hmac.compare_digest`; otherwise `401`, empty body. The token is never logged.
2. **Refresh.** Page polls `/api/state` every 15 s; an open detail polls `/api/inst/{id}` every 30 s. Dots by time since last beat: green ≤ 1 min, yellow > 1 min, red > 2 min, grey when vast says exited/offline. Any red dot shows the banner.
3. **Sparklines.** One inline `<svg viewBox="0 0 120 24">` per series, a `<polyline>` with `stroke=currentColor`, no axes, current value as text. Series: gpu_util (0–100), vram_gb (0–vram_total_gb), cpu_pct (0–100), ram_gb (0–ram_total_gb), net rx+tx Mb/s (0–max seen). Last 2 h of beats, not downsampled.
4. **ETA / $.** Shard rate = clips_done / used_h; shard ETA_h = (clips_total − clips_done) / rate, "—" if clips_done < 3. Block ETA = max over shards. Shard $ = (now − start_date)/3600 × dph_total (vast bills setup too); block $ = sum; header "$/h now" = sum of dph_total over our workers vast does not report exited/offline (loading is billed too). No vast row → "—". clips_total missing or 0 → `61/?`, no bar, ETA "—", shard left out of the block total.
5. **Labels.** `^kvs-([a-z0-9_]+)-(\d+)of(\d+)$` → block g1, shard "k/N". `^kvs-screen-(A|B|C|all)$` → block "screen", shard = tier. `kvs-monitor` → header line (`monitor` field from the first such vast row), not a block or total. Else block "other", shard = full label; a vast row whose label does not start with `kvs-` is "not ours": grey dot, excluded from every total and the banner. The vast label (joined on inst) wins over the beat's label. Guard fallback label: `"kvs-" + tag.strip("_").replace("_", "-")`; a beat label matching `^C\.\d+$` is ignored (vast's `VAST_CONTAINERLABEL` is the `C.{id}` name, not the user label).
6. **Loading.** An instance vast lists with no beats gets a row in its block, hollow grey dot, `loading · {actual_status} · {min since start_date} min`, stats blank; yellow after 30 min, red after 60 (not reaped). Reaper grace: the no-beat rule counts only when actual_status == running, from max(last beat, start_date + 30 min, monitor start + stale_min) (the last term stops a restarted monitor reaping before beats arrive; same reference for first-seen-dead).
7. **SQLite.** `PRAGMA journal_mode=WAL; synchronous=NORMAL`; one connection (`check_same_thread=False`) behind one `threading.Lock` under `ThreadingHTTPServer`. `ts` unix float. Indexes `beats(inst, ts)`, `events(ts)`, `events(inst, ts)`. Latest beat per inst: `SELECT inst, json, max(ts) FROM beats GROUP BY inst`. Retention on every 60th reaper cycle (also when the reaper is disabled): `DELETE FROM beats WHERE ts < now − 86400`; events kept. Default `--db` = `monitor.db` next to the script.
8. **Guard step and status.** step = name of the last `START (\S+) ` with no later `END {name}:`; else "done" if `PILOT DONE` exists, else null. status ∈ setup (only setup*.sh alive), running (run_pilot.py), finalize (finalize.py), triggered (one last beat before the push), dry-run. used_h = (now − pilot_start{tag}) / 3600 live; budget_h and clips_total from the budget json. Missing budget json → both null; missing start file → used_h and clips_done null; missing logs → events [] and tail "". First beat sends the last 20 pilot-log lines, then reads from a byte offset (reset to 0 if the file shrank). POST timeout 5 s; collect + post inside one `try/except Exception`. Log via `finalize.log` only on change: "monitor beats ok" / "beat failed: HTTP 401" (or the exception type) / "beat ok again". Guard log is `guard{tag}.log`.
9. **Laptop dry run.** The CONTAINER_ID/RUNPOD_POD_ID exit moves below the dry-run branch. inst = int(CONTAINER_ID or 0); two samples 1 s apart; prints the JSON and POSTs once if `--monitor` is set. `alive()` catches FileNotFoundError → []. No nvidia-smi → gpu, gpu_util, vram_gb, vram_total_gb null. No /proc or cgroup → cpu_pct, ram_gb, ram_total_gb, net null. disk_gb = free GB of `shutil.disk_usage(ROOT)`; host = `socket.gethostname()` (the page shows vast machine_id + geolocation when it has them).
10. **CPU / RAM / net.** cgroup v2 first: cpu_pct = Δ`cpu.stat` usage_usec / (Δt · 1e6 · cpus) × 100, cpus = `cpu.max` quota/period or `os.cpu_count()` if "max"; ram_gb = `memory.current`, ram_total_gb = `memory.max` or MemTotal if "max". Fallback: /proc/stat line 1, busy = 1 − Δ(idle + iowait)/Δsum(first 8 fields). Net: /proc/net/dev, sum over interfaces except lo of rx bytes (col 1) and tx bytes (col 9); Mb/s = Δbytes · 8 / Δt / 1e6. Δt = `time.monotonic()` between beats. First beat or negative delta → null. One decimal.
11. **vast_launch monitor.** Offers: `search offers 'rentable=true verified=true num_gpus=1 reliability>0.98 direct_port_count>=1 static_ip=true inet_down>=100 disk_space>=10' -o dph_total --raw`, try the 3 cheapest. Port: `show instance ID --raw` → `public_ipaddr.strip()`, `ports["8787/tcp"][0]["HostPort"]`. Poll every 10 s up to 10 min until actual_status == running and the port is mapped, then GET / with the bearer every 10 s up to 3 min until 200. On exited/offline or either timeout: `destroy instance ID`, next offer, at most 3 tries, then exit 1.
12. **ntfy.** If `MONITOR_NTFY` is set, POST `https://ntfy.sh/$MONITOR_NTFY`, 5 s timeout, errors ignored, never the token. Kill: `Title: kvs kill {label}`, `Priority: 4`, body `{label} #{inst} destroyed: {reason}; last beat {m} min ago, {used_h}/{budget_h} h, ${spent}`; dry run prefixes `[dry-run] `. First seen dead (once per episode, reset by the next beat): `Title: kvs stale {label}`, `Priority: 3`, body `{label} #{inst}: no beat {m} min, vast {actual_status}; reaped at {stale_min} min`.
13. **API shapes.** `/api/state`: `{now, reaper:{enabled, dry_run, last_ok, last_err}, totals:{running, loading, dead, gpus:{name: n}, dph_now, spent, clips_done, clips_total, eta_h}, monitor:{inst, ip, dph, up_min}, blocks:[{block, clips_done, clips_total, spent, eta_h, shards:[{inst, label, shard, dot, age_s, status, actual_status, gpu, host, step, clips_done, clips_total, used_h, budget_h, gpu_util, vram_gb, vram_total_gb, dph, spent, eta_h}]}], kills:[last 24 h kill events]}`. `/api/inst/{id}`: `{inst, label, latest, vast, series:{ts, gpu_util, vram_gb, cpu_pct, ram_gb, net_mbps}, events:[≤ 200 {ts, level, line}], tail}`. `/api/events?level=&block=&limit=200`: `[{ts, inst, label, level, line}]`, newest first.
14. **Levels.** error: `\bEND \S+: rc=(?!0,)` (rc ≠ 0, including timeout), a status change to triggered, every kill. warn: `\bSKIP \S+:`, first-seen-dead, reaper vastai failures or 429. info: everything else.
15. **Self-check.** `python scripts/monitor.py --selftest`: bind port 0, start the server in a daemon thread with db `:memory:`, token "t", MONITOR_VAST_KEY removed; assert POST /hb without token → 401; POST /hb with Bearer t, label `kvs-x-1of2`, clips 6/12, used_h 1, events [a START, an `END x: rc=1, …`] → 200; GET /api/state → block "x", shard "1/2", clips_done 6, eta_h 1.0; GET /api/events → levels [info, error]; GET /?t=t → 302 with `kvs_t=` in Set-Cookie and GET / with that cookie → 200; label parse of `kvs-screen-A` and `foo` → screen/A and other/foo. Print "selftest ok", exit 0.

## Appendix B: page UI spec

Content max-width 760 px, centred, 16 px gutters; system font stack; sizes 12 (meta, labels, events), 14 (body, rows,
titles at weight 600), 20 (tile values), 11 px monospace for the log tail; line-height 1.35; `tabular-nums` on body;
spacing 4 / 8 / 12 / 16 / 24 (gutter 16, card padding 12, card gap 12, row padding 8, tile gap 8, section gap 24).

**Skeleton.** (a) Sticky header, 48 px, bottom border: "kvs monitor" left; right "updated 12 s ago" (client clock from
the last good fetch, ticks every 1 s, red after 45 s) and a reaper chip: on (green) / dry-run (yellow) / off (grey) /
error (red, `last_err` set) / lagging (yellow, enabled and `last_ok` older than 180 s). (b) Red banner, not sticky: one
tappable line per red dot ("#1003 cosmos2b 3/4 · no beat 6 min", max 3 then "+N more") and one per kill in the last hour
("KILL … · 20 min ago"); tapping opens that instance; `reaper.last_err` is a separate yellow-bar line under it.
(c) Stat tiles. (d) One card per block, "other" last; on ≥ 720 px the detail card sits after the open row's block.
(e) Events. (f) Footer: "monitor #9001 · 1.2.3.4 · $0.06/h · up 26 h · GPUs 4×RTX 6000Ada, 1×RTX 4090", or "monitor not
in vast list".

**Stat tiles.** 3 columns under 720 px, 6 above; label 12 muted, value 20 tabular, optional 12 sub-line. Running
(sub "+N loading"; sub "N late" in yellow if any yellow dot) · Dead (value and label red when > 0) · $/h now ("$1.84") ·
Spent ("$41.20", "$412" when ≥ 100) · Clips (value "43 %", sub "1234/2880") · ETA ("5.8 h", "2.1 d" when ≥ 48 h, "—").

**Block card.** Surface, 1 px border, radius 8, padding 12. Header line 1: block name, shard counts by word ("live 2 ·
late 1 · stale 1 · loading 1", zeros omitted, stale in red), "$12.40" right-aligned. Header line 2: 6 px div bar (accent
on border track), "61/144", "eta 5.8 h". Shard rows sorted by k; top border, padding 8 px 0, min-height 48, whole row is
the tap target; open row gets `aria-current` and a 3 px accent left border. Line 1: 10 px dot, status word (min-width 64,
coloured text), shard id (600), step (flex 1, muted, ellipsis; when status ≠ running the status word comes first, e.g.
"finalize", "triggered b1:x"), eta right-aligned. Line 2: bar (flex 1, min 48 px), "61/144", "97%", "19/48G", "4.2/10h",
"$2.50", 8 px gaps, no separators, flex-wrap fallback. Words: green "live", yellow "late", red "stale", grey = vast
`actual_status` or "gone", loading "loading", step "done" → "done" (green, full bar). Loading row: hollow dot (yellow /
red fill after 30 / 60 min), "loading", shard, `actual_status` in the step slot; line 2 "12 min since start · $0.42/h", no
bar. Not-ours row: grey dot, "not ours", full label as id, step empty, whole row muted; line 2 "gpu · $0.30/h"; no kill
button in its detail.

**Detail.** Under 720 px: fixed bottom sheet (80 vh, scroll, top radius 12, padding 16), backdrop rgba(0,0,0,.45) closes
it, body overflow hidden while open, 40×40 "×" close button (aria-label) in a sticky first line. ≥ 720 px: static card
after the open row's block card, re-attached after each render; Esc closes. Order: identity (label 600 + "#inst"; muted
line gpu · host · $/h · up 4.2 h or "up 34 min"); sparklines in a 2-column grid, net spanning both: label 12 muted with
current value 14/600 right-aligned, svg viewBox 0 0 120 24, width 100 %, height 32, preserveAspectRatio none, polyline
stroke currentColor 1.5, vector-effect non-scaling-stroke, "min 12 · max 98" under it, scales per Appendix 3, line
breaks at null and at ts gaps > 90 s; log tail `<pre>` max-height 240, mono 11/1.35, newest at bottom, re-scrolls to
bottom only if it was at the bottom; events (section rows without the block badge, 50); kill button full width 44 px,
danger background, white text, "Destroy #1003", confirm "Destroy kvs-cosmos2b-3of4 (#1003)? Clips since the last HF sync
(≤ 15 min) are lost.", then an inline result line ("destroyed" / "dry-run: logged, not destroyed" / the error body);
disabled and labelled "kill needs MONITOR_VAST_KEY" when the reaper is off.

**Events.** Heading; two chip groups (single-select buttons, aria-pressed, 40 px tall, padding 0 12, radius 20): All /
Warn+ / Error and All blocks + one per block; pressed chip = text on surface with 2 px border. Row: 3 px level bar left
(red / yellow / transparent), time "09:14" (12 muted tabular, full date in title), block badge (12, 1 px border, radius 4;
the label itself when block is "other"), text with overflow-wrap anywhere; warn/error rows start with the word "warn" /
"error" in the level colour; strip a leading "YYYY-MM-DD HH:MM:SS ". limit=100. Empty: "No events for this filter."
New field: `block` on `/api/events` rows.

**Colours (light / dark).** bg #f3f4f6 / #0e1013 · surface #ffffff / #181b20 · border #d4d8de / #2d323b · text #16181d /
#e7e9ec · muted #596070 / #9ba2ae · green #17803a / #4fc46d · yellow #8a5d00 / #e2b33c · red #c42b1c / #ff6e64 · grey
#636b78 / #8f96a3 · accent #2b5fd9 / #7ea6ff · danger #b42318 / #c4291c (white text). All text pairs ≥ 4.5:1 on surface
and bg. Banner: danger with white text; yellow banner line: text on surface with a yellow left bar.

**Empty and failure.** No instances: one card "No instances yet. Workers appear after their first beat (30 s) or the
next vast listing (60 s)." plus "Reaper off: only instances that send beats are listed." when off; tiles 0 / "—".
Unreachable: keep last data at opacity .55, header "unreachable · last ok 2 min ago" in red, banner "Monitor unreachable
(network / HTTP 502). Retrying every 15 s."; on 401 "Not signed in: open the link with ?t=TOKEN once." A failed detail
fetch shows the same inside the sheet; no unhandled promise.

**Server change.** An instance with beats but no vast row, while the reaper has a `last_ok`, is dot "grey", status
"gone": out of running/dead counts, clips kept in the block totals (a finished or destroyed worker otherwise stays red
for 24 h).

**`--demo`.** db `:memory:`, token "demo", vast key "demo", dry-run, reaper thread not started, `app.vast` set directly,
reaper `last_ok` = now − 40, prints `http://localhost:8787/?t=demo`. Beats every 30 s over 2 h with a 10 min gap in
#1001. cosmos2b (`kvs-cosmos2b-kof4`): 1/4 #1001 green (beat 20 s ago, b1:cosmos2b, 61/144, 4.2/10 h, gpu 97, vram
19.4/47.5, RTX 6000Ada, machine 8821 CZ, $0.42/h); 2/4 #1002 yellow (beat 90 s ago, 40/144); 3/4 #1003 red (beat 6 min
ago, vast running, 22/144); 4/4 #1004 loading (vast loading, start now − 12 min, no beats). wan5b: 1/3 #2001 done (step
"done", status finalize, 144/144, beat 15 s ago); 2/3 #2002 green, 2/144 (ETA "—"); 3/3 #2003 vast exited, last beat
25 min ago. #3001 vast label "jupyter-sarthak", running, $0.30/h, no beats. #9001 kvs-monitor, start now − 26 h, $0.06/h,
ip 1.2.3.4. Events over 2 h: info START / END rc=0; warn "SKIP b2:wan5b: …", "stale: no beat 3 min, vast running" (#1003),
"reaper: vastai rc=1: too frequent" (inst 0, label "reaper"); error "END b1:cosmos2b: rc=1, …" (#1002), "guard triggered
on kvs-wan5b-1of3"; kill "KILL kvs-wan5b-9of9 #2999: no beat for 11 min" 20 min ago.

**Removed from the current page:** both `<select>`s, the `#hdr` text dump, `scrollIntoView`, `<progress>`, gpu/host/eta
text in the shard row, the second `<pre>`, " | "-joined kill lines, max-width 640, the detail re-render that reset the
log scroll.
