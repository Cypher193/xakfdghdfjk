# Know vs Show — pilot

Pilot for *Know vs Show: Probing and Steering Physical Knowledge in Video World Models*.
It checks that the measurement pipeline works before the main run: tracking, scoring, probe signal,
gravity-slope fits, steering and budget. Gates G1–G10 are in [configs/gates.yaml](configs/gates.yaml);
commit them before any pilot run and do not edit thresholds after seeing data.

## Layout

| Path | What |
|---|---|
| `configs/pilot.yaml` | models, input window, render settings, experiments |
| `configs/gates.yaml` | go/no-go thresholds (pre-registered) |
| `pilot/scenes.py` | scene specs with exact analytic ground truth (constant velocity, bounces, two-ball ratios, gravity switches) |
| `pilot/render.py` | Blender renders of the specs (Poly Haven HDRIs and textures) |
| `pilot/models.py` | diffusers generators with activation capture; per-model input adapters (`INPUTS`, cfg `input:`) |
| `pilot/generate.py` | one experiment × one model → `runs/{exp}/{model}/{scene}_s{seed}/`; `--export-jobs` for external repos |
| `pilot/jobs.py`, `scripts/ext/` | MAGI-1 and Self-Forcing: job files, runners in their own venvs, `setup_ext.sh` |
| `pilot/track.py` | SAM2 tracking, metre conversion, kinematic fits; P0 validation on renders and QuantiPhy val |
| `pilot/probe.py` | ridge probes, scene-split CV, VAE / control-task / mean baselines |
| `pilot/metrics.py` | symmetric MRA, original MRA, SILog, γ, residual, CRPS, bootstrap, Holm (`python -m pilot.metrics` self-check) |
| `pilot/download.py` | Physics-IQ and QuantiPhy val clips, real-clip manifest with hand-drawn boxes |

## RunPod setup

Image: `ghcr.io/cypher193/cvpr_vail:latest`, built by GitHub Actions from the [Dockerfile](Dockerfile) on every
change to it or to `requirements.txt` (torch 2.8 / CUDA 12.8 base with SSH, all Python packages, Blender in `/opt/bpy`).
The image holds dependencies only; code and data live on the network volume.

Data lives on a private Hugging Face dataset repo (`CypherNotHere/cvpr-vail-pilot`, override with `HF_DATA_REPO`):
`setup_runpod.sh` pulls it and starts a background push every 15 min (`scripts/sync.py`). Pods are disposable,
so any datacenter and community cloud work. Model weights (~100 GB) are re-downloaded on each new pod.

Pod template:
- Container image: `ghcr.io/cypher193/cvpr_vail:latest` (if the package is private, add GHCR registry credentials
  in RunPod: username = GitHub user, password = a classic token with `read:packages`; or make the package public)
- Container disk 200 GB, no volume (data is synced to HF, so nothing on the pod needs to persist)
- Expose TCP 22 (SSH) and HTTP 8888 (Jupyter)
- Environment: `HF_TOKEN = {{ RUNPOD_SECRET_HF_TOKEN }}` (store the token as a RunPod secret)
- Filter hosts to CUDA 12.8 or newer
- One pod can do everything if its GPU has RT cores and >= 48 GB (RTX A6000, L40S); A100 80 GB also works.

Before terminating a pod: `python scripts/sync.py push`.

vast.ai instance (same image and scripts; template `cvpr-vail-pilot` on the VAIL-DTU team account, datacenter hosts only):
- Launch mode SSH (direct), 300 GB disk (fixed at creation), host CUDA 12.8 or newer; the image is public on GHCR
- `HF_TOKEN` as an account-level environment variable, never in the template (anyone with its URL can use it)
- On-start script writes the variables SSH sessions need to `/etc/environment` (read by `scripts/pod_env.sh`):
  `env | grep -E "^(RUNPOD_POD_ID|CONTAINER_ID|CONTAINER_API_KEY|HF_TOKEN|HF_HOME|BPY_PYTHON|UV_PYTHON_INSTALL_DIR|PYTHONUNBUFFERED|MONITOR_URL|MONITOR_TOKEN|VAST_CONTAINERLABEL)=" >> /etc/environment`
- At the end `finalize.py` (or `guard.py` after a crash) destroys the instance with its own `CONTAINER_API_KEY` once
  every file is verified on HF; if verification failed it only stops it, so the disk and its data are kept.

## Order of work

The pilot runs on one A100 80 GB pod (template `cvpr-vail-pilot`, 300 GB container disk), budget 12 A100-hours:

    cd /workspace && git clone https://github.com/Cypher193/xakfdghdfjk.git cvpr_vail && cd cvpr_vail
    setsid nohup bash scripts/pilot_pod.sh > /workspace/pilot.out 2>&1 < /dev/null &

`pilot_pod.sh` = `setup_runpod.sh` (weights, data from HF, background sync) + guardrail check + `scripts/run_pilot.py`,
which runs the approved plan in priority order inside the budget (B1 -> B5 -> real clips -> probes + steering ->
P7 inversion/noising -> B2-nl -> B-ratio -> P8 likelihood gravity -> P11 timing for Cosmos-14B and Cosmos 3 Nano),
skipping steps that no longer fit, while `pilot.score` tracks and scores clips alongside. Progress:
`data/logs/pilot.log`, `data/logs/budget.json`; results: `results/show.csv`, `results/p6_*.csv`, `results/p8_*.csv`.

CPU tests on tiny random-weight pipelines (no GPU): `python -m tests.test_generators`, `tests.test_steer`,
`tests.test_invert`, `tests.test_likelihood`, `tests.test_jobs`; metrics: `python -m pilot.metrics`.

## Generator screen

Seven candidates, seed 0, show-only for MAGI-1 and Self-Forcing (`capture: false` in meta). Tier A (main-slot
candidates: selfforcing, magi45, skyreels13, cosmos3_nano): B1 x 8, B5 x 3, real x 2. Tier B (scale arms: ltx2,
helios, skyreels14): B1 x 4, B5 x 3. Each tier runs on its own pod (or `all` on one):

    setsid nohup bash scripts/screen_pod.sh A 10 > /workspace/screen.out 2>&1 < /dev/null &      # tier, budget (h)

Setup, guardrail check, external venvs if the tier needs them, `run_pilot.py --plan screen --tier A`, then
`finalize.py` pushes, verifies on HF and stops the pod. Logs, budget and table carry `_screen_{tier}`
(`data/logs/pilot_screen_A.log`, `results/show_screen_A.csv`). Every saved clip starts with the input window
(frame k is k/fps after the first input frame), whatever the model returns.

## Design notes found while building

- **Input windows differ per model.** Cosmos-Predict2.5 accepts at most 5 input frames (0.31 s at 16 fps); Wan-VACE
  gets 17 frames (1.06 s) and LTX 25 frames (1.04 s). Each model is compared with itself on identical input.
- **QuantiPhy val clips are 1.3–3.3 s long**, so they serve as the real-footage tracker check (9 kinematic
  questions on 3 clips) and as real inputs, not as a main scene source.
- **Physics-IQ:** only 4 of 10 candidate scenarios keep the object in view after the input window
  (ball-and-block-fall, ball-ramp, balls-collide, mirror-ball-fall); centre views are used.
- **Bounce fits use one model fit** (start height, start velocity, g, restitution) over the whole track,
  because separate per-flight parabolas fail at 16 fps.
- Probe scores have a high floor (predicting the mean already scores ~0.68 when most g = 9.81), so read
  probe results only relative to the mean, VAE and control-task baselines.

Not yet written: random-init probe baseline, automatic gates report (G1-G10 are read from results/ by hand for now).

## Run monitor and vast.ai launcher

`docs/vast_plan.md` (full run on vast.ai: live quote `scripts/vast_quote.py`, waves, shards) and `docs/monitor_design.md`
(phone-reachable run monitor). `scripts/vast_launch.py monitor` creates the monitor instance; every pod's `guard.py`
heartbeats to `MONITOR_URL`; `scripts/monitor.py` serves the page and reaps dead instances (`--selftest` checks it).
