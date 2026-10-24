"""Run the approved 12 A100-hour pilot in priority order, inside a time budget.

    python scripts/run_pilot.py --budget-h 11.5        # after setup_runpod.sh (setup time is outside the budget)
    python scripts/run_pilot.py --plan screen --tier A --budget-h 10    # generator screen (scripts/screen_pod.sh)

Steps run in order; before each one the remaining budget is checked against its estimate (per-clip times are
re-estimated from finished clips as they come in), and a step that does not fit is skipped and logged.
Generation is resumable: finished clips are skipped, so re-running continues where it stopped.
A scorer runs alongside (pilot.score --loop) and data is pushed to HF throughout (setup_runpod.sh started it).
Log: data/logs/pilot.log, budget state: data/logs/budget.json (screen: suffixed _screen_{tier}, so two pods never
write the same file).
"""
import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

PY = sys.executable
# A100 seconds per clip, from the L40S smoke test / 1.4; replaced by measured medians once clips exist
EST = {"vace13": 284, "cosmos2b": 240, "ltx2b": 27, "cosmos14b": 700, "p7_ltx2b": 6, "p7_wan22_5b": 45,
       # screen: guesses (A100-equivalent), not measured
       "skyreels13": 360, "cosmos3_nano": 960, "magi45": 300, "selfforcing": 30, "ltx2": 300, "helios": 600,
       "skyreels14": 1500, "vace14": 900}
LOAD_S = 150                                       # model load per step


def gen(model, exp, limit, seeds, ext=None):
    """ext: external repo (scripts/ext/setup_ext.sh): export job files, then run its runner in its own venv."""
    n = (limit or 7) * len(seeds)
    c = [PY, "-m", "pilot.generate", "--model", model, "--exp", exp, "--seeds", *map(str, seeds)] \
        + (["--limit", str(limit)] if limit else [])
    if ext:
        d = f"{os.environ.get('EXT_ROOT', '/workspace/ext')}/{ext}"
        runner = [f"{d}/venv/bin/python", f"scripts/ext/run_{ext}.py", "--repo", f"{d}/repo", "--exp", exp,
                  "--model", model]
        c = ["bash", "-c", f"{shlex.join(c + ['--export-jobs'])} && exec {shlex.join(runner)}"]
    return dict(name=f"{exp}:{model}", kind="gen", model=model, clips=n, out=[f"runs/{exp}/{model}"], cmd=c)


def cmd(name, args, est_s):
    return dict(name=name, kind="cmd", est_s=est_s, cmd=[PY, *args])


PLAN = [  # priority order: the gates that matter most come first (approved plan, 2026-10-17)
    gen("vace13", "b1", 20, [0]), gen("cosmos2b", "b1", 20, [0]), gen("ltx2b", "b1", 30, [0, 1]),
    gen("vace13", "b5", 15, [0]), gen("cosmos2b", "b5", 15, [0]), gen("ltx2b", "b5", 30, [0, 1]),
    gen("vace13", "b1real", None, [0]), gen("cosmos2b", "b1real", None, [0]), gen("ltx2b", "b1real", None, [0]),
    cmd("probe-hspeed:ltx2b", ["-m", "pilot.probe", "--model", "ltx2b", "--exps", "b1", "--target", "hspeed"], 300),
    cmd("probe-hspeed:cosmos2b", ["-m", "pilot.probe", "--model", "cosmos2b", "--exps", "b1", "--target", "hspeed"], 300),
    dict(name="steer:ltx2b", kind="gen", model="ltx2b", clips=40, out=["runs/steer/ltx2b"],
         cmd=[PY, "-m", "pilot.steer", "--model", "ltx2b", "--scenes", "10", "--alphas", "-2", "-1", "1", "2"]),
    dict(name="steer:cosmos2b", kind="gen", model="cosmos2b", clips=6, out=["runs/steer/cosmos2b"],
         cmd=[PY, "-m", "pilot.steer", "--model", "cosmos2b", "--scenes", "3", "--alphas", "-2", "2"]),
    dict(name="p7:ltx2b", kind="gen", model="p7_ltx2b", clips=30, out=["runs/p7/ltx2b_noise", "runs/p7/ltx2b_inv"],
         cmd=[PY, "-m", "pilot.invert", "--model", "ltx2b"]),
    dict(name="p7:wan22_5b", kind="gen", model="p7_wan22_5b", clips=30, out=["runs/p7/wan22_5b_noise", "runs/p7/wan22_5b_inv"],
         cmd=[PY, "-m", "pilot.invert", "--model", "wan22_5b"]),
    # reordered 2026-10-18 (user): timing and P8 before B2-nl / B-ratio, which are trimmed to fit the budget
    cmd("p11:timing", ["-c", "import subprocess,sys; sys.exit(subprocess.call(['bash','scripts/time_extra.sh']))"], 1800),
    cmd("p8:cosmos2b", ["-m", "pilot.likelihood", "--scenes", "5"], 900),
    gen("vace13", "b2nl", 10, [0]), gen("cosmos2b", "b2nl", 5, [0]), gen("ltx2b", "b2nl", 30, [0]),
    gen("vace13", "bratio", 3, [0]), gen("cosmos2b", "bratio", 3, [0]), gen("ltx2b", "bratio", 10, [0]),
]
# Screen of candidate generators, seed 0. Tier A: main-slot candidates, tier B: scale arms; cheapest first (EST).
# The tiers can run on two pods at once: their run dirs, tables and logs are disjoint.
TIERS = {"A": ["selfforcing", "magi45", "skyreels13", "cosmos3_nano"], "B": ["ltx2", "helios", "skyreels14"],
         "C": ["ltx2", "helios", "cosmos14b", "vace14", "skyreels14"]}      # C: 14B+ models on one A100-80GB
SCREEN_EXPS = {"A": [("b1", 8), ("b5", 3), ("b1real", 2)], "B": [("b1", 4), ("b5", 3)],
               "C": [("b1", 8), ("b1real", 2)]}   # no b5: every model broke on the gravity scenes (screen A/B)


def screen_models(tier):
    return list(dict.fromkeys(m for t in TIERS.values() for m in t)) if tier == "all" else TIERS[tier]


def screen_ext(tier):
    """External repos the tier needs (scripts/ext/setup_ext.sh arguments)."""
    models = yaml.safe_load(open("configs/pilot.yaml"))["models"]
    return sorted({models[m]["ext"] for m in screen_models(tier) if models[m].get("ext")})


def free_hf(repo):
    """Delete a model from the HF cache. huggingface_hub 1.x keeps large files in a shared hub/blobs store that the
    repo folder only links to, so blobs no remaining link points at are deleted too."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    shutil.rmtree(hub / f"models--{repo.replace('/', '--')}", ignore_errors=True)
    keep = {p.resolve() for p in hub.rglob("*") if p.is_symlink()}
    for b in (hub / "blobs").rglob("*") if (hub / "blobs").is_dir() else []:
        if b.is_file() and not b.name.startswith(".") and b.resolve() not in keep:
            b.unlink()


def screen_plan(tier, skip=()):
    models = yaml.safe_load(open("configs/pilot.yaml"))["models"]
    plan = []
    for m in [m for m in screen_models(tier) if m not in skip]:
        ext = models[m].get("ext")
        exps = SCREEN_EXPS[tier] if tier in SCREEN_EXPS else SCREEN_EXPS[next(t for t in TIERS if m in TIERS[t])]
        if m == "helios" and any(e == "b5" for e, _ in exps):                            # its 1.375 s window is longer than the 1.125 s gravity inputs
            render = "for g in 0.5 1.0 2.0; do $BPY_PYTHON -m pilot.render data/scenes/kgrav/kgrav_00_g$g.json " \
                     "--out data/renders/kgrav --gpu --fps 48 --seconds 1.5 --suffix _input_long || exit 1; done"
            plan.append(cmd("render:kgrav_long", ["-c", f"import subprocess, sys; "
                                                        f"sys.exit(subprocess.call(['bash', '-c', {render!r}]))"], 600))
        plan += [gen(m, e, n, [0], ext) for e, n in exps]
        if not ext:                                  # keep the container disk under 300 GB: one model cached at a time
            plan.append(cmd(f"free:{m}", ["-c", f"import sys; sys.path.insert(0, 'scripts'); "
                                                f"from run_pilot import free_hf; free_hf({models[m]['repo']!r})"], 10))
    return plan


def tag(plan, tier):
    return "" if plan == "main" else f"_screen_{tier}"


def score_args(plan, tier):
    """Screen pods score only their own models into their own table (no file is written by both pods)."""
    return [] if plan == "main" else ["--models", *screen_models(tier), "--table", f"results/show{tag(plan, tier)}.csv"]


FINAL_PROBES = [(m, e, t) for m in ("vace13", "cosmos2b", "ltx2b")
                for e, t in ((["b1"], "hspeed"), (["b1", "b5"], "g"), (["b1", "b5"], "e"))] + \
               [(f"{m}_{k}", ["p7"], "hspeed") for m in ("ltx2b", "wan22_5b") for k in ("noise", "inv")]


def measured(model):
    """Median wall seconds per clip of finished runs for this model (None if none yet)."""
    pat = f"p7/{model.removeprefix('p7_')}_*" if model.startswith("p7_") else f"*/{model}"   # ltx2 is not ltx2b
    walls = [json.loads(f.read_text(encoding="utf-8")).get("wall_s") for f in Path("runs").glob(f"{pat}/*/meta.json")]
    walls = [w for w in walls if w]
    return statistics.median(walls) if len(walls) >= 3 else None


def estimate(step):
    if step["kind"] == "cmd":
        return step["est_s"]
    per = measured(step["model"]) or EST[step["model"]]
    return LOAD_S + step["clips"] * per


TAG = ""                                           # log-file suffix: "" (main plan) or _screen_{tier}


def log(msg, state=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"data/logs/pilot{TAG}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if state is not None:
        Path(f"data/logs/budget{TAG}.json").write_text(json.dumps(state, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-h", type=float, default=11.5)
    ap.add_argument("--reserve-h", type=float, default=0.4, help="kept for final scoring, probes and push")
    ap.add_argument("--plan", choices=["main", "screen"], default="main")
    ap.add_argument("--tier", choices=["A", "B", "C", "all"], default="all", help="screen: A, B or both")
    ap.add_argument("--skip", nargs="*", default=[], help="screen: models to leave out")
    a = ap.parse_args()
    if not (os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID")):   # pushes to the shared HF dataset:
        sys.exit("run on a RunPod pod or vast.ai instance (RUNPOD_POD_ID / CONTAINER_ID unset)")   # never from a laptop
    global TAG
    TAG = tag(a.plan, a.tier)
    plan = PLAN if a.plan == "main" else screen_plan(a.tier, a.skip)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    start_file = Path(f"data/logs/pilot_start{TAG}")
    if not start_file.exists():
        start_file.write_text(str(time.time()))
    start = float(start_file.read_text())
    done_flag = Path(f"data/logs/pilot_gen_done{TAG}")
    done_flag.unlink(missing_ok=True)
    scorer = subprocess.Popen([PY, "-m", "pilot.score", "--loop", "300", "--until-file", str(done_flag),
                               *score_args(a.plan, a.tier)],
                              stdout=open(f"data/logs/score{TAG}.log", "a"), stderr=subprocess.STDOUT)
    state = dict(budget_h=a.budget_h, steps={})
    for step in plan:
        if step.get("out") and all(len(list(Path(o).glob("*/meta.json"))) >= step["clips"] for o in step["out"]):
            state["steps"][step["name"]] = dict(status="done earlier")
            log(f"DONE-EARLIER {step['name']}", state)
            continue
        used = (time.time() - start) / 3600
        est = estimate(step) / 3600
        left = a.budget_h - a.reserve_h - used
        if est > left:
            state["steps"][step["name"]] = dict(status="skipped (budget)", est_h=round(est, 2), left_h=round(left, 2))
            log(f"SKIP {step['name']}: needs ~{est:.2f} h, {left:.2f} h left", state)
            continue
        log(f"START {step['name']} (est {est:.2f} h, used {used:.2f} h)")
        t0 = time.time()
        with open(f"data/logs/step{TAG}_{step['name'].replace(':', '_')}.log", "a") as out:
            try:
                rc = subprocess.run(step["cmd"], stdout=out, stderr=subprocess.STDOUT,
                                    timeout=max(est * 3600 * 2.5, 900)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        dt = (time.time() - t0) / 3600
        state["steps"][step["name"]] = dict(status="ok" if rc == 0 else f"failed ({rc})", hours=round(dt, 2),
                                            est_h=round(est, 2))
        state["used_h"] = round((time.time() - start) / 3600, 2)
        log(f"END {step['name']}: rc={rc}, {dt:.2f} h (used {state['used_h']} h)", state)
    done_flag.write_text("done")
    scorer.wait()
    for model, exps, target in FINAL_PROBES if a.plan == "main" else []:
        r = subprocess.run([PY, "-m", "pilot.probe", "--model", model, "--exps", *exps, "--target", target],
                           capture_output=True, text=True)
        log(f"probe {model} {target}: " + (r.stdout.strip().splitlines() or ["(no output)"])[-1][:200])
    subprocess.run([PY, "scripts/sync.py", "push"])
    state["used_h"] = round((time.time() - start) / 3600, 2)
    log(f"PILOT DONE, {state['used_h']} h used", state)


if __name__ == "__main__":
    main()
