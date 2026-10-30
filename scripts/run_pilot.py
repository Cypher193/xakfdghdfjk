"""Run the approved 12 A100-hour pilot in priority order, inside a time budget.

    python scripts/run_pilot.py --budget-h 11.5        # after setup_runpod.sh (setup time is outside the budget)
    python scripts/run_pilot.py --plan screen --tier A --budget-h 10    # generator screen (scripts/screen_pod.sh)
    python scripts/run_pilot.py --plan ltx --job gen1 --budget-h 6      # LTX-2B full plan job (scripts/ltx_pod.sh)

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


def n_scenes(exp):
    """Scene count of an experiment from the data files, without importing pilot (guard.py and sync.py import this
    module from scripts/, where pilot is not on sys.path, and before the data is pulled: then 7, the pilot's default)."""
    root = Path(__file__).resolve().parents[1]
    scenes = yaml.safe_load(open(root / "configs/pilot.yaml"))["experiments"][exp]["scenes"]
    if scenes == "real":
        m = root / "data/real/manifest.json"
        return len(json.loads(m.read_text(encoding="utf-8"))) if m.exists() else 7
    return len(list((root / f"data/scenes/{scenes}").glob("*.json"))) or 7


def gen(model, exp, limit, seeds, ext=None, shard=None, ids=None):
    """ext: external repo (scripts/ext/setup_ext.sh): export job files, then run its runner in its own venv."""
    n = (len(ids) if ids else limit or n_scenes(exp)) * len(seeds)
    c = [PY, "-m", "pilot.generate", "--model", model, "--exp", exp, "--seeds", *map(str, seeds)] \
        + (["--limit", str(limit)] if limit else []) + (["--shard", shard] if shard else [])
    if ext:
        d = f"{os.environ.get('EXT_ROOT', '/workspace/ext')}/{ext}"
        runner = [f"{d}/venv/bin/python", f"scripts/ext/run_{ext}.py", "--repo", f"{d}/repo", "--exp", exp,
                  "--model", model]
        c = ["bash", "-c", f"{shlex.join(c + ['--export-jobs'])} && exec {shlex.join(runner)}"]
    return dict(name=f"{exp}:{model}", kind="gen", model=model, clips=n, out=[f"runs/{exp}/{model}"], cmd=c,
                **({"ids": ids, "seeds": seeds} if ids else {}))


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


# LTX-2B "full" plan: one job per pod (scripts/ltx_pod.sh). Run dirs each job pushes (sync.py imports them).
C9_SHARDS = 4                                                # phase 3: C9 (LEACE) over 4 pods
SHARDS = 7                                                   # phase 2 (2026-10-25): kmain 100 scenes over 7 pods
LTX_RUNS = {"gen1": ["runs/b1/ltx2b", "runs/b1real/ltx2b"], "gen2": ["runs/b2nl/ltx2b", "runs/b3/ltx2b"],
            "gen3": ["runs/b5/ltx2b"], "gen4": ["runs/bratio/ltx2b", "runs/p7/ltx2b_noise", "runs/p7/ltx2b_inv"],
            "probe": ["runs/steer/ltx2b"], "probe2": [], "score2": [], "bratio2": ["runs/bratio2/ltx2b"], "stats2": [],
            **{f"s{k}of{SHARDS}": [f"runs/{e}/ltx2b" for e in ("b1", "b2nl", "b3")] for k in range(1, SHARDS + 1)},
            **{f"c9s{k}of{C9_SHARDS}": ["runs/c9/ltx2b"] for k in range(1, C9_SHARDS + 1)},
            **{f"c9{x}s{k}of{C9_SHARDS}": [f"runs/c9{x}/ltx2b"] for x in "bc" for k in range(1, C9_SHARDS + 1)},
            **{f"c9cchain-{c}": [f"runs/c9c_chain/ltx2b/{c}"] for c in ("leace", "rand")}}
LTX_PULL_EXTRA = ["runs/*/ltx2b", "runs/p7/ltx2b_*"]         # the probe job needs every gen job's clips


def _probe_steps(model="ltx2b"):
    return [cmd(f"probe:{t}:{'+'.join(e)}:{pool}",
                ["-m", "pilot.probe", "--model", model, "--exps", *e, "--target", t, "--pool", pool], 300)
            for e, t in ((["b1"], "hspeed"), (["b1", "b5"], "g"), (["b1", "b5"], "e")) for pool in ("obj", "all")]


def _ltx_jobs():
    jobs = {"gen1": [gen("ltx2b", "b1", None, [0, 1, 2]), gen("ltx2b", "b1real", None, [0, 1])],
            "gen2": [gen("ltx2b", "b2nl", None, [0, 1, 2]), gen("ltx2b", "b3", None, [0, 1, 2])],
            "gen3": [gen("ltx2b", "b5", None, [0, 1, 2, 3, 4])],
            "gen4": [gen("ltx2b", "bratio", None, [0, 1, 2]),
                     dict(name="p7:ltx2b", kind="gen", model="p7_ltx2b", clips=30, out=LTX_RUNS["gen4"][1:],
                          cmd=[PY, "-m", "pilot.invert", "--model", "ltx2b", "--scenes", "30"])]}
    probe = _probe_steps()
    probe += [cmd(f"probe:{m}", ["-m", "pilot.probe", "--model", m, "--exps", "p7", "--target", "hspeed"], 300)
              for m in ("ltx2b_noise", "ltx2b_inv")]
    probe.append(dict(name="steer:ltx2b", kind="gen", model="ltx2b", clips=120, out=["runs/steer/ltx2b"],
                      cmd=[PY, "-m", "pilot.steer", "--model", "ltx2b", "--scenes", "30", "--alphas",
                           "-2", "-1", "1", "2"]))              # needs results/p6_ltx2b_hspeed_obj.csv: after the probes
    jobs["probe"] = probe
    # bigger-ball B-ratio (2026-10-26): render the input windows on the pod, then generate; the 10 s/clip score is appended below
    jobs["bratio2"] = [dict(name="render-in", kind="cmd", est_s=900, cmd=[
                           "bash", "-c", "${BPY_PYTHON:-/opt/bpy/bin/python} -m pilot.render data/scenes/kratio2 "
                           "--out data/renders/kratio2 --gpu --fps 48 --seconds 1.125 --suffix _input"]),
                       gen("ltx2b", "bratio2", None, [0, 1, 2])]
    for j, steps in jobs.items():       # p7 clips live in model dirs ltx2b_noise / ltx2b_inv, so gen4 scores those too
        models = ["ltx2b", "ltx2b_noise", "ltx2b_inv"] if j == "gen4" else ["ltx2b"]
        steps.append(cmd(f"score:{j}", ["-m", "pilot.score", "--models", *models, "--table", f"results/show_ltx_{j}.csv"],
                         10 * sum(s.get("clips", 0) for s in steps)))   # measured 3.5-7 s per clip on a 3090 (gen1/gen3)
    return jobs


def kmain_ids(k, n_shards, n):
    """Scene ids of shard k (pilot.shard on pilot.shard.scene_list order: cv_00, ca_00, cv_01, ...)."""
    return [f"kmain_{'cv' if p % 2 == 0 else 'ca'}_{p // 2:02d}" for p in range(k - 1, n, n_shards)]


def kgrav_ids(k, n_shards):
    """Scene ids of shard k of kgrav (pilot.shard on its sorted file names: kgrav_00_g0.5, kgrav_00_g1.0, ...)."""
    return [f"kgrav_{i:02d}_g{g}" for i in range(10) for g in (0.5, 1.0, 2.0)][k - 1::n_shards]


def _shard_jobs():
    """Phase 2: shard K/7 of the 100 kmain scenes renders (input window + full clip), generates b1, b2nl and b3 and scores
    its own scenes. The 30 pilot scenes are skipped at run time (their meta.json is pulled); probe2 reruns the probes."""
    n = n_scenes("b1")
    jobs = {}
    for k in range(1, SHARDS + 1):
        ids = kmain_ids(k, SHARDS, n)
        new = sum(p >= 30 for p in range(k - 1, n, SHARDS))    # the first 30 positions are the pilot scenes
        d, sh = f"data/shard_{k}", f"{k}/{SHARDS}"            # outside data/scenes: never pushed
        render = lambda name, flags, per: dict(
            name=name, kind="cmd", est_s=per * new, cmd=["bash", "-c",
            f"python -m pilot.shard --exp b1 --shard {sh} --write-shard {d} && "
            f"${{BPY_PYTHON:-/opt/bpy/bin/python}} -m pilot.render {d} --out data/renders/kmain --gpu {flags}"])
        steps = [render("render-in", "--fps 48 --seconds 1.125 --suffix _input", 90)]
        steps += [gen("ltx2b", e, None, [0, 1, 2], shard=sh, ids=ids) for e in ("b1", "b2nl", "b3")]
        steps.append(render("render-full", "", 180))
        clips = sum(s.get("clips", 0) for s in steps)
        steps.append(cmd(f"score:s{k}of{SHARDS}", ["-m", "pilot.score", "--models", "ltx2b", "--table",
                                                   f"results/show_ltx_s{k}of{SHARDS}.csv"], 10 * clips))
        jobs[f"s{k}of{SHARDS}"] = steps
    jobs["probe2"] = _probe_steps()
    # clips generated but never scored: s6/s7's first host died before their score step (sync.py picks the clips)
    jobs["score2"] = [cmd("score:score2", ["-m", "pilot.score", "--models", "ltx2b", "--table",
                                           "results/show_ltx_score2.csv"], 10 * 150)]
    return jobs


C9C_GROUPS, C9C_FIT, C9C_PASS_S = 5, 100, 55   # C9c chain: groups of blocks 7-27, fit scenes (seed 0), s per acts-only pass


def _c9_jobs():
    """Phase 3, C9 (docs/phase3.md): shard K/4 of the kmain scenes, seed 0, cond leace + rand = 2 clips per scene.
    C9b: the same with one eraser per block 7-27. C9c: the chains c9cchain-leace / -rand (two pods at once), then the
    shards, which load both finished chains (pilot.leace stops if a chain.json is missing)."""
    jobs = {f"c9cchain-{c}": [dict(name=f"c9cchain:{c}", kind="cmd", model="ltx2b",    # model: setup downloads LTX
                                   # group 0 is the B1 seed-0 acts (no pass); fits: ~15-25 s per block on the laptop
                                   est_s=LOAD_S + (C9C_GROUPS - 1) * C9C_FIT * C9C_PASS_S + 60 * len(range(7, 28)),
                                   cmd=[PY, "-m", "pilot.leace", "--model", "ltx2b", "--chain", c,
                                        "--groups", str(C9C_GROUPS), "--fit-scenes", str(C9C_FIT)])]
            for c in ("leace", "rand")}
    for exp, extra in (("c9", []), ("c9b", ["--blocks", "7-27", "--exp", "c9b"]), ("c9c", ["--exp", "c9c", "--use-groups", "3"])):
        for k in range(1, C9_SHARDS + 1):
            ids = [f"{s}_{c}" for s in kmain_ids(k, C9_SHARDS, n_scenes("b1")) for c in ("leace", "rand")]
            clips, name = len(ids), f"{exp}s{k}of{C9_SHARDS}"
            jobs[name] = [    # ids: the done-earlier check counts only this shard's clips (the pod pulls every shard's meta)
                # per_s 90: ~50-70 s/clip on a 3090 plus the C9b eraser fit (21 blocks: 15-60 min of CPU per pod)
                dict(name=f"{exp}:ltx2b", kind="gen", model="ltx2b", clips=clips, out=[f"runs/{exp}/ltx2b"], per_s=90,
                     ids=ids, seeds=[0],
                     cmd=[PY, "-m", "pilot.leace", "--model", "ltx2b", "--shard", f"{k}/{C9_SHARDS}", *extra]),
                cmd(f"score:{name}", ["-m", "pilot.score", "--models", "ltx2b", "--table", f"results/show_ltx_{name}.csv"],
                    10 * clips)]
    return jobs


MODEL_SHARDS = {"cosmos2b": 5, "vace13": 5}      # odd: kmain interleaves cv/ca, so every shard gets both families


def _model_jobs(model, n):
    """Core plan of one model over n pods, seed 0: shard K/n of B1 (kmain) and B5 (kgrav), shard 1 also b1real, then the
    shard scores its own clips. {model}probe probes the pushed acts."""
    jobs = {}
    for k in range(1, n + 1):
        sh, name = f"{k}/{n}", f"{model}s{k}of{n}"
        steps = [gen(model, "b1", None, [0], shard=sh, ids=kmain_ids(k, n, n_scenes("b1"))),
                 gen(model, "b5", None, [0], shard=sh, ids=kgrav_ids(k, n))] + \
                ([gen(model, "b1real", None, [0])] if k == 1 else [])
        steps.append(cmd(f"score:{name}", ["-m", "pilot.score", "--models", model, "--table",   # _ltx_{job}: sync.py's own files
                                           f"results/show_ltx_{name}.csv"], 10 * sum(s["clips"] for s in steps)))
        jobs[name] = steps
    jobs[f"{model}probe"] = _probe_steps(model)
    return jobs


# Round 2 (2026-10-30): B1 + B5 seeds 1-2 (G8 power), B2-nl seed 0 (H2). Cosmos B5 to 8 seeds (G4) left out: budget
ROUND2 = {"cosmos2b": (7, [1, 2]), "vace13": (7, [1, 2])}     # model: (pods, B5 seeds); 7 odd as above


def _round2_jobs(model, n, b5_seeds):
    jobs = {}
    for k in range(1, n + 1):
        sh, name, ids = f"{k}/{n}", f"{model}r{k}of{n}", kmain_ids(k, n, n_scenes("b1"))
        steps = [gen(model, "b1", None, [1, 2], shard=sh, ids=ids), gen(model, "b2nl", None, [0], shard=sh, ids=ids),
                 gen(model, "b5", None, b5_seeds, shard=sh, ids=kgrav_ids(k, n))]
        steps.append(cmd(f"score:{name}", ["-m", "pilot.score", "--models", model, "--table",
                                           f"results/show_ltx_{name}.csv"], 10 * sum(s["clips"] for s in steps)))
        jobs[name] = steps
    return jobs


MODEL_JOBS = {j: s for m, n in MODEL_SHARDS.items() for j, s in _model_jobs(m, n).items()}   # sync.py: table-driven
MODEL_JOBS |= {j: s for m, (n, b5) in ROUND2.items() for j, s in _round2_jobs(m, n, b5).items()}
# C7 steering on Cosmos-2B (G6 had only 6 pilot clips): alpha 0 is the B1 seed-0 clip
STEER_JOBS = {"cosmos2bsteer": [
    dict(name="steer:cosmos2b", kind="gen", model="cosmos2b", clips=40, out=["runs/steer/cosmos2b"],
         cmd=[PY, "-m", "pilot.steer", "--model", "cosmos2b", "--scenes", "10", "--alphas", "-2", "-1", "1", "2"]),
    cmd("score:cosmos2bsteer", ["-m", "pilot.score", "--models", "cosmos2b", "--table",
                                "results/show_ltx_cosmos2bsteer.csv"], 10 * 40)]}
STATS_MODELS = ["ltx2b", "cosmos2b", "vace13"]
# docs/phase3.md Deferred 2: the nested-CV know-show gap per model (CPU); G1a on the full renders runs in job rescore
STATS2 = [cmd(f"nested:{m}", ["-m", "pilot.analysis", "nested", "--model", m], s)
          for m, s in zip(STATS_MODELS, (2700, 1500, 1800))]     # ~2.5x a 16-thread laptop extrapolation (18, 8, 11 min)
# G1b fail action (configs/deviations.md D3): every B1 / B5 / b1real clip of the three models, plus the B-ratio set,
# re-scored with the CoTracker3 readout into score_v2.json; P0 (G1a) on the 100 full kmain renders with both trackers
RESCORE_RUNS = [f"runs/{e}/{m}" for e in ("b1", "b5", "b1real") for m in ("ltx2b", "cosmos2b", "vace13")] + \
    ["runs/bratio2/ltx2b"]
_p0 = lambda tr: cmd(f"p0:{tr}", ["-m", "pilot.track", "validate", "data/renders/kmain", "--scenes", "data/scenes/kmain",
                                  "--tracker", tr, "--out", f"results/p0_renders_{tr}_ltx_rescore.csv"], 100 * 20)
RESCORE_JOBS = {"rescore": [   # 767 clips at ~15 s (SAM2 3.5-7 s/clip on a 3090 + CoTracker3); the new mode's P0 first
    _p0("cotracker3"),
    cmd("score:rescore", ["-m", "pilot.score", "--tracker", "cotracker3", "--models", "ltx2b", "cosmos2b", "vace13",
                          "--table", "results/show_v2_ltx_rescore.csv", "--faults", "results/faults_ltx_rescore.csv"],
        767 * 15),
    _p0("sam2")]}
LTX_RUNS["rescore"] = []                                    # pull / push: sync.py's rescore branch
LTX_JOBS = _ltx_jobs() | _shard_jobs() | _c9_jobs() | MODEL_JOBS | STEER_JOBS | {"stats2": STATS2} | RESCORE_JOBS
LTX_RUNS |= {j: list(dict.fromkeys(o for s in steps for o in s.get("out", [])))
             for j, steps in (MODEL_JOBS | STEER_JOBS).items()}


def job_repos(job):
    """HF repos a job's pod downloads (setup_runpod.sh): the models its steps generate with, plus SAM2 for scoring."""
    models = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/pilot.yaml"))["models"]
    gen_models = dict.fromkeys(s["model"] for s in LTX_JOBS[job] if s.get("model") in models)   # not p7_ltx2b
    return [(models[m]["repo"], models[m].get("revision")) for m in gen_models] + [("facebook/sam2.1-hiera-large", None)]


def tag(plan, tier):
    """tier: the screen tier, or the job name for plan ltx."""
    return "" if plan == "main" else f"_ltx_{tier}" if plan == "ltx" else f"_screen_{tier}"


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
    per = step.get("per_s") or measured(step["model"]) or EST[step["model"]]
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
    ap.add_argument("--plan", choices=["main", "screen", "ltx"], default="main")
    ap.add_argument("--job", choices=list(LTX_JOBS), help="plan ltx: which job (scripts/ltx_pod.sh)")
    ap.add_argument("--tier", choices=["A", "B", "C", "all"], default="all", help="screen: A, B or both")
    ap.add_argument("--skip", nargs="*", default=[], help="screen: models to leave out")
    a = ap.parse_args()
    if not (os.environ.get("RUNPOD_POD_ID") or os.environ.get("CONTAINER_ID")):   # pushes to the shared HF dataset:
        sys.exit("run on a RunPod pod or vast.ai instance (RUNPOD_POD_ID / CONTAINER_ID unset)")   # never from a laptop
    global TAG
    if a.plan == "ltx" and not a.job:
        sys.exit("--plan ltx needs --job")
    TAG = tag(a.plan, a.job if a.plan == "ltx" else a.tier)
    plan = PLAN if a.plan == "main" else LTX_JOBS[a.job] if a.plan == "ltx" else screen_plan(a.tier, a.skip)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    start_file = Path(f"data/logs/pilot_start{TAG}")
    if not start_file.exists():
        start_file.write_text(str(time.time()))
    start = float(start_file.read_text())
    done_flag = Path(f"data/logs/pilot_gen_done{TAG}")
    done_flag.unlink(missing_ok=True)
    scorer = None if a.plan == "ltx" else subprocess.Popen(   # ltx: LTX + SAM2 do not fit a 24 GB card; the job's last step scores
        [PY, "-m", "pilot.score", "--loop", "300", "--until-file", str(done_flag), *score_args(a.plan, a.tier)],
        stdout=open(f"data/logs/score{TAG}.log", "a"), stderr=subprocess.STDOUT)
    state = dict(budget_h=a.budget_h, clips_total=sum(s.get("clips", 0) for s in plan), steps={})
    log(f"plan: {len(plan)} steps, {state['clips_total']} clips", state)
    for step in plan:
        pat = [f"{i}_s{x}/meta.json" for i in step["ids"] for x in step["seeds"]] if step.get("ids") else ["*/meta.json"]
        if step.get("out") and all(sum(len(list(Path(o).glob(g))) for g in pat) >= step["clips"] for o in step["out"]):
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
    if scorer:
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
