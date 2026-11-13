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
    cmd("p8:cosmos2b", ["-m", "pilot.likelihood", "--set", "kcont", "--shard", "1/2", "--out", "results/p8_cosmos2b.csv"], 900),
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
                ["-m", "pilot.probe", "--model", model, "--exps", *e, "--target", t, "--pool", pool], 300 * len(e) ** 2)
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


def kgrav_ids(k, n_shards, bases=10):
    """Scene ids of shard k of kgrav (pilot.shard on its sorted file names: kgrav_00_g0.5, kgrav_00_g1.0, ...)."""
    return [f"kgrav_{i:02d}_g{g}" for i in range(bases) for g in (0.5, 1.0, 2.0)][k - 1::n_shards]


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


# Round 3 (2026-11-01, docs/phase3.md "Round 3"): the rest of the B runs on all three small models. One model per pod, every
# experiment spread over its pods. b5: the 30 old + 60 new kgrav scenes to 5 seeds, finished clips are skipped. per_s: 3090.
S3, S5 = [0, 1, 2], [0, 1, 2, 3, 4]
ROUND3 = {"cosmos2b": (20, 650, [("b2nl", None, [1, 2]), ("b2num", 60, S3), ("b3", None, S3), ("b5", None, S5),
                                 ("bratio2", None, S3), ("bl3", None, S3), ("bl3nl", None, S3)]),
          "vace13": (11, 900, [("b2nl", None, [1, 2]), ("b2num", 60, S3), ("bratio2", None, S3)]),
          "ltx2b": (2, 80, [("b2num", 60, S3), ("b5", None, S5), ("bratio2", None, S3), ("bl3", None, S3), ("bl3nl", None, S3)])}
KL3_IDS = [f"kl3_{f}_{i:02d}" for f in ("co", "ra") for i in range(15)]
RATIO2_IDS = [f"kratio2_{i:02d}" for i in range(40)]


def round3_ids(exp, k, n, limit):
    if exp in ("b1", "b2nl", "b2num", "b3"):
        return kmain_ids(k, n, limit or 100)
    return kgrav_ids(k, n, 30) if exp == "b5" else RATIO2_IDS[k - 1::n] if exp == "bratio2" else KL3_IDS[k - 1::n]


def _round3_jobs(model, n, per_s, blocks, prefix="g", trackers=("sam2",)):
    jobs = {}
    for k in range(1, n + 1):
        sh, name = f"{k}/{n}", f"{model}{prefix}{k}of{n}"
        steps = [gen(model, e, limit, seeds, shard=sh, ids=round3_ids(e, k, n, limit)) | dict(per_s=per_s)
                 for e, limit, seeds in blocks]
        n_clips = sum(s["clips"] for s in steps)
        steps += [cmd(f"score:{name}:{tr}", ["-m", "pilot.score", "--models", model, "--tracker", tr, "--table",
                                             f"results/show{'_v2' * (tr != 'sam2')}_ltx_{name}.csv"], 10 * n_clips)
                  for tr in trackers]
        jobs[name] = steps
    return jobs


MODEL_JOBS = {j: s for m, n in MODEL_SHARDS.items() for j, s in _model_jobs(m, n).items()}   # sync.py: table-driven
MODEL_JOBS |= {j: s for m, (n, b5) in ROUND2.items() for j, s in _round2_jobs(m, n, b5).items()}
MODEL_JOBS |= {j: s for m, (n, per_s, blocks) in ROUND3.items() for j, s in _round3_jobs(m, n, per_s, blocks).items()}
# Scale study (2026-11-05, docs/phase3.md "Scale study", H6): Cosmos-14B on 48 GB cards. `g`: B1 on the 100 kmain scenes and
# B5 on the 90 kgrav scenes, seeds 0-1 (100 scenes, not the proposal's 60: H6 is an equivalence test). `t`: B2-nl seed 0,
# launched after `g` only if the credit allows; B2-nl seed 1 is deferred to a later budget. per_s: L40S, from the 10
# timing clips (1040-1100 s, activations captured). Both trackers score, as the 2B clips were.
SCALE = {"g": (6, [("b1", None, [0, 1]), ("b5", None, [0, 1])]), "t": (3, [("b2nl", None, [0])])}
for _p, (_n, _blocks) in SCALE.items():
    MODEL_JOBS |= _round3_jobs("cosmos14b", _n, 1100, _blocks, _p, ("sam2", "cotracker3"))
# Cosmos 3 Nano (2026-11-06, H8; docs/phase3.md "Nano"), on the RunPod team account (scripts/runpod_launch.py): the 14B
# design. `g`: B1 100 kmain x 2 seeds + B5 90 kgrav x 2 seeds; `t`: B2-nl seed 0. per_s: A40 guess from the A6000 timing
# clips (248 s). `smoke`: 2 B1 clips, both trackers, then the pod must terminate itself (first RunPod pod of the account).
NANO = {"g": (8, [("b1", None, [0, 1]), ("b5", None, [0, 1])]), "t": (2, [("b2nl", None, [0])])}
for _p, (_n, _blocks) in NANO.items():
    MODEL_JOBS |= _round3_jobs("cosmos3_nano", _n, 270, _blocks, _p, ("sam2", "cotracker3"))
# 2026-11-06 (user, RunPod): the G4 fix, B5 seeds 5-7 on LTX and Cosmos-2B (configs/gates.yaml's fail rule: 8 seeds), and
# 14B B2-nl seed 1 (deferred 2026-11-05). per_s: A40 / A5000 guesses (3090: Cosmos-2B 619 s, LTX 69 s; 14B A40 ~1900 s).
EXTRA = {"ltx2b": (2, 80, "h", [("b5", None, [5, 6, 7])]), "cosmos2b": (6, 650, "h", [("b5", None, [5, 6, 7])]),
         "cosmos14b": (6, 1900, "u", [("b2nl", None, [1])]),
         "cosmos3_nano": (2, 270, "u", [("b2nl", None, [1])])}     # 2026-11-07 (user, vast): Nano B2-nl seed 1 (D10)
NARROW_PULL = set()                   # sync.py pulls only these jobs' own shard files
for _m, (_n, _ps, _p, _blocks) in EXTRA.items():
    _jobs = _round3_jobs(_m, _n, _ps, _blocks, _p, ("sam2", "cotracker3"))
    MODEL_JOBS |= _jobs
    NARROW_PULL |= set(_jobs)
# 2026-11-07 (user, RunPod): Nano B5 seeds 2-4 (cosmos3_nanovKof5) on its 90 kgrav scenes, A40
_jobs = _round3_jobs("cosmos3_nano", 5, 270, [("b5", None, [2, 3, 4])], "v", ("sam2", "cotracker3"))
MODEL_JOBS |= _jobs
NARROW_PULL |= set(_jobs)
# Replication (2026-11-07, user, RunPod; D12): kmain and kgrav drawn again with seed 7 as kmainf / kgravf
# (`python -m pilot.scenes --fresh 7`). Shard K/10 renders its input windows, then LTX-2B (seeds 0-1) and Cosmos-2B
# (seed 0) generate B1 (b1f) and B5 (b5f), and both trackers score. `freshprobe`: probes + nested gap on a CPU pod.
FRESH_N = 10
FRESH_IDS = {"kmainf": sorted(f"kmainf_{f}_{i:02d}" for f in ("ca", "cv") for i in range(50)),
             "kgravf": sorted(f"kgravf_{i:02d}_g{g}" for i in range(30) for g in (0.5, 1.0, 2.0))}
for _k in range(1, FRESH_N + 1):
    _sh, _name = f"{_k}/{FRESH_N}", f"fresh{_k}of{FRESH_N}"
    _ids = {e: FRESH_IDS[st][_k - 1::FRESH_N] for e, st in (("b1f", "kmainf"), ("b5f", "kgravf"))}
    _steps = [dict(name=f"render-in:{st}", kind="cmd", est_s=60 * len(_ids[e]), cmd=[
                  "bash", "-c", f"python -m pilot.shard --exp {e} --shard {_sh} --write-shard data/shard_{st} && "
                                f"${{BPY_PYTHON:-/opt/bpy/bin/python}} -m pilot.render data/shard_{st} --out data/renders/{st} --gpu --fps 48 --seconds 1.125 --suffix _input"])
              for e, st in (("b1f", "kmainf"), ("b5f", "kgravf"))]
    _steps += [gen(m, e, None, sd, shard=_sh, ids=_ids[e]) | dict(per_s=ps)
               for m, sd, ps in (("ltx2b", [0, 1], 80), ("cosmos2b", [0], 650)) for e in ("b1f", "b5f")]
    _n = sum(st.get("clips", 0) for st in _steps)
    _steps += [cmd(f"score:{_name}:{tr}", ["-m", "pilot.score", "--models", "ltx2b", "cosmos2b", "--tracker", tr, "--table",
                                           f"results/show{'_v2' * (tr != 'sam2')}_ltx_{_name}.csv"], 10 * _n)
               for tr in ("sam2", "cotracker3")]
    MODEL_JOBS[_name] = _steps
    NARROW_PULL.add(_name)
# H3 (2026-11-08, user, vast): probe gamma per block on every generator's B5 activations (CPU, no tracker)
GAMMA_MODELS = ("ltx2b", "cosmos2b", "vace13", "cosmos14b", "cosmos3_nano")
MODEL_JOBS["gammaprobe"] = [cmd(f"gamma:{m}", ["-m", "pilot.analysis", "gamma", "--no-fetch", "--model", m], 1800)
                            for m in GAMMA_MODELS]
MODEL_JOBS["freshprobe"] = [cmd(f"probe:{m}f:{t}:{'+'.join(e)}:{pool}",
                                ["-m", "pilot.probe", "--model", m, "--exps", *e, "--target", t, "--pool", pool,
                                 "--out", f"results/p6_{m}f_{t}_{pool}.csv"], 300 * len(e) ** 2)
                            for m in ("ltx2b", "cosmos2b")
                            for e, t in ((["b1f"], "hspeed"), (["b1f", "b5f"], "g"), (["b1f", "b5f"], "e")) for pool in ("obj", "all")] +     [cmd(f"nested:{m}f", ["-m", "pilot.analysis", "nested", "--v2", "--no-fetch", "--fresh", "--model", m], 3600)
     for m in ("ltx2b", "cosmos2b")]
MODEL_JOBS["cosmos3_nanosmoke"] = [
    gen("cosmos3_nano", "b1", 20, [0], shard="9/10", ids=kmain_ids(9, 10, 20)) | dict(per_s=270)] + [   # cv_04, cv_09
    cmd(f"score:cosmos3_nanosmoke:{tr}", ["-m", "pilot.score", "--models", "cosmos3_nano", "--tracker", tr, "--table",
                                          f"results/show{'_v2' * (tr != 'sam2')}_ltx_cosmos3_nanosmoke.csv"], 120)
    for tr in ("sam2", "cotracker3")]
MODEL_JOBS["cosmos3_nanoprobe"] = _probe_steps("cosmos3_nano") + [
    cmd("nested:cosmos3_nano", ["-m", "pilot.analysis", "nested", "--v2", "--no-fetch", "--model", "cosmos3_nano"], 3600)]
MODEL_JOBS["cosmos14bprobe"] = [s | dict(est_s=4 * s["est_s"]) for s in _probe_steps("cosmos14b")] + [   # 5120-d acts
    cmd("nested:cosmos14b", ["-m", "pilot.analysis", "nested", "--v2", "--no-fetch", "--model", "cosmos14b"], 3600)]
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
# round-2 clips (seeds 1-2) postdate rescore: rescore2 scores only the Cosmos / VACE clips that lack score_v2.json
RESCORE_JOBS["rescore2"] = [cmd("score:rescore2", ["-m", "pilot.score", "--tracker", "cotracker3", "--models", "cosmos2b",
                                                   "vace13", "--table", "results/show_v2_ltx_rescore2.csv",
                                                   "--faults", "results/faults_ltx_rescore2.csv"], 550 * 15)]
LTX_RUNS["rescore"] = LTX_RUNS["rescore2"] = []                                    # pull / push: sync.py's rescore branch
# Round 3 clips, plus steer (G6) and the one LTX B1 clip rescore missed: one pod per model, clips that lack score_v2.json
RESCORE3_RUNS = {m: [f"runs/{e}/{m}" for e in exps] for m, exps in (
    ("ltx2b", ("b1", "b2nl", "b2num", "b3", "b5", "bl3", "bl3nl", "bratio2", "steer")),
    ("cosmos2b", ("b2nl", "b2num", "b3", "b5", "bl3", "bl3nl", "bratio2", "steer")),
    ("vace13", ("b2nl", "b2num", "bratio2")))}
for _m, _n in (("ltx2b", 1471), ("cosmos2b", 1480), ("vace13", 600)):
    _j = f"{_m}rescore3"
    RESCORE_JOBS[_j] = [cmd(f"score:{_j}", ["-m", "pilot.score", "--tracker", "cotracker3", "--models", _m, "--table",
                                            f"results/show_v2_ltx_{_j}.csv", "--faults", f"results/faults_ltx_{_j}.csv"],
                            _n * 15)]
    LTX_RUNS[_j] = []
# Clips whose score.json / score_v2.json holds a readout exception (track.fit_bounce bounds, fixed 2026-11-03): both trackers
REDERR_DIRS = (Path(__file__).resolve().parents[1] / "configs/rederr.txt").read_text(encoding="utf-8").split()
RESCORE_JOBS["rederr"] = [cmd(f"score:{tr}", ["-m", "pilot.score", "--tracker", tr, "--table",
                                              f"results/show{'_v2' * (tr != 'sam2')}_ltx_rederr.csv"], len(REDERR_DIRS) * 15)
                          for tr in ("sam2", "cotracker3")]
LTX_RUNS["rederr"] = []
RESCORE_JOBS["vace13rescore3"].insert(0, cmd("p0:kl3", ["-m", "pilot.track", "validate", "data/renders/kl3full", "--scenes", "data/scenes/kl3",
                                                        "--tracker", "cotracker3", "--out", "results/p0_kl3_cotracker3_ltx_vace13rescore3.csv"],
                                             21 * 20))          # kl3 was only validated with SAM2
# Round 3 input windows on 3 pods; finished renders are pulled and skipped, so only the new scenes render. Pod 1 also
# renders 6 full kl3 clips and runs the tracker on them: the collide and ramp readouts against ground truth (G1a).
RENDER3_EXPS, RENDER3_PODS = {"b5": "kgrav", "bratio2": "kratio2", "bl3": "kl3"}, 3
BPY = "${BPY_PYTHON:-/opt/bpy/bin/python}"


def _render3_jobs():
    jobs = {}
    for k in range(1, RENDER3_PODS + 1):
        sh, name = f"{k}/{RENDER3_PODS}", f"render3s{k}of{RENDER3_PODS}"
        steps = [dict(name=f"render:{s}", kind="cmd", est_s=90 * {"kgrav": 60, "kratio2": 30, "kl3": 30}[s] // RENDER3_PODS, cmd=[
            "bash", "-c", f"python -m pilot.shard --exp {e} --shard {sh} --write-shard data/shard_{s} && "
                          f"{BPY} -m pilot.render data/shard_{s} --out data/renders/{s} --gpu --fps 48 --seconds 1.125 --suffix _input"])
                 for e, s in RENDER3_EXPS.items()]
        if k == 1:
            steps += [dict(name="render:kl3full", kind="cmd", est_s=6 * 180, cmd=[
                          "bash", "-c", "python -m pilot.shard --exp bl3 --shard 1/5 --write-shard data/shard_kl3full && "
                                        f"{BPY} -m pilot.render data/shard_kl3full --out data/renders/kl3full --gpu"]),
                      cmd("p0:kl3", ["-m", "pilot.track", "validate", "data/renders/kl3full", "--scenes", "data/scenes/kl3",
                                     "--tracker", "sam2", "--out", f"results/p0_kl3_ltx_{name}.csv"], 6 * 20)]
        jobs[name] = steps
    return jobs


RENDER3_JOBS = _render3_jobs()
# ramp readout check: all 15 ramps rendered in full, SAM2 on each (scripts/kl3_diag.py)
RENDER3_JOBS["kl3diag"] = [
    dict(name="render:ramps", kind="cmd", est_s=15 * 60, cmd=[
        "bash", "-c", "mkdir -p data/shard_ramps && cp data/scenes/kl3/kl3_ra_*.json data/shard_ramps/ && "
                      f"{BPY} -m pilot.render data/shard_ramps --out data/renders/kl3full --gpu"]),
    cmd("diag:kl3", ["scripts/kl3_diag.py"], 15 * 60)]
LTX_RUNS |= dict.fromkeys(RENDER3_JOBS, [])                                         # pull / push: sync.py's render3 branch
# Nested gap with the CoTracker3 readout (score_v2.json), on the round 3 B5 sets (450 clips for LTX and Cosmos)
STATS2V2 = [cmd(f"nested:{m}", ["-m", "pilot.analysis", "nested", "--v2", "--no-fetch", "--model", m], s)
            for m, s in zip(STATS_MODELS, (5400, 3600, 1800))]
LTX_RUNS["stats2v2"] = []
# C2 / C2-g (2026-11-03, docs/phase3.md "C2"): denoising loss of velocity-scaled copies of the kmain scenes
# (pilot.scenes.VEL_SETS) and of the kcont gravity continuations. Each pod renders its own scene bases, then scores them.
# model: (scene set, pods, render fps and seconds, s per variant at 32 passes on its card: Cosmos 4090, LTX 3090)
C2_PODS = {"cosmos2b": ("kvelc", 4, 16, 5.8125, 140), "ltx2b": ("kvell", 2, 24, 5.0417, 25)}
KVEL_BASES = sorted(f"{f}_{i:02d}" for f in ("ca", "cv") for i in range(50))     # pilot.likelihood.bases order
C2_BASES = {}                                                                     # job: {scene set: its bases}


def _c2_jobs():
    jobs = {}
    for model, (s, n, fps, sec, per_s) in C2_PODS.items():
        for k in range(1, n + 1):
            sh, name = f"{k}/{n}", f"c2{s[-1]}{k}of{n}"
            C2_BASES[name] = {s: [f"{s}_{b}" for b in KVEL_BASES[k - 1::n]]}
            steps = [_render(s, sh, len(C2_BASES[name][s]) * 5, fps, sec)]
            if model == "cosmos2b":                          # C2-g: the 10 kcont bases (rendered), 16 noise levels
                C2_BASES[name]["kcont"] = [f"kcont_{i:02d}" for i in range(10)][k - 1::n]
                steps.append(_likelihood(model, "kcont", sh, name, 16, len(C2_BASES[name]["kcont"]) * 7 * 2 * per_s))
            steps.append(_likelihood(model, s, sh, name, 8, len(C2_BASES[name][s]) * 5 * per_s))
            jobs[name] = steps
            LTX_RUNS[name] = []
    return jobs


def _render(s, sh, n_clips, fps, sec):
    return dict(name=f"render:{s}", kind="cmd", est_s=n_clips * 15, cmd=[
        "bash", "-c", f"python -m pilot.likelihood --set {s} --shard {sh} --write-shard data/shard_{s} && "
                      f"{BPY} -m pilot.render data/shard_{s} --out data/renders/{s} --gpu --fps {fps} --seconds {sec}"])


def _likelihood(model, s, sh, name, sigmas, est_s):
    exp = "c2g" if s.startswith("kcont") else "c2"
    out = f"results/{exp}_{s}_{model}_ltx_{name}.csv"
    return cmd(f"{exp}:{model}:{s}", ["-m", "pilot.likelihood", "--model", model, "--set", s,
                                      "--shard", sh, "--sigmas", str(sigmas), "--out", out],
               est_s) | dict(model=model)                   # model: job_repos downloads it


C2_JOBS = _c2_jobs()
# C2 follow-ups (2026-11-04, docs/phase3.md): VACE C2 on kvelv, C2-turn (same speed, turned) on kturnc / kturnl,
# LTX C2-g on kcontl. prefix: model, pods, s per variant at 32 passes (VACE ~3.9 s per pass on a 4090, from generation),
# [(scene set, noise levels, variants per base: kturn* 3 for cv and 2 for bounces, render fps, seconds)]
C3_PODS = {"c3v": ("vace13", 4, 140, [("kvelv", 8, 5, 16, 5.0625)]),
           "c3c": ("cosmos2b", 3, 140, [("kturnc", 8, 2.5, 16, 5.8125)]),
           "c3l": ("ltx2b", 2, 25, [("kcontl", 16, 7, 24, 5.0417), ("kturnl", 8, 2.5, 24, 5.0417)]),
           # 2026-11-06 (user, RunPod): Cosmos 3 Nano C2 on its own window (kveln), ~3.5 s per pass on an A40
           "c3n": ("cosmos3_nano", 4, 120, [("kveln", 8, 5, 24, 5.0417)]),
           # 2026-11-06 (user, RunPod): VACE C2-g and C2-turn on its window, ~275 s per copy on an A40 (2.2x a 4090)
           "c3w": ("vace13", 6, 275, [("kcontv", 16, 7, 16, 5.0625), ("kturnv", 8, 2.5, 16, 5.0625)]),
           # 2026-11-06 (user, vast credit): Nano C2-g and C2-turn on its window; ~90 s per copy measured on A40 (c3n)
           "c3m": ("cosmos3_nano", 3, 120, [("kcontn", 16, 7, 24, 5.0417), ("kturnn", 8, 2.5, 24, 5.0417)])}
C3_SETS = {"kcontl": [f"kcontl_{i:02d}" for i in range(10)], "kcontv": [f"kcontv_{i:02d}" for i in range(10)],
           "kcontn": [f"kcontn_{i:02d}" for i in range(10)]}     | {s: [f"{s}_{b}" for b in KVEL_BASES] for s in ("kvelv", "kturnc", "kturnl", "kveln", "kturnv", "kturnn")}


def _c3_jobs():
    jobs = {}
    for prefix, (model, n, per_s, sets) in C3_PODS.items():
        for k in range(1, n + 1):
            sh, name = f"{k}/{n}", f"{prefix}{k}of{n}"
            C2_BASES[name] = {s: C3_SETS[s][k - 1::n] for s, *_ in sets}
            jobs[name] = [_render(s, sh, len(C2_BASES[name][s]) * v, fps, sec) for s, _, v, fps, sec in sets]                 + [_likelihood(model, s, sh, name, sig, len(C2_BASES[name][s]) * v * per_s * sig / 8)
                   for s, sig, v, *_ in sets]
            LTX_RUNS[name] = []
    return jobs


C2_JOBS |= _c3_jobs()
# Cosmos-14B C2 (2026-11-06, user: 50 bases, RunPod): shards 1-6 of 12 of the 100 kvelc bases, on the renders Cosmos-2B
# used (same window and clip), so no render step. ~350 s per variant at 32 passes on an RTX PRO 6000 (B2-nl: 600 s a clip)
for _k in range(1, 7):
    _name = f"c3b{_k}of6"
    C2_BASES[_name] = {"kvelc": [f"kvelc_{b}" for b in KVEL_BASES][_k - 1::12]}
    C2_JOBS[_name] = [_likelihood("cosmos14b", "kvelc", f"{_k}/12", _name, 8, len(C2_BASES[_name]["kvelc"]) * 5 * 350)]
    LTX_RUNS[_name] = []
# 2026-11-07 (user, RunPod): Cosmos-14B C2-g on kcont (c3gKof5) and C2-turn on kturnc (c3tKof8), on the Cosmos-2B renders
# (same window and clip), so no render step. ~650 s per copy at 8 noise levels on an A40 (c3b, measured)
for _pre, _set, _n, _sig, _bases in (("c3g", "kcont", 5, 16, [f"kcont_{i:02d}" for i in range(10)]),
                                     ("c3t", "kturnc", 8, 8, [f"kturnc_{b}" for b in KVEL_BASES])):
    for _k in range(1, _n + 1):
        _name = f"{_pre}{_k}of{_n}"
        C2_BASES[_name] = {_set: _bases[_k - 1::_n]}
        _copies = len(C2_BASES[_name][_set]) * (7 if _set == "kcont" else 2.5)
        C2_JOBS[_name] = [_likelihood("cosmos14b", _set, f"{_k}/{_n}", _name, _sig, _copies * 650 * _sig / 8)]
        LTX_RUNS[_name] = []
# V-JEPA 2 (2026-11-05, docs/phase3.md "V-JEPA 2"): C1 acts on the B1 / B5 input windows, C2 / C2-g surprise on the
# rendered kvelv / kturnl / kcontl sets, probes, C3 gamma per layer; both sizes on one RTX 4090, existing renders only
VJEPA_MODELS = ("vjepa2l", "vjepa2g")
VJEPA_JOB = [cmd(f"acts:{m}:{e}", ["-m", "pilot.vjepa", "acts", "--model", m, "--exp", e], 900)
             for m in VJEPA_MODELS for e in ("b1", "b5")]     + [cmd(f"surprise:{m}:{s}", ["-m", "pilot.vjepa", "surprise", "--model", m, "--set", s,
                                 "--out", f"results/{'c2g' if s.startswith('kcont') else 'c2'}_{s}_{m}_ltx_vjepa2.csv"], 1800)
       for m in VJEPA_MODELS for s in ("kvelv", "kturnl", "kcontl")]     + [st for m in VJEPA_MODELS for st in _probe_steps(m)]     + [cmd(f"gamma:{m}", ["-m", "pilot.vjepa", "gamma", "--model", m, "--out", f"results/c3_{m}_gamma_ltx_vjepa2.csv"], 900)
       for m in VJEPA_MODELS]
C2_JOBS["vjepa2"] = VJEPA_JOB
# frame-stride check (2026-11-05): at 16-24 fps the reversed copy scored least surprising; pretraining samples sparsely
C2_JOBS["vjepa2s"] = [cmd(f"surprise:{m}:{s}:s{k}", ["-m", "pilot.vjepa", "surprise", "--model", m, "--set", s, "--stride", str(k),
                          "--out", f"results/{'c2g' if s.startswith('kcont') else 'c2'}_{s}_{m}_s{k}_ltx_vjepa2s.csv"], 1200)
                      for k in (2, 3, 4) for m in VJEPA_MODELS for s in ("kvelv", "kturnl", "kcontl")]
LTX_RUNS["vjepa2"] = [f"runs/{e}/{m}" for m in VJEPA_MODELS for e in ("b1", "b5")]
LTX_RUNS["vjepa2s"] = []
# Wrap-up (2026-11-05, user: on vast, not the laptop): pilot.wrapup (H1, H2, H5, H7, C4, B-ratio, rederr) and the gates
# report per model with both trackers, from score files and earlier results only (no videos, no activations)
WRAPUP_EXPS = ["b1", "b2nl", "b2num", "b3", "bl3", "bl3nl", "b5", "bratio2", "steer", "b1real"]
C2_JOBS["wrapup"] = [cmd("wrapup", ["-m", "pilot.wrapup"], 1800)] +     [cmd(f"gates:{m}{':v2' * v2}", ["-m", "pilot.analysis", "gates", "--no-fetch", "--model", m] + ["--v2"] * v2, 600)
     for m in STATS_MODELS for v2 in (True, False)]
LTX_RUNS["wrapup"] = []
# Red-team fix (2026-11-11, user approved; scratchpad fix_plan.md v2): re-capture at 25% of denoising with
# trajectory-free pools (pilot.recapture), seed 0, acts_only. Pods: (model, [(set, shard)], random-init too, s/clip).
RC_PODS = {"rcltx": ("ltx2b", [("kmain", None), ("kgrav", None)], True, 25),
           "rccos1of2": ("cosmos2b", [("kmain", "1/2"), ("kgrav", "1/2")], False, 95),
           "rccos2of2": ("cosmos2b", [("kmain", "2/2"), ("kgrav", "2/2")], False, 95),
           "rcvace": ("vace13", [("kmain", None)], False, 115),
           "rcnano": ("cosmos3_nano", [("kmain", None), ("kgrav", None)], False, 75)}
# Cosmos-14B fixed-box re-capture (plan docs/redteam/cos14b_plan.md): 4 shards; ~335 s/clip on an A40 is extrapolated
# from Cosmos-2B (96 s) x 3.5 (14B/2B full-generation time on an A100); the smoke shard measures it first.
RC_PODS |= {f"rccos14b{k}of4": ("cosmos14b", [("kmain", f"{k}/4"), ("kgrav", f"{k}/4")], False, 335) for k in (1, 2, 3, 4)}
RC_PODS["rccos14bsmoke"] = ("cosmos14b", [("kmain", "1/100")], False, 335)   # one clip: time and VRAM
RC_N = {"kmain": 100, "kgrav": 90}


def _rc_jobs():
    jobs = {}
    for job, (m, parts, rand, per) in RC_PODS.items():
        steps = [dict(name=f"rc:{m}{'_rand' * r}:{s}", kind="cmd", model=m, est_s=LOAD_S + -(-RC_N[s] // int(sh.split("/")[1] if sh else 1)) * per,
                      cmd=[PY, "-m", "pilot.recapture", "gen", "--model", m, "--set", s]
                      + (["--shard", sh] if sh else []) + (["--random-init"] if r else []))
                 for s, sh, r in [(s, sh, False) for s, sh in parts] + ([("kmain", None, True)] if rand else [])]
        jobs[job] = steps
        LTX_RUNS[job] = [f"runs/rc/{m}"] + ([f"runs/rc/{m}_rand"] if rand else [])
    rc_models = ["ltx2b", "ltx2b_rand", "cosmos2b", "vace13", "cosmos3_nano"]
    jobs["rcprobe"] = [cmd(f"rccheck:{m}:{s}", ["-m", "pilot.recapture", "check", "--model", m, "--set", s], 300)
                       for m in rc_models if not m.endswith("_rand") for s in ("kmain", "kgrav") if m != "vace13" or s == "kmain"] \
        + [cmd(f"rcprobe:{m}:{s}:{t}", ["-m", "pilot.recapture", "probe", "--model", m, "--set", s, "--target", t], 1800)
           for m in rc_models for s, t in (("kmain", "hspeed"), ("kgrav", "g"))
           if not (s == "kgrav" and m in ("vace13", "ltx2b_rand"))] \
        + [cmd("rcsummary", ["-m", "pilot.recapture", "summary"], 60)]
    # 14B: its own family of two tests (pre-registered), so the 8-test Holm family above is unchanged
    jobs["rcprobe14b"] = [cmd(f"rccheck:cosmos14b:{s}", ["-m", "pilot.recapture", "check", "--model", "cosmos14b", "--set", s], 600)
                          for s in ("kmain", "kgrav")]         + [cmd(f"rcprobe:cosmos14b:{s}:{t}", ["-m", "pilot.recapture", "probe", "--model", "cosmos14b", "--set", s,
                                               "--target", t], 3600) for s, t in (("kmain", "hspeed"), ("kgrav", "g"))]         + [cmd("rcsummary14b", ["-m", "pilot.recapture", "summary", "--family", "cosmos14b"], 60)]
    jobs["rcinshow"] = [cmd("rcinshow", ["-m", "pilot.recapture", "inshow"], 1800)]
    # paired 14B - 2B fixed-box know: rerun both probes (deterministic; they also save per-clip MRA), then pair
    jobs["rcpair14b"] = [cmd(f"rcprobe:{m}:{s}:{t}", ["-m", "pilot.recapture", "probe", "--model", m, "--set", s, "--target", t], 3600)
                         for m in ("cosmos2b", "cosmos14b") for s, t in (("kmain", "hspeed"), ("kgrav", "g"))]         + [cmd("rcpair14b", ["-m", "pilot.recapture", "pair"], 300)]
    # rcpair14b ran out of budget before the 14B gravity probe (est 1 h; measured 14B probe 0.33 h) and the rcinshow
    # rerun inherited the 2026-11-11 start file. One fresh job finishes both: the missing probe, the pair, input vs show.
    # VACE gravity (user, 2026-11-13, option A): fixed-box re-capture on the 30 kgrav scenes VACE has (kgrav_00-09 sort
    # first, so --limit 30), then its check, its probe (exploratory, outside both Holm families) and input vs show
    jobs["rcvacegrav"] = [dict(name="rc:vace13:kgrav", kind="cmd", model="vace13", est_s=LOAD_S + 30 * 115,
                               cmd=[PY, "-m", "pilot.recapture", "gen", "--model", "vace13", "--set", "kgrav", "--limit", "30"]),
                          cmd("rccheck:vace13:kgrav", ["-m", "pilot.recapture", "check", "--model", "vace13", "--set", "kgrav"], 300),
                          cmd("rcprobe:vace13:kgrav:g", ["-m", "pilot.recapture", "probe", "--model", "vace13", "--set", "kgrav",
                                                         "--target", "g"], 900),
                          cmd("rcinshow", ["-m", "pilot.recapture", "inshow"], 900)]
    LTX_RUNS["rcvacegrav"] = ["runs/rc/vace13"]
    # rcvacegrav generation took 2.05 h (est 1 h), so the budget skipped the rest; this finishes it without the model
    jobs["rcvacegravfin"] = jobs["rcvacegrav"][1:]
    jobs["rc14bfinish"] = [cmd("rcprobe:cosmos14b:kgrav:g", ["-m", "pilot.recapture", "probe", "--model", "cosmos14b", "--set", "kgrav",
                                                             "--target", "g"], 1500),
                           cmd("rcpair14b", ["-m", "pilot.recapture", "pair"], 120),
                           cmd("rcinshow", ["-m", "pilot.recapture", "inshow"], 900)]
    jobs["rtknowscale"] = [cmd("rtknowscale", ["-m", "pilot.redteam_fixes", "knowscale"], 600)]
    # host benchmark only: 120 s sustained bf16 matmul with power and clock sampling (the pre-setup check runs 30 s)
    jobs["gpubench"] = [cmd("gpubench", ["scripts/gpu_check.py", "--tag", "_ltx_gpubench_long", "--seconds", "120"], 300)]
    jobs["rtfix"] = [cmd("rtfix", ["-m", "pilot.redteam_fixes", "all"], 3600)]
    LTX_RUNS["rcprobe"] = LTX_RUNS["rcprobe14b"] = LTX_RUNS["rcinshow"] = LTX_RUNS["rcpair14b"] = LTX_RUNS["rc14bfinish"] = LTX_RUNS["rcvacegravfin"] = LTX_RUNS["rtknowscale"] = LTX_RUNS["gpubench"] = LTX_RUNS["rtfix"] = []
    return jobs


RC_JOBS = _rc_jobs()
LTX_JOBS = _ltx_jobs() | _shard_jobs() | _c9_jobs() | MODEL_JOBS | STEER_JOBS | {"stats2": STATS2} | RESCORE_JOBS | RENDER3_JOBS \
    | {"stats2v2": STATS2V2} | C2_JOBS | RC_JOBS
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
