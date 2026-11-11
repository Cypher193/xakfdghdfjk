"""Keep pilot data on a private Hugging Face dataset repo, so any pod (any datacenter) can be thrown away.

    python scripts/sync.py pull                 # at pod start: renders, scenes, runs, results, logs
    python scripts/sync.py push                 # once
    python scripts/sync.py push --every 900     # loop: push every 15 min (run in the background)

Model weights are not synced (re-download from the Hub). Uses HF_TOKEN. Pushes are resumable and only
upload changed files. Repo: HF_DATA_REPO, default <your HF user>/cvpr-vail-pilot.
LTX pods (LTX_JOB=gen1..gen4|probe|probe2|stats2|sKofN|c9[bc]sKofN|c9cchain-COND|{model}sKofN|{model}probe|rescore,
scripts/ltx_pod.sh; {model}: run_pilot.MODEL_JOBS,
patterns derived from the job's steps): PATTERNS (pull) = inputs + the job's runs, logs and
results (probe, probe2: also every runs/*/ltx2b; shards sKofN: only runs/*/ltx2b/*/meta.json and score.json, so
PATTERNS is not a superset of PUSH_PATTERNS); PUSH_PATTERNS = only the job's own files.
Screen pods (SCREEN_TIER=A|B|all, set by scripts/screen_pod.sh) pull the inputs plus their own runs, logs and table
and push only those: two pods share the repo, and a pod must never re-upload a stale pulled copy of the other's files.
"""
import argparse
import os
import re
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, whoami

PATTERNS = ["data/scenes/**", "data/renders/**", "data/logs/**", "data/real/**", "runs/**", "results/**"]


def repo_id():
    return os.environ.get("HF_DATA_REPO") or f"{whoami()['name']}/cvpr-vail-pilot"


IGNORE = ["**/*.partial.mp4", "**/*.tmp"]      # clips still being written
BROKEN = ["runs/b1/ltx2b/kmain_cv_41_s2/video.mp4"]   # HF size metadata wrong: every pull of it fails ("File size mismatch")
ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("SCREEN_TIER"):
    from run_pilot import screen_models, tag
    t = tag("screen", os.environ["SCREEN_TIER"])
    PATTERNS = ["data/scenes/**", "data/renders/**", "data/real/**", f"data/logs/*{t}*", f"results/*{t}*"] + \
        [f"runs/*/{m}/**" for m in screen_models(os.environ["SCREEN_TIER"])]
elif os.environ.get("LTX_JOB"):
    from run_pilot import LTX_PULL_EXTRA, LTX_RUNS, MODEL_JOBS, tag
    j = os.environ["LTX_JOB"]
    base = ["data/scenes/**", "data/renders/**", "data/real/**"]
    mine = [f"data/logs/*{tag('ltx', j)}*", f"results/*{tag('ltx', j)}*"]
    PUSH_PATTERNS = base + mine + [f"{r}/**" for r in LTX_RUNS[j]] +         (["results/p6_ltx2b*"] if j.startswith("probe") else [])    # pilot.probe writes results/p6_{model}_{target}_{pool}.csv
    PATTERNS = PUSH_PATTERNS + ([f"{r}/**" for r in LTX_PULL_EXTRA] if j.startswith("probe") else [])   # pull: a superset
    if re.fullmatch(r"s\d+of\d+", j):     # phase 2 shard: only the old clips' meta/score (skip check, scorer), no videos
        PATTERNS = base + mine + [f"{r}/*/{f}" for r in LTX_RUNS[j] for f in ("meta.json", "score.json")]
        PATTERNS.append("assets_cache/**")  # Poly Haven lists + maps from the laptop: same choices as the pilot, no downloads
    if j.startswith("c9cchain-"):          # C9c chain: input renders, B1 seed-0 acts (group 0), its own chain (resume)
        PATTERNS = ["data/scenes/**", "data/renders/kmain/*_input.mp4", "runs/b1/ltx2b/*_s0/acts.npz"] + mine + \
            [f"{r}/**" for r in LTX_RUNS[j]]
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]]
    if re.fullmatch(r"c9[bc]?s\d+of\d+", j):   # phase 3 C9 / C9b: input renders, the B1 fit data (acts + meta, no videos);
        fit = ["runs/c9c_chain/ltx2b/*/chain.json", "runs/c9c_chain/ltx2b/*/g*.npz"] if j.startswith("c9c") else \
            ["runs/b1/ltx2b/*/acts.npz", "runs/b1/ltx2b/*/meta.json", "results/p6_ltx2b_hspeed_obj.csv"]   # C9c: the chains
        PATTERNS = ["data/scenes/**", "data/renders/kmain/*_input.mp4"] + fit + mine
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]]
        # a relaunched shard: every shard's meta.json (cheap), and its own clips' video + score, so the job's score step
        # scores clips an earlier pod generated but never scored (it timed out first)
        from run_pilot import LTX_JOBS
        PATTERNS += [f"{r}/*/meta.json" for r in LTX_RUNS[j]] + \
            [f"{r}/{i}_s0/{f}" for r in LTX_RUNS[j] for i in LTX_JOBS[j][0]["ids"] for f in ("video.mp4", "score.json")]
    if j == "score2":                     # every LTX clip with meta.json but no score.json on HF, video + meta only
        BAD = {"runs/b1/ltx2b/kmain_cv_41_s2"}       # HF holds a size-inconsistent video.mp4 there: unreadable
        fs = HfApi().list_repo_files(repo_id(), repo_type="dataset")
        meta = {f.rsplit("/", 1)[0] for f in fs if re.fullmatch(r"runs/(b1|b2nl|b3|c9b)/ltx2b/[^/]+/meta\.json", f)}
        todo = sorted(meta - {f.rsplit("/", 1)[0] for f in fs if f.endswith("/score.json")} - BAD)
        PATTERNS = ["data/scenes/**"] + mine + [f"{d}/{x}" for d in todo for x in ("video.mp4", "meta.json")]
        PUSH_PATTERNS = mine + [f"{d}/score.json" for d in todo]
    if j.startswith("render3s"):          # round 3 input windows: old ones in (skipped by pilot.render), new ones out
        from run_pilot import RENDER3_EXPS
        sets = RENDER3_EXPS.values()
        PATTERNS = ["data/scenes/**", "assets_cache/**"] + mine + [f"data/renders/{s}/*_input*" for s in sets]
        PUSH_PATTERNS = mine + [f"data/renders/{s}/*_input*" for s in sets] + ["data/renders/kl3full/*"]
    if j == "kl3diag":                    # full ramp renders in (6 exist) and out, its own results
        PATTERNS = ["data/scenes/**", "assets_cache/**", "data/renders/kl3full/*"] + mine
        PUSH_PATTERNS = mine + ["data/renders/kl3full/*"]
    if j == "bratio2":                    # renders its own inputs: scenes + Poly Haven cache in; renders + runs out
        PATTERNS = ["data/scenes/kratio2/**", "assets_cache/**"] + mine + [f"{r}/**" for r in LTX_RUNS[j]]
        PUSH_PATTERNS = ["data/renders/kratio2/**"] + mine + [f"{r}/**" for r in LTX_RUNS[j]]
    if j == "probe2":                    # pilot.probe reads acts.npz + meta.json + the scene specs only, no videos/renders
        PATTERNS = ["data/scenes/**"] + mine + [f"{r}/*/{f}" for r in LTX_PULL_EXTRA for f in ("acts.npz", "meta.json")]
    if j == "cosmos2bsteer":             # pilot.steer: probe direction from the B1 acts, kmain input windows
        PATTERNS = ["data/scenes/**", "data/renders/kmain/*_input.mp4", "runs/b1/cosmos2b/*/acts.npz",
                    "runs/b1/cosmos2b/*/meta.json", "results/p6_cosmos2b_hspeed_obj.csv"] + mine + \
            [f"{r}/*/{f}" for r in LTX_RUNS[j] for f in ("meta.json", "score.json", "video.mp4")]
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]]
    if j == "stats2":                     # nested gap: acts/meta/score + p6 best rows
        from run_pilot import STATS_MODELS
        PATTERNS = ["data/scenes/**"] + mine \
            + [f"runs/{e}/{m}/*/{f}" for e in ("b1", "b5") for m in STATS_MODELS for f in ("acts.npz", "meta.json", "score.json")] \
            + [f"results/p6_{m}_{t}_obj.csv" for m in STATS_MODELS for t in ("hspeed", "g", "e")]
        PUSH_PATTERNS = mine + ["results/c3_*_nested*"]
    if j in ("rescore", "rescore2") or j.endswith("rescore3"):   # G1b re-score: videos + meta (+ score_v2 of a relaunch), full kmain renders for P0
        from run_pilot import RESCORE3_RUNS, RESCORE_RUNS
        if j == "rescore2":               # round-2 Cosmos / VACE clips only; no P0
            RESCORE_RUNS = [r for r in RESCORE_RUNS if not r.endswith("ltx2b")]
        if j.endswith("rescore3"):        # one model's round 3 clips; the VACE pod also runs P0 on the full kl3 renders
            RESCORE_RUNS = RESCORE3_RUNS[j.removesuffix("rescore3")]
        PATTERNS = ["data/scenes/**", "data/real/manifest.json"] + ["data/renders/kmain/kmain_*_??.mp4"] * (j == "rescore") \
            + ["data/renders/kl3full/*.mp4"] * (j == "vace13rescore3") \
            + mine + [f"{r}/*/{f}" for r in RESCORE_RUNS for f in ("video.mp4", "meta.json", "score_v2.json")]
        PUSH_PATTERNS = mine + [f"{r}/*/score_v2.json" for r in RESCORE_RUNS]
    if j == "stats2v2":                   # as stats2, CoTracker3 readout
        from run_pilot import STATS_MODELS
        PATTERNS = ["data/scenes/**"] + mine \
            + [f"runs/{e}/{m}/*/{f}" for e in ("b1", "b5") for m in STATS_MODELS for f in ("acts.npz", "meta.json", "score_v2.json")] \
            + [f"results/p6_{m}_{t}_obj.csv" for m in STATS_MODELS for t in ("hspeed", "g", "e")]
        PUSH_PATTERNS = mine + ["results/c3_*_v2_nested*"]
    if j.startswith(("c2", "c3")):        # C2 / C2-g: its scene bases' renders (rendered here, except kcont)
        from run_pilot import C2_BASES
        rend = [f"data/renders/{s}/{b}_x*" for s, bs in C2_BASES[j].items() for b in bs]
        PATTERNS = [f"data/scenes/{s}/**" for s in C2_BASES[j]] + ["assets_cache/**"] + rend + mine   # own sets only
        PUSH_PATTERNS = mine + [r for r in rend if "/kcont/" not in r]
    if j == "rederr":                     # only the listed clips, both score files
        from run_pilot import REDERR_DIRS
        PATTERNS = ["data/scenes/**", "data/real/manifest.json"] + mine + \
            [f"{d}/{f}" for d in REDERR_DIRS for f in ("video.mp4", "meta.json", "score.json", "score_v2.json")]
        PUSH_PATTERNS = mine + [f"{d}/score*.json" for d in REDERR_DIRS]
    gens = [s for s in MODEL_JOBS.get(j, []) if s["kind"] == "gen"]
    if gens:                              # cosmos2b / vace13 shard: inputs, every shard's meta + score + video (skip check;
        import yaml                       # a relaunch scores clips an earlier pod never scored; ~3 MB a clip). Not the
        exps = yaml.safe_load(open(ROOT / "configs/pilot.yaml"))["experiments"]   # shard's ids: on a fresh pod this runs
        scenes = dict.fromkeys(exps[s["name"].split(":")[0]]["scenes"] for s in gens)   # before data/scenes exists
        PATTERNS = ["data/scenes/**", "data/real/**"] + [f"data/renders/{x}/*_input.mp4" for x in scenes if x != "real"] \
            + mine + [f"{r}/*/{f}" for r in LTX_RUNS[j] for f in ("meta.json", "score.json", "video.mp4")]
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]]
        from run_pilot import NARROW_PULL     # 2026-11-06 jobs: only the shard's scenes, inputs and clips; every
        if j in NARROW_PULL:                  # shard's files (~2,700 per pod) hit HF's 429s on parallel setups
            sets = {s["out"][0]: exps[s["name"].split(":")[0]]["scenes"] for s in gens}
            PATTERNS = [f"data/scenes/{x}/**" for x in dict.fromkeys(sets.values()) if x != "real"] + ["data/real/**"]                 + mine + [f"data/renders/{sets[s['out'][0]]}/{i}_input.mp4" for s in gens for i in s["ids"]]                 + [f"{s['out'][0]}/{i}_s{sd}/{f}" for s in gens for i in s["ids"] for sd in s["seeds"]
                   for f in ("meta.json", "score.json", "video.mp4")]
            if any(s["name"].startswith("render") for s in MODEL_JOBS[j]):    # replication shards render their inputs
                PATTERNS += ["assets_cache/**"]
    elif j in MODEL_JOBS:                 # {model}probe: as probe2, for that model's B1 + B5 clips (score_v2: nested --v2)
        m = j.removesuffix("probe")
        PATTERNS = ["data/scenes/**"] + mine + [f"runs/{e}/{m}/*/{f}" for e in ("b1", "b5")
                                                for f in ("acts.npz", "meta.json", "score_v2.json")]
        PUSH_PATTERNS = mine + [f"results/p6_{m}*", f"results/c3_{m}_*"]
    if j == "gammaprobe":                 # H3: every generator's B5 acts in, per-block probe gamma out
        from run_pilot import GAMMA_MODELS
        PATTERNS = ["data/scenes/kgrav/**"] + mine + [f"runs/b5/{m}/*/{f}" for m in GAMMA_MODELS for f in ("acts.npz", "meta.json")]
        PUSH_PATTERNS = mine + ["results/c3_*_gamma.csv"]
    if j == "freshprobe":                 # replication (D12): b1f / b5f acts + CoTracker3 scores in; probes and gaps out
        PATTERNS = ["data/scenes/kmainf/**", "data/scenes/kgravf/**"] + mine +             [f"runs/{e}/{m}/*/{f}" for e in ("b1f", "b5f") for m in ("ltx2b", "cosmos2b") for f in ("acts.npz", "meta.json", "score_v2.json")]
        PUSH_PATTERNS = mine + ["results/p6_*f_*", "results/c3_*f_*"]
    if j == "wrapup":                     # score files + meta of every clip, earlier results; gates reports out
        from run_pilot import STATS_MODELS, WRAPUP_EXPS
        PATTERNS = ["data/scenes/**", "data/real/**", "results/**"] + mine +             [f"runs/{e}/{m}/*/{f}" for e in WRAPUP_EXPS for m in STATS_MODELS for f in ("meta.json", "score.json", "score_v2.json")]
        PUSH_PATTERNS = mine + ["results/gates_*.md", "results/c3_*_gates*"]
    if j == "vjepa2s":                    # surprise at frame strides 2-4: the C2 renders in, its own results out
        PATTERNS = ["data/scenes/**"] + [f"data/renders/{s}/*.mp4" for s in ("kvelv", "kturnl", "kcontl")] + mine
        PUSH_PATTERNS = mine
    if j == "vjepa2":                     # input windows (C1) and the C2 renders in; acts, probes, surprise out
        from run_pilot import VJEPA_MODELS
        PATTERNS = ["data/scenes/**"] + [f"data/renders/{s}/*_input*.mp4" for s in ("kmain", "kgrav")]             + [f"data/renders/{s}/*.mp4" for s in ("kvelv", "kturnl", "kcontl")] + mine + [f"{r}/**" for r in LTX_RUNS[j]]
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]] + [f"results/p6_{m}*" for m in VJEPA_MODELS]
    if j in ("rcltx", "rcvace", "rcnano") or j.startswith("rccos"):   # red-team re-capture: inputs in, rc acts out
        PATTERNS = [f"data/scenes/{s}/**" for s in ("kmain", "kgrav")] + \
            [f"data/renders/{s}/*_input*.mp4" for s in ("kmain", "kgrav")] + mine + [f"{r}/**" for r in LTX_RUNS[j]]
        PUSH_PATTERNS = mine + [f"{r}/**" for r in LTX_RUNS[j]]
    if j == "rcprobe":                    # rc acts + the original seed-0 acts (reproduction check) in; rc results out
        PATTERNS = ["data/scenes/kmain/**", "data/scenes/kgrav/**", "runs/rc/**"] + mine + \
            [f"runs/{e}/{m}/*_s0/{f}" for e in ("b1", "b5") for m in ("ltx2b", "cosmos2b", "vace13", "cosmos3_nano")
             for f in ("acts.npz", "meta.json")]
        PUSH_PATTERNS = mine + ["results/rc_*"]
    if j == "rcprobe14b":
        PATTERNS = ["data/scenes/kmain/**", "data/scenes/kgrav/**", "runs/rc/cosmos14b/**"] + mine +             [f"runs/{e}/cosmos14b/*_s0/{f}" for e in ("b1", "b5") for f in ("acts.npz", "meta.json")]
        PUSH_PATTERNS = mine + ["results/rc_cosmos14b_*", "results/rc_check_cosmos14b_*", "results/rc_summary_cosmos14b.csv"]
    if j == "rcinshow":                   # rc acts (VAE pools) + the tracker clip tables in; one table out
        PATTERNS = ["data/scenes/kmain/**", "data/scenes/kgrav/**"] + mine + \
            [f"runs/rc/{m}/*_s0/{f}" for m in ("ltx2b", "cosmos2b", "vace13", "cosmos3_nano") for f in ("acts.npz", "meta.json")] + \
            [f"results/c3_{m}_v2_nested_clips.csv" for m in ("ltx2b", "cosmos2b", "vace13", "cosmos3_nano")]
        PUSH_PATTERNS = mine + ["results/rc_inshow.csv"]
    if j == "rtfix":                      # CSV results + scene specs in (no acts, no videos); rt tables out
        PATTERNS = ["data/scenes/**", "results/c3_*_v2_nested_clips.csv", "results/c2_*", "results/c2g_*"] + mine
        PUSH_PATTERNS = mine + ["results/rt_*"]
PUSH_PATTERNS = globals().get("PUSH_PATTERNS", PATTERNS)       # what push uploads and finalize verifies


def pull():
    for attempt in range(4):    # xet transfers fail now and then ("File size mismatch", probe2 setup); resumes on retry
        try:
            snapshot_download(repo_id(), repo_type="dataset", local_dir=ROOT, allow_patterns=PATTERNS,
                              ignore_patterns=BROKEN)
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"pull failed ({type(e).__name__}: {str(e)[:120]}), retrying", flush=True)
            time.sleep(30)
    print("pulled", repo_id(), flush=True)


def push():
    if not any(next(ROOT.glob(p), None) for p in PUSH_PATTERNS):    # patterns are globs ("results/p6_vace13*"), not dirs
        print("nothing to push yet", flush=True)
        return
    api = HfApi()
    # upload_large_folder pages through every commit of the repo only to warn when there are many (3,660 commits =
    # ~74 API calls a push); with ~28 pods that used most of HF's 1,000 calls per 5 min (2026-11-06)
    api.list_repo_commits = lambda *a, **k: []
    api.upload_large_folder(repo_id=repo_id(), repo_type="dataset", folder_path=ROOT,
                            allow_patterns=PUSH_PATTERNS, ignore_patterns=IGNORE, private=True, print_report=False)
    print(time.strftime("%H:%M:%S"), "pushed", repo_id(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "push"])
    ap.add_argument("--every", type=int, default=0, help="seconds between pushes (loop)")
    a = ap.parse_args()
    if a.cmd == "pull":
        pull()
    else:
        while True:
            try:
                push()
            except Exception as e:          # a failed push must not kill the loop; the next one retries
                print("push failed:", e, flush=True)
            if not a.every:
                break
            time.sleep(a.every)
