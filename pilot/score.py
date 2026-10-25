"""Show readout for every generated clip: track, fit, score against ground truth.

    python -m pilot.score                 # all unscored runs, then results/show.csv
    python -m pilot.score --loop 300      # keep scoring new clips every 5 min (runs next to generation)
    python -m pilot.score --models ltx2 helios --table results/show_screen_B.csv   # one screen pod's models only

Per clip -> score.json next to video.mp4. Rendered scenes are tracked from the ground-truth box on frame 0;
text-only clips (B2-nl) have no input frames, so the object is found on frame 0 by its colour; real clips use
the hand-drawn box from the manifest. Scoring starts after the model's input window (meta cond_seconds).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pilot import metrics, track
from pilot.probe import target_value

HUE = {"red": [(0, 10), (170, 180)], "orange": [(10, 22)], "yellow": [(22, 35)], "green": [(35, 85)],
       "blue": [(95, 130)], "purple": [(130, 160)]}


def colour_box(frame, colour):
    """Box around the largest blob of a named colour (for clips with no input frames)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if colour == "white":
        m = (s < 40) & (v > 180)
    else:
        m = np.zeros(h.shape, bool)
        for lo, hi in HUE[colour]:
            m |= (h >= lo) & (h < hi)
        m &= (s > 80) & (v > 60)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, hh, area = stats[k]
    return [x, y, x + w, y + hh] if area >= 30 else None


def score_rendered(d, meta, spec):
    if meta["cond_frames"] == 0:                           # text only: find the object by colour
        frames, _ = track.load_video(d / "video.mp4")
        cam = spec["camera"]
        f0 = cv2.resize(frames[0], (cam["width"], cam["height"]), interpolation=cv2.INTER_AREA)
        boxes = [colour_box(f0, o["color"]) for o in spec["objects"]]
        if any(b is None for b in boxes):
            return dict(measurable=False, reason="object not found by colour")
        r = track.track_clip(d / "video.mp4", spec, boxes=boxes, t_start=0.0, cond_frames=5)
    else:
        r = track.track_clip(d / "video.mp4", spec, t_start=meta["cond_seconds"],
                             cond_frames=meta["cond_frames"])
    r.pop("track_px", None)
    gt = dict(spec["gt"], hspeed=target_value(spec["gt"], "hspeed"))
    if r.get("measurable"):
        if "v" in r or "vx" in r:
            r["hspeed"] = r.get("vx", r.get("v"))
        for k in ("v", "vx", "hspeed", "g", "e", "t_ratio", "h_ratio"):
            if k in r and gt.get(k) is not None:
                r[f"{k}_gt"] = gt[k]
                r[f"{k}_sym_mra"] = float(metrics.sym_mra(r[k], gt[k]))
                r[f"{k}_orig_mra"] = float(metrics.orig_mra(r[k], gt[k])) if gt[k] else None
    return r


def score_real(d, meta, entry):
    """Real inputs have no ground-truth continuation numbers: measurability and law-following only."""
    frames, fps = track.load_video(d / "video.mp4")
    H, W = frames.shape[1:3]
    sx, sy = W / entry["wh"][0], H / entry["wh"][1]
    x0, y0, x1, y1 = entry["box"]
    m = track.sam2_masks(frames, [[x0 * sx, y0 * sy, x1 * sx, y1 * sy]])[0]
    c, area = track.centroids(m)
    i0 = int(round(meta["cond_seconds"] * fps))
    r = dict(measurable=track.measurable(area, i0))
    sel = ~np.isnan(c[i0:, 0])
    if r["measurable"] and sel.sum() > 6:
        r["residual"] = metrics.accel_residual(c[i0:][sel], 1 / fps)
    return r


def find_spec(scene, manifest):
    f = next(Path("data/scenes").glob(f"*/{scene}.json"), None)
    if f is not None:
        return "rendered", json.loads(f.read_text(encoding="utf-8"))
    return "real", manifest.get(scene)


def score_all(runs="runs", models=None):
    manifest = {e["id"]: e for e in json.loads(Path("data/real/manifest.json").read_text(encoding="utf-8"))}
    done = 0
    for v in sorted(Path(runs).glob("*/*/*/video.mp4")):
        d = v.parent
        if (d / "score.json").exists() or not (d / "meta.json").exists() or d.parts[-3] == "timing"                 or (models and d.parts[-2] not in models):
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        kind, spec = find_spec(meta["scene"], manifest)
        try:
            r = score_rendered(d, meta, spec) if kind == "rendered" else score_real(d, meta, spec)
        except Exception as e:                                # one bad clip must not stop the scorer
            r = dict(measurable=False, reason=f"error: {type(e).__name__}: {e}"[:300])
        r = {k: (float(x) if isinstance(x, (np.floating, np.integer)) else x) for k, x in r.items()}
        (d / "score.json").write_text(json.dumps(r, indent=1))
        done += 1
        print("scored", d, {k: r[k] for k in r if k.endswith("sym_mra") or k == "measurable"}, flush=True)
    return done


def table(runs="runs", out="results/show.csv", models=None):
    rows = []
    for s in sorted(Path(runs).glob("*/*/*/score.json")):
        if models and s.parts[-3] not in models:
            continue
        meta = json.loads((s.parent / "meta.json").read_text(encoding="utf-8"))
        rows.append(dict(exp=s.parts[-4], model=s.parts[-3], scene=meta["scene"], seed=meta["seed"],
                         alpha=meta.get("alpha"), cond=meta.get("cond"), wall_s=meta.get("wall_s"))
                    | json.loads(s.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = one pass")
    ap.add_argument("--until-file", default="", help="loop exits after a pass once this file exists")
    ap.add_argument("--models", nargs="*", help="only these model run dirs (default: all)")
    ap.add_argument("--table", default="results/show.csv")
    a = ap.parse_args()
    while True:
        score_all(a.runs, a.models)
        df = table(a.runs, a.table, a.models)
        print(time.strftime("%H:%M:%S"), f"scored clips: {len(df)}", flush=True)
        if not a.loop or (a.until_file and Path(a.until_file).exists()):
            break
        time.sleep(a.loop)
