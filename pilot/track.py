"""Show readout: SAM2 masks -> centroids -> metres -> kinematic fits.

    python -m pilot.track validate data/renders/kmain --scenes data/scenes/kmain   # P0 on renders
    ... --tracker cotracker3                     # P0 of the CoTracker3 readout (configs/deviations.md D3)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pilot import metrics

_SAM = {}


def load_video(path):
    """-> uint8 frames (T, H, W, 3) RGB, fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    return np.stack(frames), float(fps)


def sam2_masks(frames, boxes, model_id=None, device=None, prompt_frame=0):
    """frames uint8 (T, H, W, 3); boxes [[x0, y0, x1, y1], ...] on prompt_frame. Returns bool (n_obj, T, H, W).
    Propagates forward from the prompt frame, then backward if it is not the first frame."""
    import os
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    model_id = model_id or os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-large")   # tests use the tiny one
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if model_id not in _SAM:
        _SAM[model_id] = (Sam2VideoProcessor.from_pretrained(model_id),
                          Sam2VideoModel.from_pretrained(model_id).to(device, dtype=dtype).eval())
    proc, model = _SAM[model_id]
    T, H, W = frames.shape[:3]
    sess = proc.init_video_session(video=list(frames), inference_device=device, dtype=dtype)
    proc.add_inputs_to_inference_session(sess, frame_idx=prompt_frame, obj_ids=list(range(1, len(boxes) + 1)),
                                         input_boxes=[[list(map(float, b)) for b in boxes]])
    out = np.zeros((len(boxes), T, H, W), bool)
    with torch.inference_mode():
        model(inference_session=sess, frame_idx=prompt_frame)   # segment the prompted frame first
        for reverse in ([False, True] if prompt_frame > 0 else [False]):
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=prompt_frame, reverse=reverse):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                out[:, o.frame_idx] = m.reshape(len(boxes), H, W).cpu().numpy()
    return out


def centroids(masks):
    """masks (T, H, W) -> centroid (T, 2) in pixels (nan where empty), area (T,)."""
    T = masks.shape[0]
    c = np.full((T, 2), np.nan)
    area = masks.reshape(T, -1).sum(1).astype(float)
    for i in np.flatnonzero(area > 0):
        v, u = np.nonzero(masks[i])
        c[i] = u.mean(), v.mean()
    return c, area


def cotracker_tracks(frames, masks0, n_max=64, de_max=30.0, device=None):
    """CoTracker3 (offline, torch.hub facebookresearch/co-tracker) on a grid of points inside each object's frame-0 SAM2
    mask. Per object, like centroids(): centroid (T, 2) = mean of the visible points, and an area in frame-0 mask pixels =
    mask area x (convex hull of the visible points / its frame-0 value), so measurable()'s 0.5-2x band reads the same.
    A point counts as visible only if CoTracker says so AND its pixel is within de_max (CIELAB) of its frame-0 colour,
    so points left on the background where a generated ball faded out cannot make it "present". A frame with under
    half the points visible counts as absent (area 0, centroid nan)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cotracker3" not in _SAM:
        _SAM["cotracker3"] = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True).to(device)
    q, owner = [], []
    for k, m in enumerate(masks0):
        inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # keep seeds off the soft edge
        ys, xs = np.nonzero(inner if inner.sum() >= 3 else m)
        step = max(1, int(np.sqrt(len(xs) / n_max)))
        sel = (xs % step == 0) & (ys % step == 0)
        q += [[0.0, x, y] for x, y in zip(xs[sel], ys[sel])]
        owner += [k] * int(sel.sum())
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    with torch.inference_mode():
        pts, vis = _SAM["cotracker3"](video, queries=torch.tensor(q, dtype=torch.float32, device=device)[None])
    pts, vis, owner = pts[0].cpu().numpy(), vis[0].cpu().numpy(), np.array(owner)
    H, W = frames.shape[1:3]
    u = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
    lab = np.stack([cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[v[i], u[i]] for i, f in enumerate(frames)])
    vis &= np.linalg.norm(lab - lab[:1], axis=-1) < de_max
    hull = lambda p: cv2.contourArea(cv2.convexHull(p.astype(np.float32))) if len(p) >= 3 else 0.0
    out = []
    for k, m in enumerate(masks0):
        p, v = pts[:, owner == k], vis[:, owner == k]
        c, area, h0 = np.full((len(p), 2), np.nan), np.zeros(len(p)), max(hull(p[0]), 1e-6)
        for i in np.flatnonzero(v.mean(1) >= 0.5):
            c[i] = p[i, v[i]].mean(0)
            area[i] = m.sum() * hull(p[i, v[i]]) / h0
        out.append((c, area))
    return out


def object_tracks(frames, masks, tracker="sam2"):
    """Per object (centroid (T, 2) px, area (T,)): from the SAM2 masks, or CoTracker3 seeded from their frame 0."""
    return [centroids(m) for m in masks] if tracker == "sam2" else cotracker_tracks(frames, masks[:, 0])


def px_per_m(masks, size_m):
    """Scale from the object's known size: median over frames of min(mask width, mask height).
    Area overstates size (visible top face of boxes, motion blur, soft edges); blur stretches one axis,
    the top face the other, so the smaller extent is the cleaner one."""
    ext = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs):
            ext.append(min(np.ptp(xs), np.ptp(ys)) + 1)
    return float(np.median(ext)) / size_m


def measurable(area, i0, min_present=0.8, area_lo=0.5, area_hi=2.0):
    """Object present in >= 80% of scored frames with area within 0.5-2x its frame-0 area."""
    ref, a = area[0], area[i0:]
    if ref <= 0 or len(a) == 0:
        return False
    ok = (a > 0) & (a >= area_lo * ref) & (a <= area_hi * ref)
    return bool(ok.mean() >= min_present)


FAULTS = ("vanished", "offframe", "edge", "split", "recoloured", "shrunk", "grown", "lost_seen")


def mask_faults(frames, mask, i0, de_max=30.0):
    """Frame counts of what went wrong with one object's mask over the scored frames (i >= i0), against frame 0.
    Empty frames are vanished, or offframe if the last mask touched the border. Non-empty frames: edge (touches the
    border), split (>= 2 parts of >= 10% of the frame-0 area), recoloured (mean CIELAB colour > de_max from frame 0),
    shrunk / grown (outside measurable's 0.5-2x band). So measurable == (n - vanished - offframe - shrunk - grown) / n
    >= 0.8. lost_seen: empty frames with >= half the frame-0 area of frame-0-coloured pixels near the last position,
    i.e. the ball is probably still there and the tracker lost it. Diagnostic only, not the gate."""
    lab = lambda i: cv2.cvtColor(frames[i].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    ref = int(mask[0].sum())
    out = dict.fromkeys(FAULTS, 0) | dict(n=len(mask) - i0, ref_px=ref)
    if ref == 0:
        return out
    c0 = lab(0)[mask[0]].mean(0)
    r = int(3 * np.sqrt(ref))                             # search window: ~3 ball diameters around the last centroid
    edge, (u, v) = False, centroids(mask[:1])[0][0]
    for i in range(i0, len(mask)):
        m = mask[i]
        if not m.any():
            out["offframe" if edge else "vanished"] += 1
            near = lab(i)[max(int(v) - r, 0):int(v) + r, max(int(u) - r, 0):int(u) + r]
            out["lost_seen"] += int((np.linalg.norm(near - c0, axis=-1) < de_max).sum() >= ref / 2)
            continue
        edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        out["edge"] += edge
        ys, xs = np.nonzero(m)
        u, v = xs.mean(), ys.mean()
        _, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        out["split"] += int((st[1:, cv2.CC_STAT_AREA] >= 0.1 * ref).sum() >= 2)
        out["recoloured"] += int(np.linalg.norm(lab(i)[m].mean(0) - c0) > de_max)
        out["shrunk"] += int(len(xs) < 0.5 * ref)
        out["grown"] += int(len(xs) > 2.0 * ref)
    return out


# ---------------------------------------------------------------- fits (metres, z up, t in seconds)
def flight_segments(t, z, min_pts=5):
    """Split a bouncing trajectory at floor contacts; fit z = c0 + c1 t + c2 t^2 per flight.
    Contact samples are excluded, so no fit spans a bounce. Resting stretches are skipped."""
    ok = ~np.isnan(z)
    t, z = t[ok], z[ok]
    if len(z) < min_pts:
        return []
    span = np.ptp(z)
    low = z.min() + 0.15 * span
    dz = np.diff(z)
    contacts = [i + 1 for i in range(len(dz) - 1) if dz[i] < 0 <= dz[i + 1] and z[i + 1] < low]
    edges = [0] + contacts + [len(z)]
    segs = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        idx = np.arange(a if k == 0 else a + 1, b)          # drop the contact sample that starts a flight
        if len(idx) >= min_pts and np.ptp(z[idx]) > 0.05 * span:
            c2, c1, c0 = np.polyfit(t[idx], z[idx], 2)
            if c2 < 0:                                      # a free flight is concave down
                segs.append(dict(t0=t[idx[0]], t1=t[idx[-1]], c0=c0, c1=c1, c2=c2, n=len(idx)))
    return segs


def speed_at(seg, z_floor):
    """|dz/dt| where a flight parabola is at height z_floor: (dz/dt)^2 = c1^2 + 4 c2 (z - c0)."""
    return float(np.sqrt(max(seg["c1"] ** 2 + 4 * seg["c2"] * (z_floor - seg["c0"]), 0.0)))


def fit_bounce(t, z, z_floor, g_grid=np.geomspace(0.3, 60, 25), e_grid=(0.3, 0.5, 0.7, 0.85, 0.95)):
    """Fit the bouncing-ball law (start height, start velocity, g, restitution e) to a whole track.
    Works at low frame rates because it uses every sample, not separate per-flight parabolas.
    Returns g, e, rmse (m) and the fitted segments; g is in the clip's own clock."""
    from scipy.optimize import least_squares
    from pilot.scenes import eval_segments, vertical_segments
    ok = ~np.isnan(z)
    t, z = t[ok] - t[ok][0], z[ok]
    if len(z) < 6:
        return None
    T = t[-1] + 1e-6

    def model(p):
        segs, _ = vertical_segments(max(p[0], z_floor), p[1], [(0.0, abs(p[2]))], np.clip(p[3], 0, 0.99), z_floor, T)
        return eval_segments(segs, t)

    v0 = (z[1] - z[0]) / (t[1] - t[0])
    g_seeds = [-2 * s["c2"] for s in flight_segments(t, z)]           # per-flight estimates as extra starts
    starts = sorted(((z[0], v0, g, e) for g in [*g_grid, *g_seeds] for e in e_grid),
                    key=lambda p: np.sum((model(p) - z) ** 2))[:3]
    fits = [least_squares(lambda p: model(p) - z, p0, x_scale=[0.1, 0.5, p0[2], 0.1],
                          bounds=([z_floor, -20, 0.05, 0.0], [z_floor + 10, 20, 200, 0.99])) for p0 in starts]
    r = min(fits, key=lambda f: f.cost)
    p = r.x
    segs, contacts = vertical_segments(p[0], p[1], [(0.0, p[2])], p[3], z_floor, T)
    return dict(g=float(p[2]), e=float(p[3]), z0=float(p[0]), vz0=float(p[1]),
                rmse=float(np.sqrt(np.mean(r.fun ** 2))), contacts=contacts)


def readout(family, t, xs, zs, t_start):
    """Kinematic readout from tracks in metres. xs, zs: lists per object. Only t >= t_start is scored."""
    w = t >= t_start
    out = {}
    if family == "cv":
        x, z = xs[0][w], zs[0][w]
        ok = ~np.isnan(x)
        c = np.polyfit(t[w][ok], x[ok], 2)
        out.update(v=abs(np.polyval(np.polyder(c), t[w][ok]).mean()), a=2 * c[0],
                   residual=metrics.accel_residual(np.c_[x[ok], z[ok]], t[1] - t[0]))
        return out
    if family == "bounce":
        x, z = xs[0][w], zs[0][w]
        ok = ~np.isnan(x)
        out["vx"] = abs(np.polyfit(t[w][ok], x[ok], 1)[0])
        fit = fit_bounce(t[w], z, np.nanmin(zs[0]))
        if fit:
            out.update(g=fit["g"], e=fit["e"], fit_rmse=fit["rmse"])
        segs = flight_segments(t[w], z)
        if segs:
            longest = max(segs, key=lambda s: s["n"])
            sel = (t[w] >= longest["t0"]) & (t[w] <= longest["t1"]) & ok
            out["residual"] = metrics.accel_residual(z[sel], t[1] - t[0])
        return out
    if family == "ratio":
        flights = []
        for z in zs:
            z_floor = np.nanmin(z)
            fit = fit_bounce(t[w], z[w], z_floor)
            if not fit or not fit["contacts"]:
                return out
            t_land = t[w][0] + fit["contacts"][0]           # launch at t = 0, so flight time = first landing
            v_up = np.sqrt(2 * fit["g"] * max(fit["z0"] - z_floor, 0) + fit["vz0"] ** 2)
            flights.append((t_land, v_up ** 2 / (2 * fit["g"])))
        (T1, h1), (T2, h2) = flights
        out.update(t_ratio=T1 / T2, h_ratio=h1 / h2)
        return out
    raise ValueError(family)


def gt_boxes(spec, frame=0):
    """Ground-truth boxes on one frame from the analytic tracks."""
    boxes = []
    for o, tr in zip(spec["objects"], spec["tracks"]):
        u, v, r = tr["u_px"][frame], tr["v_px"][frame], tr["radius_px"]
        boxes.append([u - r, v - r, u + r, v + r])
    return boxes


def clip_frames(video_path, spec):
    """Frames at the scene's render geometry (generated clips come back at the model's size), fps."""
    frames, fps = load_video(video_path)
    cam = spec["camera"]
    if frames.shape[1:3] != (cam["height"], cam["width"]):
        frames = np.stack([cv2.resize(f, (cam["width"], cam["height"]), interpolation=cv2.INTER_AREA) for f in frames])
    return frames, fps


def track_clip(video_path, spec, boxes=None, t_start=0.0, cond_frames=1, tracker="sam2"):
    """Track a clip of a known scene. Returns readout dict plus measurability, SAM2 mask faults and pixel tracks.
    The metre scale comes from the SAM2 masks of the input frames with either tracker."""
    frames, fps = clip_frames(video_path, spec)
    cam = spec["camera"]
    t = np.arange(len(frames)) / fps
    masks = sam2_masks(frames, boxes or gt_boxes(spec))
    i0 = int(np.searchsorted(t, t_start))
    xs, zs, ok = [], [], True
    for o, m, (c, area) in zip(spec["objects"], masks, object_tracks(frames, masks, tracker)):
        s = px_per_m(m[:cond_frames], o["size_m"])
        xs.append((c[:, 0] - cam["cx"]) / s)
        zs.append(-(c[:, 1] - cam["cy"]) / s)
        ok &= measurable(area, i0)
    res = dict(measurable=ok, fps=fps, n_frames=len(frames), tracker=tracker,
               faults=[mask_faults(frames, m, i0) for m in masks])
    if ok:
        res.update(readout(spec["family"], t, xs, zs, t_start))
    res["track_px"] = [np.c_[(x * 1), (z * 1)].round(4).tolist() for x, z in zip(xs, zs)]
    return res


def validate(render_dir, scene_dir, out_csv, tracker="sam2"):
    """P0 on renders: tracker readout vs exact ground truth (gate G1a)."""
    import pandas as pd
    rows = []
    for vid in sorted(v for v in Path(render_dir).glob("*.mp4") if v.stem.count("_") == 2):   # full clips only
        spec = json.loads((Path(scene_dir) / f"{vid.stem}.json").read_text(encoding="utf-8"))
        r = track_clip(vid, spec, cond_frames=5, tracker=tracker)
        row = dict(scene=vid.stem, family=spec["family"], measurable=r["measurable"])
        for k, v in spec["gt"].items():
            if k in r:
                row[f"{k}_gt"], row[f"{k}_tr"] = v, r[k]
                row[f"{k}_sym_mra"] = float(metrics.sym_mra(r[k], v))
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    mra = df.filter(like="_sym_mra").stack().mean()
    print(f"P0 renders: mean symmetric MRA {mra:.3f}, measurable {df.measurable.mean():.2f}")
    return df


def local_kinematics(t, xy, t_q, half_window=0.125):
    """Speed and acceleration magnitude at t_q from a local quadratic fit of a 2D track (metres)."""
    sel = (np.abs(t - t_q) <= half_window) & ~np.isnan(xy[:, 0])
    if sel.sum() < 4:
        return np.nan, np.nan
    cx, cy = (np.polyfit(t[sel] - t_q, xy[sel, k], 2) for k in (0, 1))
    return float(np.hypot(cx[1], cy[1])), float(np.hypot(2 * cx[0], 2 * cy[0]))


def validate_qval(qfile="data/real/qval_questions.json", root="data/quantiphy_val/validation_videos",
                  out_csv="results/p0_qval.csv"):
    """P0 on real footage: QuantiPhy val answers for 2D-motion clips recovered by the tracker (gate G1a).
    Each object has its own box, prompt time and known size (the scale comes from the object itself)."""
    import pandas as pd
    rows = []
    for clip, q in json.loads(Path(qfile).read_text(encoding="utf-8")).items():
        frames, fps = load_video(Path(root) / f"{clip}.mp4")
        t = np.arange(len(frames)) / fps
        tracks = {}
        for name, o in q["objects"].items():
            f0 = int(round(o["prompt_t"] * fps))
            m = sam2_masks(frames, [o["box"]], prompt_frame=f0)[0]
            c, area = centroids(m)
            tracks[name] = c / px_per_m(m[max(f0 - 3, 0):f0 + 4], o["size_m"])   # size around the prompt frame
        for obj, quantity, tq, unit, answer in q["questions"]:
            if quantity == "avg_speed":                          # tq = (t0, t1): mean displacement rate
                i0, i1 = (int(round(x * fps)) for x in tq)
                pred = np.linalg.norm(tracks[obj][i1] - tracks[obj][i0]) / (t[i1] - t[i0]) / unit
            else:
                v, acc = local_kinematics(t, tracks[obj], tq, half_window=0.125 if quantity == "speed" else 0.25)
                pred = (v if quantity == "speed" else acc) / unit
            rows.append(dict(clip=clip, object=obj, quantity=quantity, t=str(tq), answer=answer, tracked=pred,
                             sym_mra=float(metrics.sym_mra(pred, answer)) if np.isfinite(pred) else 0.0))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"P0 QuantiPhy val: mean symmetric MRA {df.sym_mra.mean():.3f} over {len(df)} questions")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("renders")
    v.add_argument("--scenes", required=True)
    v.add_argument("--out", default="results/p0_renders.csv")
    v.add_argument("--tracker", choices=["sam2", "cotracker3"], default="sam2")
    q = sub.add_parser("qval")
    q.add_argument("--out", default="results/p0_qval.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "validate":
        validate(a.renders, a.scenes, a.out, a.tracker)
    else:
        validate_qval(out_csv=a.out)
