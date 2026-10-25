"""Pilot scene specs with exact, analytic ground truth.

Objects move in the plane y = 0, facing a fixed pinhole camera, so pixels convert to metres
with one scale (the QuantiPhy assumption). Motion is closed-form (constant velocity, or
constant gravity with restitution bounces), so the renderer just keyframes positions and
the ground truth is exact. Gravity can switch value at one time (used for C2-g continuations).

    python -m pilot.scenes --out data/scenes
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

G = 9.81
REST_SPEED = 0.05      # bounce speed (m/s) below which a ball comes to rest
RATIO_CAMERA = dict(distance=9.0, cam_z=1.3)   # two-ball throws: farther and higher, frame spans z -1.06..3.66 m
COLORS = ["red", "blue", "yellow", "green", "orange", "white", "purple"]


# ---------------------------------------------------------------- camera
def camera(cfg, distance=None, cam_z=0.0):
    r = cfg["render"]
    fx = (r["width"] / 2) / math.tan(math.radians(r["hfov_deg"] / 2))
    return dict(width=r["width"], height=r["height"], fx=fx, cx=r["width"] / 2, cy=r["height"] / 2,
                distance=distance or r["camera_distance_m"], cam_z=cam_z, hfov_deg=r["hfov_deg"],
                floor_z=r["floor_z_m"])


def project(cam, x, z):
    """World point in the plane y = 0 -> pixel (u, v). Camera at (0, -D, cam_z) looking along +y."""
    s = cam["fx"] / cam["distance"]
    return cam["cx"] + s * np.asarray(x), cam["cy"] - s * (np.asarray(z) - cam.get("cam_z", 0.0))


def px_per_m(cam):
    return cam["fx"] / cam["distance"]


# ---------------------------------------------------------------- motion
def vertical_segments(z0, vz0, gravity, e, floor, t_end):
    """Piecewise-ballistic vertical motion of a ball centre above `floor`.
    gravity = [(t_start, g), ...] sorted. Returns segments (t0, t1, z0, vz0, g) and contact times."""
    segs, contacts = [], []
    t, z, vz, gi, resting = 0.0, z0, vz0, 0, False
    while t < t_end - 1e-12:
        g = gravity[gi][1]
        t_switch = gravity[gi + 1][0] if gi + 1 < len(gravity) else math.inf
        if resting:
            t1 = min(t_switch, t_end)
            segs.append((t, t1, floor, 0.0, 0.0))
            t, gi = t1, gi + (t1 == t_switch)
            continue
        tau = (vz + math.sqrt(vz * vz + 2 * g * max(z - floor, 0.0))) / g
        t1 = min(t + tau, t_switch, t_end)
        segs.append((t, t1, z, vz, g))
        dt = t1 - t
        z, vz = z + vz * dt - 0.5 * g * dt * dt, vz - g * dt
        if t1 == t + tau:                      # floor contact
            contacts.append(t1)
            z, vz = floor, -e * vz
            resting = abs(vz) < REST_SPEED
        if t1 == t_switch:
            gi += 1
        t = t1
    return segs, contacts


def eval_segments(segs, t):
    t = np.atleast_1d(np.asarray(t, float))
    z = np.empty_like(t)
    for i, ti in enumerate(t):
        s = next((s for s in segs if s[0] <= ti <= s[1]), segs[-1])
        d = ti - s[0]
        z[i] = s[2] + s[3] * d - 0.5 * s[4] * d * d
    return z


def object_track(scene, obj, t):
    """Centre (x, z) in metres at times t."""
    floor = scene["floor_z"] + obj["size_m"] / 2
    x = obj["x0"] + obj["vx"] * np.asarray(t, float)
    if obj["motion"] == "floor":
        return x, np.full_like(x, floor)
    segs, _ = vertical_segments(obj["z0"], obj["vz0"], scene["gravity"], obj["e"], floor, scene["duration_s"])
    return x, eval_segments(segs, t)


# ---------------------------------------------------------------- sampling
def _ball(rng, motion="bounce"):
    return dict(shape="sphere", motion=motion, color=str(rng.choice(COLORS)),
                size_m=round(float(rng.uniform(0.12, 0.35)), 3))


def make_cv(rng, sid, cam):
    shape = str(rng.choice(["sphere", "cube", "cylinder"]))
    d = float(rng.choice([-1, 1]))
    v = float(rng.uniform(0.2, 0.65))   # stays in frame for the 6 s render
    obj = dict(shape=shape, motion="floor", color=str(rng.choice(COLORS)),
               size_m=round(float(rng.uniform(0.15, 0.40)), 3), x0=-d * 1.9, vx=d * v, z0=None, vz0=0.0, e=None)
    return dict(id=sid, family="cv", gravity=[(0.0, G)], objects=[obj],
                gt=dict(v=abs(v), a=0.0))


def make_bounce(rng, sid, gmult=1.0, base=None):
    if base is None:
        d = float(rng.choice([-1, 1]))
        o = _ball(rng)
        h = float(rng.uniform(0.8, 1.6))                # centre height above its resting height
        o.update(x0=-d * 1.6, vx=d * float(rng.uniform(0.1, 0.5)), vz0=float(rng.uniform(-0.5, 1.0)),
                 e=round(float(rng.uniform(0.70, 0.90)), 3), h=h)
    else:
        o = dict(base)
    g = G * gmult
    apex = o["h"] + max(o["vz0"], 0) ** 2 / (2 * g)
    if apex > 1.9:                                      # keep the ball inside the frame
        o["vz0"] = 0.0
    floor_rest = -1.0 + o["size_m"] / 2
    o["z0"] = floor_rest + o["h"]
    return dict(id=sid, family="bounce", gravity=[(0.0, g)], objects=[o],
                gt=dict(vx=abs(o["vx"]), g=g, e=o["e"], gmult=gmult))


def make_ratio(rng, sid):
    # Throws fast enough that the first landing (2 v / g >= 1.2 s) falls after the longest input window
    # (~1.06 s); the wider, higher ratio camera (RATIO_CAMERA) keeps apexes up to ~4.4 m in frame.
    while True:
        slow = rng.uniform(5.9, 7.1)
        fast = slow * rng.uniform(1.3, 1.5)
        if fast <= 9.2:
            break
    v1, v2 = (slow, fast) if rng.random() < 0.5 else (fast, slow)
    size = round(float(rng.uniform(0.25, 0.40)), 3)
    e = round(float(rng.uniform(0.6, 0.8)), 3)
    objs = []
    for x0, v in [(-0.9, v1), (0.9, v2)]:
        o = _ball(rng)
        o.update(size_m=size, x0=x0, vx=0.0, vz0=float(v), e=e, h=0.0, z0=-1.0 + size / 2)
        objs.append(o)
    objs[1]["color"] = next(c for c in COLORS if c != objs[0]["color"])
    return dict(id=sid, family="ratio", gravity=[(0.0, G)], objects=objs,
                gt=dict(t_ratio=float(v1 / v2), h_ratio=float((v1 / v2) ** 2)))


# ---------------------------------------------------------------- text
def _bin(x, edges, words):
    return words[int(np.searchsorted(edges, x))]


SIZE = lambda s: _bin(s, [0.15, 0.22, 0.30], ["the size of a tennis ball", "the size of a grapefruit",
                                              "the size of a football", "the size of a beach ball"])
SPEED = lambda v: _bin(v, [0.3, 0.45, 0.55], ["very slowly", "slowly", "at a gentle walking pace", "at a brisk walking pace"])
HEIGHT = lambda h: _bin(h, [0.9, 1.2, 1.45], ["thigh height", "waist height", "chest height", "shoulder height"])
BOUNCY = lambda e: _bin(e, [0.75, 0.85], ["moderately bouncy", "bouncy", "very bouncy"])
VERB = {"sphere": "rolls", "cube": "slides", "cylinder": "slides"}
NOUN = {"sphere": "ball", "cube": "box", "cylinder": "can"}   # upright cylinder, height = diameter


def caption(scene):
    """B1 caption: what is in the scene, no quantities."""
    o = scene["objects"][0]
    if scene["family"] == "cv":
        return f"A {o['color']} {NOUN[o['shape']]} {VERB[o['shape']]} across a floor. Static camera."
    if scene["family"] == "ratio":
        return f"A {o['color']} ball and a {scene['objects'][1]['color']} ball are thrown straight up from the floor. Static camera."
    return f"A {o['color']} ball bounces on a floor. Static camera."


def describe(scene):
    """B2-nl: the same information in plain words, without numbers."""
    o = scene["objects"][0]
    side = "right" if o["vx"] > 0 else "left"
    if scene["family"] == "cv":
        return (f"A {o['color']} {NOUN[o['shape']]}, {SIZE(o['size_m'])}, {VERB[o['shape']]} {SPEED(abs(o['vx']))} "
                f"to the {side} across a smooth floor, keeping the same speed the whole time. "
                "Seen side-on from a fixed camera.")
    if scene["family"] == "ratio":
        a, b = scene["objects"]
        faster = a if a["vz0"] > b["vz0"] else b
        return (f"Two balls, each {SIZE(a['size_m'])}, are thrown straight up from the floor at the same moment: "
                f"a {a['color']} one on the left and a {b['color']} one on the right. "
                f"The {faster['color']} ball is thrown noticeably harder. Both fall back and bounce. "
                "Seen side-on from a fixed camera.")
    up = " after a small upward toss" if o["vz0"] > 0.3 else ""
    return (f"A {BOUNCY(o['e'])} {o['color']} ball, {SIZE(o['size_m'])}, is released from about {HEIGHT(o['h'])}{up} "
            f"while drifting {SPEED(abs(o['vx']))} to the {side}. It falls under normal Earth gravity "
            "and bounces on a hard floor, each bounce lower than the last. Seen side-on from a fixed camera.")


# ---------------------------------------------------------------- ground-truth export
def finalise(scene, cam, cfg, rng):
    r = cfg["render"]
    scene.update(camera=cam, floor_z=cam["floor_z"], duration_s=r["duration_s"], fps=r["fps"],
                 env_seed=int(rng.integers(1 << 31)))
    t = np.arange(int(round(r["duration_s"] * r["fps"]))) / r["fps"]
    tracks = []
    for o in scene["objects"]:
        x, z = object_track(scene, o, t)
        u, v = project(cam, x, z)
        tracks.append(dict(x_m=x.round(5).tolist(), z_m=z.round(5).tolist(),
                           u_px=u.round(3).tolist(), v_px=v.round(3).tolist(),
                           radius_px=o["size_m"] / 2 * px_per_m(cam)))
        assert np.all((u > 0) & (u < cam["width"]) & (v > 0) & (v < cam["height"])), f"{scene['id']} leaves frame"
    scene["tracks"] = tracks
    scene["caption"] = caption(scene)
    scene["b2nl"] = describe(scene)
    return scene


def _draw(make, rng, sid, cam, cfg):
    """New (phase 2) scenes only: redraw from the same stream if the draw leaves the frame."""
    while True:
        try:
            return finalise(make(rng, sid, cam) if make is make_cv else make(rng, sid), cam, cfg, rng)
        except AssertionError:
            pass


def build(cfg, seed=0, n_cv=15, n_bounce=15, n_grav=10, n_ratio=10, n_more=35):
    # one random stream per family, so changing one family never reshuffles the others
    streams = {f: np.random.default_rng([seed, k]) for k, f in enumerate(("kmain", "kgrav", "kratio", "kcont"))}
    cam = camera(cfg)
    out = {}
    rng = streams["kmain"]
    out["kmain"] = [finalise(make_cv(rng, f"kmain_cv_{i:02d}", cam), cam, cfg, rng) for i in range(n_cv)]
    out["kmain"] += [finalise(make_bounce(rng, f"kmain_ca_{i:02d}"), cam, cfg, rng) for i in range(n_bounce)]
    # phase 2, 2026-10-25: 35 more cv + 35 more ca (ids 15..49) from their own stream, so the pilot's 30 scenes stay identical
    rng = np.random.default_rng([seed, 4])
    for fam, make in (("cv", make_cv), ("ca", make_bounce)):
        for i in range(n_cv if fam == "cv" else n_bounce, (n_cv if fam == "cv" else n_bounce) + n_more):
            out["kmain"].append(_draw(make, rng, f"kmain_{fam}_{i:02d}", cam, cfg))
    rng = streams["kgrav"]
    out["kgrav"] = []
    for i in range(n_grav):
        base = make_bounce(rng, "tmp")["objects"][0]
        # released from rest, high and bouncy: at 2 g the flights still span enough frames at 16 fps
        base.update(vz0=0.0, h=float(rng.uniform(1.2, 1.6)), e=round(float(rng.uniform(0.75, 0.90)), 3))
        env = int(rng.integers(1 << 31))
        for gm in (0.5, 1.0, 2.0):
            s = finalise(make_bounce(rng, f"kgrav_{i:02d}_g{gm}", gm, base), cam, cfg, rng)
            s["env_seed"] = env                          # same look across gravity values
            out["kgrav"].append(s)
    rng = streams["kratio"]
    cam_ratio = camera(cfg, **RATIO_CAMERA)
    out["kratio"] = [finalise(make_ratio(rng, f"kratio_{i:02d}"), cam_ratio, cfg, rng) for i in range(n_ratio)]
    # C2-g: first part at Earth gravity, then switch at the end of the Cosmos input window.
    # Own bases, lower and less bouncy: after a switch to 0.25 g the bounce height is ~ e^2 (h + 1.44 m).
    t_switch = cfg["models"]["cosmos2b"]["cond_frames"] / cfg["models"]["cosmos2b"]["fps"]
    rng = streams["kcont"]
    out["kcont"] = []
    for i in range(n_grav):
        base = make_bounce(rng, "tmp")["objects"][0]
        base.update(vz0=0.0, h=float(rng.uniform(0.8, 1.2)), e=round(float(rng.uniform(0.60, 0.75)), 3))
        env = int(rng.integers(1 << 31))
        for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
            c = make_bounce(rng, f"kcont_{i:02d}_x{k}", 1.0, base)
            c["gravity"] = [(0.0, G), (t_switch, G * k)]
            c["gt"] = dict(c["gt"], g_switch=G * k, t_switch=t_switch, cont_mult=k)
            c = finalise(c, cam, cfg, rng)
            c["env_seed"] = env
            out["kcont"].append(c)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sets = build(cfg, a.seed)
    for name, scenes in sets.items():
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        for s in scenes:
            (d / f"{s['id']}.json").write_text(json.dumps(s))
        print(f"{name}: {len(scenes)} scenes -> {d}")
