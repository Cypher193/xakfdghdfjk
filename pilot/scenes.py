"""Pilot scene specs with exact, analytic ground truth.

Objects move in the plane y = 0, facing a fixed pinhole camera, so pixels convert to metres
with one scale (the QuantiPhy assumption). Motion is closed-form (constant velocity, or
constant gravity with restitution bounces), so the renderer just keyframes positions and
the ground truth is exact. Gravity can switch value at one time (used for C2-g continuations).

    python -m pilot.scenes --out data/scenes
"""
import argparse
import copy
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


def ramp_foot(scene, obj):
    """Slide of a ramp object -> (acceleration along the slope, time at the foot, speed at the foot)."""
    th, g = math.radians(obj["theta_deg"]), scene["gravity"][0][1]
    a = g * (math.sin(th) - obj["mu"] * math.cos(th))
    t_foot = math.sqrt(2 * obj["ramp_len"] / a)
    return a, t_foot, a * t_foot


def ramp_track(scene, obj, t):
    """Box centre: slides from rest along the ramp, then brakes on the floor with the same friction. The kink at the foot
    removes the normal component, so the box keeps only its horizontal speed."""
    th, g = math.radians(obj["theta_deg"]), scene["gravity"][0][1]
    a, t_foot, v_foot = ramp_foot(scene, obj)
    v_floor = v_foot * math.cos(th)
    t = np.asarray(t, float)
    s = 0.5 * a * np.minimum(t, t_foot) ** 2
    tau = np.clip(t - t_foot, 0.0, v_floor / (obj["mu"] * g))
    x = obj["x0"] + obj["dir"] * (s * math.cos(th) + v_floor * tau - 0.5 * obj["mu"] * g * tau ** 2)
    return x, obj["z0"] - s * math.sin(th)


def object_track(scene, obj, t):
    """Centre (x, z) in metres at times t. scene["vswitch"] = [t_s, k]: the velocity is multiplied by k at t_s (C2).
    scene["vturn"] = [t_s, deg]: the velocity turns by deg at t_s, upward relative to the direction of travel
    (180 = reversed), same speed; floor objects then glide in a straight line, balls stay ballistic (C2-turn)."""
    if obj["motion"] == "ramp":
        return ramp_track(scene, obj, t)
    t = np.asarray(t, float)
    floor = scene["floor_z"] + obj["size_m"] / 2
    if "vturn" in scene:
        return _turned_track(scene, obj, t, floor)
    ts, k = scene.get("vswitch", (math.inf, 1.0))
    x = obj["x0"] + obj["vx"] * (np.minimum(t, ts) + k * np.maximum(t - ts, 0.0))
    if obj["motion"] == "floor":
        if "t_c" in obj:                       # collision: new constant velocity from t_c on
            x = np.where(t < obj["t_c"], x, obj["x0"] + obj["vx"] * obj["t_c"] + obj["v_after"] * (t - obj["t_c"]))
        return x, np.full_like(x, floor)
    segs, _ = vertical_segments(obj["z0"], obj["vz0"], scene["gravity"], obj["e"], floor, scene["duration_s"])
    if ts == math.inf:
        return x, eval_segments(segs, t)
    z1, vz1 = _vertical_state(scene, segs, ts)
    late, _ = vertical_segments(z1, k * vz1, scene["gravity"], obj["e"], floor, scene["duration_s"] - ts)
    return x, np.where(t < ts, eval_segments(segs, t), eval_segments(late, t - ts))


def _vertical_state(scene, segs, ts):
    assert len(scene["gravity"]) == 1, "a switch restarts the vertical motion under one gravity"
    s = next(s for s in segs if s[0] <= ts <= s[1])
    return eval_segments([s], ts)[0], s[3] - s[4] * (ts - s[0])


def _turned_track(scene, obj, t, floor):
    assert "t_c" not in obj
    ts, deg = scene["vturn"]
    plain = {f: v for f, v in scene.items() if f != "vturn"}
    x, z = object_track(plain, obj, t)
    (xs,), (zs,) = object_track(plain, obj, [ts])
    vz = 0.0
    if obj["motion"] != "floor":
        segs, _ = vertical_segments(obj["z0"], obj["vz0"], scene["gravity"], obj["e"], floor, scene["duration_s"])
        vz = _vertical_state(scene, segs, ts)[1]
    phi = math.radians(deg) * math.copysign(1.0, obj["vx"])          # counter-clockwise turns a rightward mover up
    vx2, vz2 = obj["vx"] * math.cos(phi) - vz * math.sin(phi), obj["vx"] * math.sin(phi) + vz * math.cos(phi)
    late = np.maximum(t - ts, 0.0)
    if obj["motion"] == "floor":
        z2 = zs + vz2 * late
    else:
        segs, _ = vertical_segments(zs, vz2, scene["gravity"], obj["e"], floor, scene["duration_s"] - ts)
        z2 = eval_segments(segs, late)
    return np.where(t < ts, x, xs + vx2 * late), np.where(t < ts, z, z2)


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


def make_ratio(rng, sid, size_m=(0.25, 0.40), vmax=9.2):
    # Throws fast enough that the first landing (2 v / g >= 1.2 s) falls after the longest input window
    # (~1.06 s); the wider, higher ratio camera (RATIO_CAMERA) keeps apexes up to ~4.4 m in frame.
    while True:
        slow = rng.uniform(5.9, 7.1)
        fast = slow * rng.uniform(1.3, 1.5)
        if fast <= vmax:
            break
    v1, v2 = (slow, fast) if rng.random() < 0.5 else (fast, slow)
    size = round(float(rng.uniform(*size_m)), 3)
    e = round(float(rng.uniform(0.6, 0.8)), 3)
    objs = []
    for x0, v in [(-0.9, v1), (0.9, v2)]:
        o = _ball(rng)
        o.update(size_m=size, x0=x0, vx=0.0, vz0=float(v), e=e, h=0.0, z0=-1.0 + size / 2)
        objs.append(o)
    objs[1]["color"] = next(c for c in COLORS if c != objs[0]["color"])
    return dict(id=sid, family="ratio", gravity=[(0.0, G)], objects=objs,
                gt=dict(t_ratio=float(v1 / v2), h_ratio=float((v1 / v2) ** 2)))


def make_collide(rng, sid):
    """L3, momentum + restitution: a rolling ball hits a resting one after the longest input window. Same density, so
    mass goes with size^3. gt v = the struck ball's speed, the number the input window does not show."""
    d = float(rng.choice([-1, 1]))
    a, b = _ball(rng, "floor"), _ball(rng, "floor")
    b["color"] = next(c for c in rng.permutation(COLORS).tolist() if c != a["color"])
    sa, sb = round(float(rng.uniform(0.25, 0.40)), 3), round(float(rng.uniform(0.25, 0.40)), 3)
    v, e, t_c = float(rng.uniform(0.5, 0.9)), round(float(rng.uniform(0.6, 0.9)), 3), float(rng.uniform(1.3, 2.0))
    ma, mb = sa ** 3, sb ** 3
    va, vb = d * v * (ma - e * mb) / (ma + mb), d * v * (1 + e) * ma / (ma + mb)
    x0 = -d * 1.9
    a.update(size_m=sa, x0=x0, vx=d * v, z0=None, vz0=0.0, e=None, t_c=t_c, v_after=va)
    b.update(size_m=sb, x0=x0 + d * (v * t_c + (sa + sb) / 2), vx=0.0, z0=None, vz0=0.0, e=None, t_c=t_c, v_after=vb)
    return dict(id=sid, family="collide", gravity=[(0.0, G)], objects=[a, b], gt=dict(v=abs(vb), v0=v, e=e, t_c=t_c))


def make_ramp(rng, sid):
    """L3, gravity + sliding friction: a box released from rest on a rough ramp, past the input window on the slope.
    gt a_slope = horizontal acceleration on the ramp, what the tracker reads from x(t)."""
    d = float(rng.choice([-1, 1]))
    size = round(float(rng.uniform(0.30, 0.40)), 3)
    theta, mu, length = float(rng.uniform(20, 28)), float(rng.uniform(0.2, 0.35)), float(rng.uniform(1.8, 2.6))
    o = dict(shape="cube", motion="ramp", color=str(rng.choice(COLORS)), size_m=size, x0=-d * 2.0, vx=0.0,
             z0=-1.0 + size / 2 + length * math.sin(math.radians(theta)), vz0=0.0, e=None, theta_deg=theta, mu=mu,
             ramp_len=length, dir=d)
    sc = dict(id=sid, family="ramp", gravity=[(0.0, G)], objects=[o])
    assert math.sin(math.radians(theta)) > mu * math.cos(math.radians(theta)) + 0.08, "box barely moves"
    a, t_foot, _ = ramp_foot(sc, o)
    assert t_foot >= 1.6, "too short on the slope"
    sc["gt"] = dict(a_slope=a * math.cos(math.radians(theta)), mu=mu, theta_deg=theta, t_foot=t_foot)
    return sc


# ---------------------------------------------------------------- text
def _bin(x, edges, words):
    return words[int(np.searchsorted(edges, x))]


SIZE = lambda s: _bin(s, [0.15, 0.22, 0.30], ["the size of a tennis ball", "the size of a grapefruit",
                                              "the size of a football", "the size of a beach ball"])
SPEED = lambda v: _bin(v, [0.3, 0.45, 0.55], ["very slowly", "slowly", "at a gentle walking pace", "at a brisk walking pace"])
HEIGHT = lambda h: _bin(h, [0.9, 1.2, 1.45], ["thigh height", "waist height", "chest height", "shoulder height"])
BOUNCY = lambda e: _bin(e, [0.75, 0.85], ["moderately bouncy", "bouncy", "very bouncy"])
STEEP = lambda th: _bin(th, [22.5, 25.5], ["gentle", "moderate", "steep"])
ROUGH = lambda mu: _bin(mu, [0.25, 0.30], ["slightly rough", "rough", "very rough"])
VERB = {"sphere": "rolls", "cube": "slides", "cylinder": "slides"}
NOUN = {"sphere": "ball", "cube": "box", "cylinder": "can"}   # upright cylinder, height = diameter


def caption(scene):
    """B1 caption: what is in the scene, no quantities."""
    o = scene["objects"][0]
    if scene["family"] == "cv":
        return f"A {o['color']} {NOUN[o['shape']]} {VERB[o['shape']]} across a floor. Static camera."
    if scene["family"] == "ratio":
        return f"A {o['color']} ball and a {scene['objects'][1]['color']} ball are thrown straight up from the floor. Static camera."
    if scene["family"] == "collide":
        return f"A {o['color']} ball rolls across a floor towards a {scene['objects'][1]['color']} ball. Static camera."
    if scene["family"] == "ramp":
        return f"A {o['color']} box slides down a ramp onto a floor. Static camera."
    return f"A {o['color']} ball bounces on a floor. Static camera."


def describe(scene):
    """B2-nl: the same information in plain words, without numbers."""
    o = scene["objects"][0]
    if scene["family"] == "ramp":
        return (f"A {o['color']} box, {SIZE(o['size_m'])}, is released from rest at the top of a {STEEP(o['theta_deg'])}, "
                f"{ROUGH(o['mu'])} ramp and slides down to the {'right' if o['dir'] > 0 else 'left'}, speeding up. "
                "At the bottom it slides on across the floor and friction brings it to a stop. Seen side-on from a fixed camera.")
    side = "right" if o["vx"] > 0 else "left"
    if scene["family"] == "collide":
        b = scene["objects"][1]
        return (f"A {o['color']} ball, {SIZE(o['size_m'])}, rolls {SPEED(abs(o['vx']))} to the {side} across a smooth floor "
                f"and hits a {b['color']} ball, {SIZE(b['size_m'])}, resting in its path. The balls are "
                f"{BOUNCY(scene['gt']['e'])}, and the {b['color']} ball rolls away after the hit. Seen side-on from a fixed camera.")
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


def describe_num(scene):
    """B2-num: the same scene in numbers (metres, seconds). Cv and bounce only: the 60 B2-num scenes are kmain."""
    o = scene["objects"][0]
    side = "right" if o["vx"] > 0 else "left"
    if scene["family"] == "cv":
        return (f"A {o['color']} {NOUN[o['shape']]}, {o['size_m']:.2f} m wide, {VERB[o['shape']]} to the {side} across a "
                f"smooth floor at a constant {abs(o['vx']):.2f} m/s: x = {o['x0']:.2f} m at t = 0 s, x = {o['x0'] + o['vx']:.2f} m "
                "at t = 1 s. Seen side-on from a fixed camera.")
    return (f"A {o['color']} ball, {o['size_m']:.2f} m wide, is released {o['h']:.2f} m above its resting height with vertical "
            f"speed {o['vz0']:+.2f} m/s while drifting to the {side} at {abs(o['vx']):.2f} m/s. It falls with gravity "
            f"{scene['gravity'][0][1]:.2f} m/s^2 and bounces on a hard floor with restitution {o['e']:.2f}. "
            "Seen side-on from a fixed camera.")


# ---------------------------------------------------------------- ground-truth export
def set_tracks(scene, cam, cfg, strict=True):
    """Exact tracks at the render frame rate; strict: the object must stay in frame."""
    r = cfg["render"]
    t = np.arange(int(round(r["duration_s"] * r["fps"]))) / r["fps"]
    tracks = []
    for o in scene["objects"]:
        x, z = object_track(scene, o, t)
        u, v = project(cam, x, z)
        tracks.append(dict(x_m=x.round(5).tolist(), z_m=z.round(5).tolist(),
                           u_px=u.round(3).tolist(), v_px=v.round(3).tolist(),
                           radius_px=o["size_m"] / 2 * px_per_m(cam)))
        assert not strict or np.all((u > 0) & (u < cam["width"]) & (v > 0) & (v < cam["height"])), \
            f"{scene['id']} leaves frame"
    scene["tracks"] = tracks


def finalise(scene, cam, cfg, rng):
    r = cfg["render"]
    scene.update(camera=cam, floor_z=cam["floor_z"], duration_s=r["duration_s"], fps=r["fps"],
                 env_seed=int(rng.integers(1 << 31)))
    set_tracks(scene, cam, cfg)
    scene["caption"] = caption(scene)
    scene["b2nl"] = describe(scene)
    if scene["family"] in ("cv", "bounce"):
        scene["b2num"] = describe_num(scene)
    return scene


def _draw(make, rng, sid, cam, cfg):
    """New (phase 2) scenes only: redraw from the same stream if the draw leaves the frame."""
    while True:
        try:
            return finalise(make(rng, sid, cam) if make is make_cv else make(rng, sid), cam, cfg, rng)
        except AssertionError:
            pass


def build(cfg, seed=0, n_cv=15, n_bounce=15, n_grav=10, n_ratio=10, n_more=35, n_grav_all=30, n_ratio2=40, n_l3=15):
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
    out["kgrav"] = []
    for i in range(n_grav_all):
        # 2026-11-01: 20 more bases (ids 10..29) from their own stream, so the pilot's 30 scenes stay identical
        rng = streams["kgrav"] if i < n_grav else np.random.default_rng([seed, 6, i])
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
    # kratio2 (2026-10-26): kratio balls are 25-41 px, about one 32 px LTX latent cell, and LTX smears or drops them
    # (5/30 measurable). 0.7-0.9 m balls are 71-92 px; vmax 8.5 m/s keeps the whole ball in frame at the apex.
    rng = np.random.default_rng([seed, 5])
    big = lambda r, sid: make_ratio(r, sid, size_m=(0.7, 0.9), vmax=8.5)
    out["kratio2"] = [_draw(big, rng, f"kratio2_{i:02d}", cam_ratio, cfg) for i in range(n_ratio2)]   # first 10: 2026-10-26
    # B-L3 (2026-11-01): two laws combined, 15 collisions + 15 ramps with friction, own stream
    rng = np.random.default_rng([seed, 7])
    out["kl3"] = [_draw(make_collide, rng, f"kl3_co_{i:02d}", cam, cfg) for i in range(n_l3)]
    out["kl3"] += [_draw(make_ramp, rng, f"kl3_ra_{i:02d}", cam, cfg) for i in range(n_l3)]
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
    # C2 (2026-11-03): every kmain scene with its velocity scaled from the end of a model's input window on, no new draws.
    # 2026-11-04: VACE's window (kvelv), the same scenes turned instead of scaled (kturn*), kcont at LTX's window (kcontl).
    # 2026-11-06: VACE's turn and gravity-switch sets (kturnv, kcontv), then Nano's (kturnn, kcontn).
    for name, model in VEL_SETS.items():
        out[name] = switch_copies(out["kmain"], name, cfg["models"][model], cam, cfg, "vswitch", "v_scale", V_SCALES)
    for name, model in TURN_SETS.items():
        c = switch_copies(out["kmain"], name, cfg["models"][model], cam, cfg, "vturn", "turn_deg", TURN_DEGS)
        # a ball falls at ~3 m/s at the turn: 30 degrees flings it sideways out of frame (in frame 0.31), so cv only
        out[name] = [s for s in c if not (s["family"] != "cv" and s["gt"]["turn_deg"] == 30)]
    for name, model in CONT_SETS.items():     # kcont with the switch at another model's window end
        mc = cfg["models"][model]
        t_l = mc["cond_frames"] / mc["fps"]
        out[name] = []
        for s in out["kcont"]:
            c = copy.deepcopy(s)
            c.update(id=s["id"].replace("kcont", name), gravity=[(0.0, G), (t_l, c["gravity"][1][1])])
            c["gt"] = dict(c["gt"], t_switch=t_l, in_frame=_in_frame(c, mc, cam))
            set_tracks(c, cam, cfg, strict=False)
            out[name].append(c)
    return out


VEL_SETS = {"kvelc": "cosmos2b", "kvell": "ltx2b", "kvelv": "vace13", "kveln": "cosmos3_nano"}   # C2 scene set: model whose window and clip it uses
V_SCALES = (0.5, 0.8, 1.0, 1.25, 2.0)
TURN_SETS = {"kturnc": "cosmos2b", "kturnl": "ltx2b", "kturnv": "vace13", "kturnn": "cosmos3_nano"}
CONT_SETS = {"kcontl": "ltx2b", "kcontv": "vace13", "kcontn": "cosmos3_nano"}
TURN_DEGS = (0, 30, 180)


def _in_frame(scene, mcfg, cam):
    """Share of the model's clip frames with the object centre in frame."""
    u, v = project(cam, *object_track(scene, scene["objects"][0], np.arange(mcfg["num_frames"]) / mcfg["fps"]))
    return float(np.mean((u > 0) & (u < cam["width"]) & (v > 0) & (v < cam["height"])))


def switch_copies(main, name, mcfg, cam, cfg, key, gt_key, values):
    """Copies of the kmain scenes, same look, changed from the end of the model's input window on; ids kvelc_cv_00_x0.5.
    key "vswitch" scales the velocity, "vturn" turns it (object_track). Copies may leave the frame (gt in_frame)."""
    t_switch = mcfg["cond_frames"] / mcfg["fps"]
    out = []
    for m in main:
        for k in values:
            c = {f: v for f, v in copy.deepcopy(m).items() if f not in ("b2nl", "b2num")}
            c.update({"id": m["id"].replace("kmain", name) + f"_x{k}", key: [t_switch, k]})
            if key == "vturn":        # centred at the turn: a reversed copy has as much room as the true one
                c["objects"][0]["x0"] = -c["objects"][0]["vx"] * t_switch
            c["gt"] = dict(m["gt"], **{gt_key: k}, t_switch=t_switch)
            set_tracks(c, cam, cfg, strict=False)
            c["gt"]["in_frame"] = _in_frame(c, mcfg, cam)
            out.append(c)
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
