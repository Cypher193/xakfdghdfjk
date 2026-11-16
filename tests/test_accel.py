"""CPU check of the acceleration family: independent draws, and the readout recovers |ax| and its sign from exact
tracks with 2 px of noise at 16 and 24 fps.    python -m tests.test_accel"""
import numpy as np
import yaml

from pilot import metrics, scenes
from pilot.track import readout

cfg = yaml.safe_load(open("configs/pilot.yaml"))
k = scenes.build(cfg)["kacc"]
assert len(k) == 100 and {s["gt"]["scenario"] for s in k} == set(scenes.ACC_SCENARIOS)
ax, v0 = np.array([s["gt"]["ax"] for s in k]), np.array([abs(s["gt"]["v0"]) for s in k])
assert abs(np.corrcoef(ax, v0)[0, 1]) < 0.2, "start speed predicts acceleration"
rng = np.random.default_rng(0)
mra = {}
for s in k:
    edge = s["camera"]["cx"] / scenes.px_per_m(s["camera"])
    for fps in (16, 24):
        t = np.arange(0, 5.8, 1 / fps)
        x, z = scenes.object_track(s, s["objects"][0], t)
        r = readout("accel", t, [x + rng.normal(0, 0.01, len(x))], [z], 0.3125, x_edge=edge)
        mra.setdefault(s["gt"]["scenario"], []).append(metrics.sym_mra(r["ax"], s["gt"]["ax"]))
        assert r["speeding"] == s["gt"]["speeding"], s["id"]
assert all(np.mean(v) >= 0.95 for v in mra.values()), {k: np.mean(v) for k, v in mra.items()}
print("acceleration family checks passed", {k: round(float(np.mean(v)), 3) for k, v in mra.items()})

# exit_frame: an object that slides out of the right edge is scored up to the edge, not counted as vanished
from pilot.track import exit_frame, measurable
n, w = 140, 854
cx = np.linspace(400, 1000, n)
area = np.where(cx < w - 10, 700.0, 0.0)
c = np.c_[cx, np.full(n, 300.0)]
e = exit_frame(c, area, w, 5)
assert 0 < e < n and cx[e] > w - 1.2 * np.sqrt(700 / np.pi) - 15 and measurable(area[:e], 5) and not measurable(area, 5), e
assert exit_frame(np.c_[np.full(n, 400.0), np.full(n, 300.0)], np.full(n, 700.0), w, 5) == n
print("exit_frame checks passed")
