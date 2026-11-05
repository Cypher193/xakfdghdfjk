"""CPU check of pilot.vjepa on a tiny random V-JEPA 2: acts in the generators' layout (pilot.probe reads them), the
context never sees the future, surprise rows in pilot.likelihood's layout. Needs data/scenes.  python -m tests.test_vjepa"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import VJEPA2Config, VJEPA2Model

from pilot import probe
from pilot.vjepa import N_FRAMES, VJEPA, run_surprise
from tests import tiny

torch.manual_seed(0)
cfg = VJEPA2Config(crop_size=32, patch_size=16, tubelet_size=2, frames_per_clip=N_FRAMES, hidden_size=32, num_hidden_layers=2,
                   num_attention_heads=2, pred_hidden_size=16, pred_num_hidden_layers=1, pred_num_attention_heads=2)
vj = VJEPA(VJEPA2Model(cfg).float(), [0.5] * 3, [0.25] * 3, "cpu")
f = tiny.frames(N_FRAMES, wh=(64, 48), seed=1)

a = vj.acts(f[:16], (10, 10, 30, 30), (64, 48))
assert a["encoder/obj"].shape == a["encoder/all"].shape == (2, 1, 32) and a["vae/obj"].shape == (32,)
assert a["predictor/all"].shape == (1, 1, 32) and all(np.isfinite(v).all() for v in a.values())
assert not np.allclose(a["encoder/obj"], a["encoder/all"])

# the context is encoded alone: changing the future leaves the predicted tokens unchanged, but not the surprise
g = f[:16] + tiny.frames(N_FRAMES - 16, wh=(64, 48), seed=2)
p1 = vj.predict(vj.model.encoder(pixel_values_videos=vj.pixels(f[:16])).last_hidden_state, 16, N_FRAMES)
p2 = vj.predict(vj.model.encoder(pixel_values_videos=vj.pixels(g[:16])).last_hidden_state, 16, N_FRAMES)
assert torch.equal(p1, p2) and p1.shape == (1, (N_FRAMES - 16) // 2 * 4, 32)
assert vj.surprise(f, 16) != vj.surprise(g, 16)

tmp = Path("tests/_vjepa_tmp")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
out = tmp / "c2_kturnl_tiny.csv"
frames = lambda spec: tiny.frames(N_FRAMES, wh=(64, 48), seed=int(spec["gt"]["turn_deg"]))
df, summ = run_surprise(vj, "data/scenes/kturnl", None, out, 24, shard="1/25", frames_fn=frames)
n = len(df)
df, summ = run_surprise(vj, "data/scenes/kturnl", None, out, 24, shard="1/25", frames_fn=frames)
assert len(df) == n and df.true.sum() == len(summ) == 4 and np.isfinite(df.loss).all() and summ.k_implied.isna().all()

# acts.npz written as pilot.generate does: pilot.probe loads them
for i, spec in enumerate(sorted(Path("data/scenes/kmain").glob("*.json"))[:12]):
    sid = json.loads(spec.read_text(encoding="utf-8"))["id"]
    d = tmp / "runs" / "b1" / "tiny" / f"{sid}_s0"
    d.mkdir(parents=True)
    np.savez_compressed(d / "acts.npz", **vj.acts(tiny.frames(16, wh=(64, 48), seed=i), (10, 10, 30, 30), (64, 48)))
    (d / "meta.json").write_text(json.dumps(dict(scene=sid)))
feats, y, grp = probe.load(tmp / "runs", "tiny", ["b1"], "hspeed")
assert feats["encoder/obj"].shape == (2, 1, 12, 32) and feats["vae/obj"].shape == (12, 32) and len(y) == 12
shutil.rmtree(tmp)
print("vjepa checks passed")
