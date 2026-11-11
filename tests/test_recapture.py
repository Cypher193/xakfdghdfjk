"""CPU check of the red-team re-capture (pilot.recapture): extra token pools, acts_only VAE pools, random init.
    python -m tests.test_recapture"""
import numpy as np
import torch

from pilot.models import Generator, token_grid
from pilot.recapture import box_mask, cell_mask, randomise
from tests import tiny

RUN = {"capture_fractions": [0.25]}
BUILD = {**tiny.BUILDERS, **tiny.SCREEN_BUILDERS}
for name in ("ltx2b", "cosmos2b", "vace13", "cosmos3_nano"):
    pipe, cfg = BUILD[name]()
    gen = Generator(name, cfg, RUN, device="cpu", pipe=pipe, dtype=torch.float32)
    fr, wh, box = tiny.frames(cfg["cond_frames"]), (64, 48), [10, 10, 30, 30]
    n_lat = 1 + (cfg["cond_frames"] - 1) // cfg["vae_stride"][0]
    tg = token_grid(cfg)
    vg = (n_lat, cfg["height"] // cfg["vae_stride"][1], cfg["width"] // cfg["vae_stride"][2])
    tok = {"fix": cell_mask(tg, n_lat, (20, 20), (20, 20), wh), "fix_pf": cell_mask(tg, n_lat, (20, 20), (20, 20), wh),
           "f0": cell_mask(tg, n_lat, (20, 20), (8, 8), wh, frame0_only=True)}
    fv = cell_mask(vg, n_lat, (20, 20), (20, 20), wh)
    vae = {"obj": (box_mask(vg, n_lat, box, wh), "mean"), "fix": (fv, "mean"), "fix_pf": (fv, "pf"), "fix_flat": (fv, "flat")}
    full = gen.generate(fr, "a ball", 0, box, wh)
    r = gen.generate(fr, "a ball", 0, box, wh, acts_only=True, masks=tok, vae_pools=vae)
    g0 = cfg["blocks"][0]
    D = r["acts"][f"{g0}/obj"].shape[-1]
    assert r["acts"][f"{g0}/fix"].shape[-1] == D and r["acts"][f"{g0}/f0"].shape[-1] == D, name
    assert r["acts"][f"{g0}/fix_pf"].shape[-1] == D * int(tok["fix_pf"].any((1, 2)).sum()), name
    # acts_only follows the same trajectory: the obj pool matches a full run at the captured step
    assert np.allclose(r["acts"][f"{g0}/obj"], full["acts"][f"{g0}/obj"], atol=1e-3), name
    C = r["vae"]["fix"].shape[0]
    assert r["vae"]["fix_pf"].shape[0] == C * n_lat and r["vae"]["fix_flat"].shape[0] == C * int(fv.sum()), name
    # constant cell count wherever the centre is (shifted, never clipped)
    assert cell_mask(tg, n_lat, (1, 1), (20, 20), wh).sum() == cell_mask(tg, n_lat, (60, 40), (20, 20), wh).sum(), name
    print(name, {k: v.shape for k, v in r["acts"].items() if k != "steps"}, {k: v.shape for k, v in r["vae"].items()})
assert randomise(pipe.transformer) > 0
print("recapture checks passed")
