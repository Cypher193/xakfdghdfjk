"""CPU check: the erase hook changes exactly the object tokens by r(h) - h, and a generate call with it runs.  python -m tests.test_leace"""
import numpy as np
import torch

from pilot import erase
from pilot.leace import erase_hook
from pilot.models import Generator, object_token_mask, token_grid
from tests import tiny

RUN = {"capture_fractions": [0.25, 0.5, 0.75]}
for name, build in tiny.BUILDERS.items():
    pipe, cfg = build()
    gen = Generator(name, cfg, RUN, device="cpu", pipe=pipe, dtype=torch.float32)
    fr, box = tiny.frames(cfg["cond_frames"]), [10, 10, 30, 30]
    idx = torch.from_numpy(np.flatnonzero(object_token_mask(cfg, token_grid(cfg), box, (64, 48))))
    D = gen.generate(fr, "a ball", 0, box, (64, 48))["acts"][f"{cfg['blocks'][0]}/all"].shape[-1]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, D)); er = erase.fit(X, X @ rng.normal(size=D) + rng.normal(size=200))
    h = torch.randn(2, 5 * len(idx) + 7, D)
    if h.shape[1] <= int(idx.max()):
        h = torch.randn(2, int(idx.max()) + 1, D)
    out = erase_hook(er, idx)(None, None, (h, "extra"))
    want = h.clone(); want[:, idx] = torch.from_numpy(er.apply(h[:, idx].numpy().astype(np.float64))).float()
    rest = torch.ones(h.shape[1], dtype=bool); rest[idx] = False
    assert torch.allclose(out[0], want, atol=1e-4) and out[1] == "extra" and (out[0][:, idx] != h[:, idx]).any(), name
    assert torch.equal(out[0][:, rest], h[:, rest]), name
    vid = gen.generate(fr, "a ball", 0, box, (64, 48), hooks=[(gen.blocks()[0], erase_hook(er, idx))])
    assert vid["acts"] and len(vid["video"]) == cfg["num_frames"], name
    print(name, "ok")
print("leace checks passed")
