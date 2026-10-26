"""CPU check: the erase hook changes exactly the object tokens by r(h) - h, and a generate call with it runs.  python -m tests.test_leace"""
import numpy as np
import torch

from pilot import erase
from pilot.leace import check_ok, erase_hook, rand_ok
from pilot.models import Generator, object_token_mask, token_grid
from tests import tiny

# check targets: res[cond, step] = clean, (a), (b), pert; rand checked only where clean >= 0.3, at min(0.6, 0.75 clean)
r = lambda *rows: np.array([[[0, 0, 0, 0]] * len(rows), rows], float)
assert rand_ok(r([0.8, 0.65, 0.6, 0.2], [0.55, 0.42, 0.42, 0.2], [0.2, 0.0, -0.1, 0.2]))
assert not rand_ok(r([0.8, 0.59, 0.7, 0.2])) and not rand_ok(r([0.5, 0.37, 0.45, 0.2]))
res = lambda lea, rnd: np.array([[lea], [rnd]], float)
assert check_ok(res([0.7, 0.05, -0.04, 0.11], [0.7, 0.65, 0.62, 0.24]))            # small leace edit: fine
assert not check_ok(res([0.7, 0.07, -0.04, 0.11], [0.7, 0.65, 0.62, 0.24]))        # leace leaves speed
assert not check_ok(res([0.7, 0.05, -0.04, 0.25], [0.7, 0.65, 0.62, 0.12]))        # leace edit > 2x rand

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
