"""CPU check of the C9c chain (pilot.leace.chain) on the tiny LTX: upstream erasers change what the next fit sees, a
crashed chain resumes without redoing finished clips or groups and ends where an uninterrupted one does, and
chain.json / load_chain record the chain.  python -m tests.test_c9c"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from pilot import erase
from pilot.leace import chain, load_chain, rand_kept
from pilot.models import Generator
from tests import tiny

pipe, cfg = tiny.ltx()
gen = Generator("ltx2b", cfg, {"capture_fractions": [0.25, 0.5, 0.75]}, device="cpu", pipe=pipe, dtype=torch.float32)
scenes = [dict(id=f"t{i}", caption="a ball", camera=dict(width=64, height=48), gt=dict(v=1.0 + i)) for i in range(8)]
inputs = lambda s: (tiny.frames(cfg["cond_frames"], seed=int(s["id"][1:])), [10, 10, 30, 30])
groups = [[0], [1]]
real = gen.generate
accept = lambda res: True              # tiny random weights cannot meet the real targets


def run(root, crash_after=None):
    """chain() with gen.generate counting its calls (hooked block indices) and dying after crash_after of them."""
    calls = []

    def counted(*a, **k):
        if len(calls) == crash_after:
            raise RuntimeError("pod died")
        calls.append(sorted(list(gen.blocks()).index(m) for m, _ in k["hooks"]))
        return real(*a, **k)
    gen.generate = counted
    try:
        return chain(gen, root, scenes, inputs, groups, "leace", ok_fn=accept), calls
    finally:
        gen.generate = real


with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "leace"
    try:
        run(root, crash_after=8 + 3)
        raise AssertionError("no crash")
    except RuntimeError:
        pass
    assert (root / "g0.npz").exists() and not (root / "g1.npz").exists() and len(list((root / "g1").glob("*.npy"))) == 3
    assert not (root / "chain.json").exists()
    ers, calls = run(root)
    assert calls == [[0]] * 5, calls            # resumed: only the 5 missing group-1 clips, each with block 0's eraser
    again, calls = run(root)
    assert calls == [] and sorted(again) == [0, 1]                                 # finished: nothing redone
    fresh, _ = run(Path(d) / "fresh")
    for b in (0, 1):
        assert np.allclose(fresh[b].U, ers[b].U) and np.allclose(fresh[b].mu, ers[b].mu), b   # same end state

    # upstream erasers change the activations the next fit sees: block-1 acts with block 0 erased != without
    s = scenes[0]
    frames, box = inputs(s)
    clean = real(frames, "a ball", 0, box, (64, 48), acts_only=True)["acts"]["transformer_blocks/obj"][1]
    hooked = np.load(root / "g1" / "t0.npy")[0]
    assert hooked.shape == clean.shape == (3, 32) and np.abs(hooked - clean).max() > 1e-3, np.abs(hooked - clean).max()
    A1 = np.stack([np.load(root / "g1" / f"{x['id']}.npy")[0] for x in scenes])        # (n, T, D): what g1 was fit on
    assert np.allclose(ers[1].mu, A1.astype(np.float64).transpose(1, 0, 2).reshape(-1, 32).mean(0), atol=1e-6)

    info = json.loads((root / "chain.json").read_text())
    assert info["groups"] == groups and info["fit_scenes"] == [x["id"] for x in scenes] and info["cond"] == "leace"
    assert sorted(info["checks"]) == ["0", "1"] and np.array(info["checks"]["1"]).shape == (2, 3, 4)
    loaded, meta = load_chain(root)
    assert sorted(loaded) == [0, 1] and meta["groups"] == groups and meta["file"].endswith("leace")
    (root / "chain.json").unlink()                     # a chain that stopped after group 0: its first group still loads
    part, meta = load_chain(root, 1)
    assert sorted(part) == [0] and meta["groups"] == groups[:1] and meta["n_groups"] == 1
    # rand chain on the same group-0 acts (main() seeds g0 from B1 the same way): fit to permuted z, not to z
    (Path(d) / "rand" / "g0").mkdir(parents=True)
    for f in (root / "g0").glob("*.npy"):
        shutil.copy(f, Path(d) / "rand" / "g0" / f.name)
    calls = []
    gen.generate = lambda *a, **k: calls.append(1)
    rnd = chain(gen, Path(d) / "rand", scenes, inputs, groups[:1], "rand", ok_fn=accept)
    gen.generate = real
    X0 = np.stack([np.load(Path(d) / "rand" / "g0" / f"{x['id']}.npy")[0] for x in scenes], 1).astype(np.float64)
    want = erase.random_like(X0.reshape(-1, 32), np.tile(np.log([x["gt"]["v"] for x in scenes]), 3), 0,
                             np.repeat(np.arange(3), len(scenes)))
    assert calls == [] and np.allclose(rnd[0].U, want.U) and not np.allclose(rnd[0].U, ers[0].U)
    assert rand_kept(np.array([[[0, 0, 0, 0.1]], [[0.6, 0.5, 0.46, 0.2]]])) and \
        not rand_kept(np.array([[[0, 0, 0, 0.1]], [[0.6, 0.5, 0.46, 0.31]]]))     # rand perturbation <= 0.3
    try:
        load_chain(Path(d) / "missing" / "rand")
        raise AssertionError("missing chain accepted")
    except SystemExit as e:
        assert "c9cchain-rand" in str(e), e
print("c9c chain checks passed")
