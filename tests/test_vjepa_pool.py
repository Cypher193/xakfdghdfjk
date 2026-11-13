import numpy as np
import torch

from pilot.vjepa import VJEPA


class _Stub:
    """Hidden states (embeddings + 2 layers) of a (T=2, 2x2) token grid, D=3: token i of layer L holds L*100 + i."""

    def __call__(self, **kw):
        hs = [torch.arange(8, dtype=torch.float32)[None, :, None].repeat(1, 1, 3) + 100 * L for L in range(3)]
        return type("O", (), {"hidden_states": hs})()


def test_pooled_modes():
    vj = VJEPA.__new__(VJEPA)
    vj.model, vj.device = _Stub(), "cpu"
    vj.pixels = lambda frames: None
    m = np.zeros((2, 2, 2), bool)
    m[0, 0, 0] = m[1, 0, 0] = m[1, 1, 1] = True          # tokens 0 (frame 0); 4 and 7 (frame 1)
    out = vj.pooled(None, {"a": (m, "mean"), "b": (m, "pf"), "c": (m, "flat")})
    assert out["encoder/a"].shape == (2, 1, 3)
    assert np.allclose(out["encoder/a"][:, 0, 0], [100 + 11 / 3, 200 + 11 / 3])
    assert np.allclose(out["encoder/b"][1, 0], [200, 200, 200, 205.5, 205.5, 205.5])
    assert np.allclose(out["vae/c"], [0, 0, 0, 4, 4, 4, 7, 7, 7])
    assert "encoder/c" not in out
