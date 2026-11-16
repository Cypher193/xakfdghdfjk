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


def test_videomae_input_is_patch_embedding_without_position():
    """A tiny random VideoMAE: the vae/* readout must equal the patch projection (no sinusoidal positions), and the
    encoder pools must come from the layer outputs (hidden states 1..L)."""
    from transformers import VideoMAEConfig, VideoMAEModel

    from pilot.vjepa import VideoMAE
    cfg = VideoMAEConfig(image_size=32, patch_size=16, num_frames=4, tubelet_size=2, hidden_size=8, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=16)
    torch.manual_seed(0)
    vm = VideoMAE(VideoMAEModel(cfg).eval(), [0.5] * 3, [0.25] * 3, "cpu")
    frames = [np.random.default_rng(i).integers(0, 255, (24, 40, 3), np.uint8) for i in range(4)]
    m = np.ones((2, 2, 2), bool)
    out = vm.pooled(frames, {"a": (m, "mean"), "f": (m, "flat")})
    with torch.no_grad():
        pix = vm.pixels(frames)
        emb = vm.model.embeddings.patch_embeddings(pix)[0]
        hs = vm.model(pixel_values=pix, output_hidden_states=True).hidden_states
    assert np.allclose(out["vae/f"].astype(np.float32), emb.reshape(-1).numpy(), atol=1e-2)
    assert not np.allclose(emb.numpy(), hs[0][0].numpy(), atol=1e-3)          # hidden state 0 carries positions
    assert out["encoder/a"].shape == (2, 1, 8)
    assert np.allclose(out["encoder/a"][1, 0], hs[2][0].mean(0).numpy(), atol=1e-5)
