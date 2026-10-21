"""CPU check of generation + activation capture on tiny pipelines.  python -m tests.test_generators"""
import numpy as np
import torch

from pilot.models import Generator, _pil
from tests import tiny

RUN = {"capture_fractions": [0.25, 0.5, 0.75]}

for name, build in {**tiny.BUILDERS, **tiny.SCREEN_BUILDERS}.items():
    pipe, cfg = build()
    gen = Generator(name, cfg, RUN, device="cpu", pipe=pipe, dtype=torch.float32)
    fr = tiny.frames(cfg["cond_frames"])
    r = gen.generate(fr, "a ball", 0, obj_box=[10, 10, 30, 30], in_wh=(64, 48))
    shapes = {k: v.shape for k, v in r["acts"].items()}
    print(name, r["video"].shape, r["video"].dtype, shapes, {k: v.shape for k, v in r["vae"].items()},
          None if r["final_latent"] is None else tuple(r["final_latent"].shape))
    # input window + continuation on one timeline: frame k is k/fps after the first input frame
    assert r["video"].shape == (cfg["num_frames"], cfg["height"], cfg["width"], 3), name
    assert r["video"].dtype == np.uint8 and r["meta"]["cond_frames"] == cfg["cond_frames"]
    if cfg["input"] == "helios_v2v":        # input pasted in front of the new frames
        want = np.stack([np.asarray(f) for f in _pil(fr, cfg["width"], cfg["height"])])
        assert np.array_equal(r["video"][:cfg["cond_frames"]], want), name
    group = cfg["blocks"][0]
    assert r["acts"][f"{group}/all"].shape[1] == 3 and r["final_latent"] is not None, name
    assert (f"{group}/obj" in r["acts"]) == (cfg.get("obj_tokens") is not False), name
    t2 = gen.generate([], "a ball", 0)                       # text only (B2-nl)
    assert t2["video"].ndim == 4 and t2["video"].shape[1:] == r["video"].shape[1:], name

# Blocks that return (text, video) or (video, audio) tuples: the pooled stream is the video one.
for name, stream, call_of_step in (("cosmos3_nano", lambda o: o[1], lambda s: 2 * s),     # separate cond/uncond calls
                                   ("ltx2", lambda o: o[0][1], lambda s: s)):            # one batched call, cond = 1
    pipe, cfg = tiny.SCREEN_BUILDERS[name]()
    gen = Generator(name, cfg, RUN, device="cpu", pipe=pipe, dtype=torch.float32)
    outs = []
    blk = getattr(pipe.transformer, cfg["blocks"][0])[0]
    r = gen.generate(tiny.frames(cfg["cond_frames"]), "a ball", 0, [10, 10, 30, 30], (64, 48),
                     hooks=[(blk, lambda m, i, o: outs.append(o))])
    n_grid = int(np.prod(r["meta"]["token_grid"]))
    for j, s in enumerate(r["acts"]["steps"]):
        want = stream(outs[call_of_step(s)])[:n_grid].float().mean(0).numpy()      # (N, D) video tokens
        assert np.allclose(r["acts"][f"{cfg['blocks'][0]}/all"][0, j], want, atol=1e-2), (name, s)
    print(name, "captures the video stream")
print("generator checks passed")
