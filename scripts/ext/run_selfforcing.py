"""Self-Forcing (causal Wan2.1-1.3B, DMD checkpoint) video extension for exported jobs (pilot/jobs.py).
Runs in the Self-Forcing venv:

    /workspace/ext/selfforcing/venv/bin/python scripts/ext/run_selfforcing.py --repo /workspace/ext/selfforcing/repo \
        --exp b1 --model selfforcing

CausalInferencePipeline.inference(noise, prompts, initial_latent) caches the input latents block by block and
decodes [input latents + new latents] (pipeline/causal_inference.py), so the output already starts with the input
window (VAE-decoded). Source read at guandeh17/Self-Forcing 33593df3e81fa3ec10239271dd2c100facac6de1.
"""
import argparse
import os
import sys
import time
from pathlib import Path

SHA = "33593df3e81fa3ec10239271dd2c100facac6de1"
ROOT = Path(__file__).resolve().parents[2]
BLOCK, MAX_LATENTS, LATENT_HW = 3, 21, (60, 104)    # num_frame_per_block; KV cache 32760 / 1560 tokens; 480x832

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--runs", default=str(ROOT / "runs"))
ap.add_argument("--exp", required=True)
ap.add_argument("--model", required=True)
a = ap.parse_args()
runs = Path(a.runs).resolve()
sys.path[:0] = [str(Path(a.repo).resolve()), str(ROOT)]
os.chdir(a.repo)                                    # wan_models/ and configs/ are cwd-relative in the repo

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pilot import jobs  # noqa: E402
from pipeline import CausalInferencePipeline  # noqa: E402


def latents(n_frames):
    assert (n_frames - 1) % 4 == 0, f"{n_frames} frames: the causal Wan VAE needs 1 + 4k"
    return 1 + (n_frames - 1) // 4


def load():
    torch.set_grad_enabled(False)
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device="cuda")
    sd = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu", weights_only=False)   # trusted, pinned
    pipe.generator.load_state_dict(sd["generator_ema"])                 # README: inference.py --use_ema
    pipe = pipe.to(dtype=torch.bfloat16)
    for m in (pipe.text_encoder, pipe.generator, pipe.vae):
        m.to("cuda")
    return pipe


def continue_video(pipe, job, frames):
    t, h, w, _ = frames.shape
    n_in, n_all = latents(t), latents(job["num_frames"])
    assert (h, w) == (480, 832), "frame_seq_length 1560 is hardcoded in CausalInferencePipeline"
    assert n_in % BLOCK == 0 and (n_all - n_in) % BLOCK == 0 and n_all <= MAX_LATENTS, (n_in, n_all)
    x = torch.from_numpy(frames).cuda().permute(3, 0, 1, 2)[None].float() / 127.5 - 1      # (1, 3, T, H, W)
    lat = pipe.vae.encode_to_latent(x.to(torch.bfloat16)).to(torch.bfloat16)               # (1, T_lat, 16, 60, 104)
    torch.manual_seed(job["seed"])                     # the pipeline re-noises with torch.randn_like (global RNG)
    noise = torch.randn([1, n_all - n_in, 16, *LATENT_HW], device="cuda", dtype=torch.bfloat16)
    video = pipe.inference(noise=noise, text_prompts=[job["prompt"]], initial_latent=lat)   # (1, T, 3, H, W) in [0, 1]
    pipe.vae.model.clear_cache()
    return (video[0].permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()


def main():
    pipe = load()
    for d in jobs.pending(runs, a.exp, a.model):
        job, frames = jobs.load(d)
        assert job["fps"] == 16 and job["cond_frames"] > 0, "Self-Forcing runs at 16 fps; text-only is not wired"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        video = continue_video(pipe, job, frames)
        torch.cuda.synchronize()
        jobs.write(d, job, video, time.time() - t0, ext_commit=SHA, checkpoint="self_forcing_dmd.pt (generator_ema)",
                   prefix="VAE-decoded input latents")
        print(a.model, a.exp, d.name, "done", flush=True)


if __name__ == "__main__":
    main()
