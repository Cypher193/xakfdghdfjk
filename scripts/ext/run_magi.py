"""MAGI-1 4.5B-distill video continuation for exported jobs (pilot/jobs.py). Runs in the MAGI-1 venv:

    /workspace/ext/magi/venv/bin/python scripts/ext/run_magi.py --repo /workspace/ext/magi/repo \
        --exp b1 --model magi45

Uses the building blocks of the repo's ComfyUI node (comfyui/comfy_nodes.py) so the model loads once and the
prefix comes from memory. MAGI-1 returns only the new frames (video_generate.py: chunks inside the prefix are not
yielded), so the input window is pasted in front.
Source read at SandAI-org/MAGI-1 0fcefdef8ce2df37a3b8890979433c06eb003328.
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

SHA = "0fcefdef8ce2df37a3b8890979433c06eb003328"
ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--ckpt", help="local_dir holding ckpt/{magi,t5,vae}; default: <repo>/../ckpt (setup_ext.sh)")
ap.add_argument("--runs", default=str(ROOT / "runs"))
ap.add_argument("--exp", required=True)
ap.add_argument("--model", required=True)
a = ap.parse_args()
repo = Path(a.repo).resolve()
ckpt = Path(a.ckpt) if a.ckpt else repo.parent / "ckpt"
for k, v in {"MASTER_ADDR": "localhost", "MASTER_PORT": "6009", "RANK": "0", "WORLD_SIZE": "1",
             "PAD_HQ": "1", "PAD_DURATION": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
             "SPECIAL_TOKEN_PATH": str(repo / "example/assets/special_tokens.npz")}.items():   # run.sh settings
    os.environ.setdefault(k, v)
sys.path[:0] = [str(repo), str(ROOT)]
try:
    import flashinfer  # noqa: F401
except ImportError:   # imported at module top (dit_module.py) but only called for fp8 / gated MLP, both off for 4.5B
    fi, gemm = types.ModuleType("flashinfer"), types.ModuleType("flashinfer.gemm")
    gemm.bmm_fp8, fi.gemm = None, gemm
    sys.modules.update({"flashinfer": fi, "flashinfer.gemm": gemm})

import numpy as np  # noqa: E402
import torch  # noqa: E402

from inference.common import MagiConfig, set_random_seed  # noqa: E402
from inference.infra.distributed import dist_init  # noqa: E402
from inference.infra.distributed import parallel_state as mpu  # noqa: E402
from inference.model.dit import get_dit  # noqa: E402
from inference.pipeline.prompt_process import get_txt_embeddings  # noqa: E402
from inference.pipeline.video_generate import generate_per_chunk  # noqa: E402
from inference.pipeline.video_process import encode_prefix_video, post_chunk_process  # noqa: E402
from pilot import jobs  # noqa: E402


def continue_video(cfg, dit, frames, job):
    rc = cfg.runtime_config
    t, h, w, _ = frames.shape
    n_new = job["num_frames"] - t
    assert t % 4 == 0 and h % 16 == 0 and w % 16 == 0 and n_new % 4 == 0, (t, h, w, n_new)   # non-causal VAE: 4 frames/latent
    rc.num_frames, rc.video_size_h, rc.video_size_w, rc.seed = n_new, h, w, job["seed"]       # new frames only
    caption_embs, emb_masks = get_txt_embeddings(job["prompt"], cfg)
    prefix = encode_prefix_video(torch.from_numpy(np.ascontiguousarray(frames)), rc.fps, rc.vae_pretrained,
                                 rc.scale_factor, parallel_group=mpu.get_tp_group(with_context_parallel=True))
    set_random_seed(job["seed"])                       # the latent noise is drawn inside generate_per_chunk
    chunks = [post_chunk_process(c, cfg) for c in generate_per_chunk(model=dit, prefix_video=prefix,
                                                                      caption_embs=caption_embs, emb_masks=emb_masks)]
    new = torch.cat(chunks, 0).permute(0, 2, 3, 1).cpu().numpy()          # uint8 (T, H, W, 3), prefix excluded
    assert len(new) >= n_new, (len(new), n_new)                          # rounded up to whole 24-frame chunks
    return np.concatenate([frames, new[:n_new]])


def main():
    cfg = MagiConfig.from_json(str(repo / "example/4.5B/4.5B_distill_config.json"))
    rc = cfg.runtime_config
    rc.load = str(ckpt / "ckpt/magi/4.5B_distill")
    rc.t5_pretrained, rc.vae_pretrained = str(ckpt / "ckpt/t5"), str(ckpt / "ckpt/vae")
    dist_init(cfg)
    dit = get_dit(cfg)
    for d in jobs.pending(a.runs, a.exp, a.model):
        job, frames = jobs.load(d)
        assert job["fps"] == rc.fps and job["cond_frames"] > 0, "MAGI-1 runs at 24 fps; text-only is not wired"
        rc.num_steps = job["steps"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        video = continue_video(cfg, dit, frames, job)
        torch.cuda.synchronize()
        jobs.write(d, job, video, time.time() - t0, ext_commit=SHA, variant="4.5B_distill", cfg_number=rc.cfg_number,
                   prefix="input pixels")
        print(a.model, a.exp, d.name, "done", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
