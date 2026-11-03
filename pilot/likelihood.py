"""C2 / C2-g: which continuation does the model find most likely? Denoising loss as a likelihood (LikePhys-style).

Each scene base has variants that share the input window and differ after it:
  kcont (C2-g, Cosmos): gravity switches to k x g at the end of the window, k in {0.25 ... 4}
  kvelc / kvell (C2, Cosmos / LTX): the velocity is scaled by k in {0.5, 0.8, 1, 1.25, 2} (pilot.scenes.VEL_SETS)
For each variant: loss ||v_pred - (eps - x0)||^2 on the continuation latent frames, the window frames clean, averaged
over noise levels and fixed noise draws (identical across a base's variants). Per base: the implied multiplier (minimum
of a parabola in ln k), its symmetric MRA against the true k = 1, and the share of altered variants with a higher loss
than the true one (pref_acc; chance 0.5).

    python -m pilot.likelihood --model cosmos2b --set kcont --sigmas 16 --out results/c2g_cosmos2b.csv
    python -m pilot.likelihood --model ltx2b --set kvell --shard 1/2 --out results/c2_kvell_ltx2b.csv
    python -m pilot.likelihood --set kvelc --shard 1/4 --write-shard data/shard_kvelc      # scene JSONs to render

Rows are appended after each base, and a rerun skips the bases already in --out. The summary goes to *_summary.csv.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pilot import metrics
from pilot.models import _pil
from pilot.track import load_video

RENDER = {"kcont": (16, 5.8125), "kvelc": (16, 5.8125), "kvell": (24, 5.0417)}   # fps, seconds: >= the model's clip


class CosmosLoss:
    def __init__(self, pipe, cfg, device, dtype):
        self.pipe, self.cfg, self.device, self.dtype = pipe, cfg, device, dtype

    def sigmas(self, n):
        self.pipe.scheduler.set_timesteps(self.cfg["steps"])
        s = self.pipe.scheduler.sigmas.cpu().numpy()[:-1]
        return [float(v) for v in s[np.linspace(0, len(s) - 1, n).astype(int)]]

    @torch.no_grad()
    def latents(self, frames):
        c, p = self.cfg, self.pipe
        x = p.video_processor.preprocess_video(_pil(frames, c["width"], c["height"]), c["height"], c["width"])
        z = p.vae.encode(x.to(self.device, p.vae.dtype)).latent_dist.mode().float()
        return (z - p.latents_mean.to(z)) * p.latents_std.to(z)       # pipeline: latents_std holds 1/std

    @torch.no_grad()
    def text(self, prompt):
        self.pipe.maybe_free_model_hooks()        # offload: the transformer leaves the GPU before the text encoder loads
        self.pipe.safety_checker.to(self.device)                      # as the pipeline does before its own check
        if not self.pipe.safety_checker.check_text_safety(prompt):    # NVIDIA guardrail stays in the loop
            raise ValueError("prompt rejected by the Cosmos guardrail")
        return self.pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device=self.device)[0]

    @torch.no_grad()
    def loss(self, x0, emb, sigmas, noises, n_cond_latent, cond_t=1e-4):
        B, C, T, H, W = x0.shape
        ind = torch.zeros(B, 1, T, 1, 1, device=self.device)
        ind[:, :, :n_cond_latent] = 1.0
        mask = ind.expand(-1, -1, -1, H, W)
        pad = x0.new_zeros(1, 1, self.cfg["height"], self.cfg["width"])
        out = []
        for s in sigmas:
            for eps in noises:
                xs = mask * x0 + (1 - mask) * ((1 - s) * x0 + s * eps)
                ts = ind * cond_t + (1 - ind) * s
                v = self.pipe.transformer(hidden_states=xs.to(self.dtype), condition_mask=mask.to(self.dtype),
                                          timestep=ts.to(self.dtype), encoder_hidden_states=emb,
                                          padding_mask=pad.to(self.dtype), return_dict=False)[0].float()
                err = (v - (eps - x0)) ** 2
                out.append(err[:, :, n_cond_latent:].mean().item())   # continuation frames only
        return float(np.mean(out))


class LTXLoss:
    """LTXConditionPipeline's conditioning: window tokens clean at timestep 0, the rest noised; output v = eps - x0."""

    def __init__(self, pipe, cfg, device, dtype):
        self.pipe, self.cfg, self.device, self.dtype = pipe, cfg, device, dtype

    def sigmas(self, n):
        from diffusers.pipelines.ltx.pipeline_ltx_condition import linear_quadratic_schedule
        s = np.asarray(linear_quadratic_schedule(self.cfg["steps"]), float)
        s = s[s > 0]
        return [float(v) for v in s[np.linspace(0, len(s) - 1, n).astype(int)]]

    @torch.no_grad()
    def latents(self, frames):
        c, p = self.cfg, self.pipe
        x = p.video_processor.preprocess_video(_pil(frames, c["width"], c["height"]), c["height"], c["width"])
        z = p.vae.encode(x.to(self.device, p.vae.dtype)).latent_dist.mode()
        return p._normalize_latents(z, p.vae.latents_mean, p.vae.latents_std).float()

    @torch.no_grad()
    def text(self, prompt):
        out = self.pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device=self.device)
        return out[0], out[1]

    @torch.no_grad()
    def loss(self, x0, emb, sigmas, noises, n_cond_latent):
        p, (txt, txt_mask) = self.pipe, emb
        B, C, T, H, W = x0.shape
        ps, pt = p.transformer_spatial_patch_size, p.transformer_temporal_patch_size
        ids = p._prepare_video_ids(B, T, H, W, patch_size=ps, patch_size_t=pt, device=self.device)
        cond = (ids[:, 0] < n_cond_latent).float()                    # (B, tokens): window tokens
        coords = p._scale_video_ids(ids, p.vae_spatial_compression_ratio, p.vae_temporal_compression_ratio,
                                    0, self.device).float()
        coords[:, 0] = coords[:, 0] / self.cfg["fps"]
        ind = torch.zeros(B, 1, T, 1, 1, device=self.device)
        ind[:, :, :n_cond_latent] = 1.0
        out = []
        for s in sigmas:
            for eps in noises:
                xs = ind * x0 + (1 - ind) * ((1 - s) * x0 + s * eps)
                v = p.transformer(hidden_states=p._pack_latents(xs, ps, pt).to(self.dtype), encoder_hidden_states=txt,
                                  timestep=(1 - cond) * s * 1000.0, encoder_attention_mask=txt_mask,
                                  video_coords=coords, return_dict=False)[0]
                v = p._unpack_latents(v.float(), T, H, W, ps, pt)
                out.append(((v - (eps - x0)) ** 2)[:, :, n_cond_latent:].mean().item())
        return float(np.mean(out))


LOSSES = {"cosmos2b": CosmosLoss, "ltx2b": LTXLoss}


def implied_multiplier(ks, losses):
    """Minimum of a parabola in ln k through the losses (falls back to the best grid point at the edges)."""
    x, y = np.log(ks), np.asarray(losses)
    a, b, _ = np.polyfit(x, y, 2)
    if a > 0:
        xm = -b / (2 * a)
        if x.min() <= xm <= x.max():
            return float(np.exp(xm))
    return float(ks[int(np.argmin(y))])


def mult(spec):
    return spec["gt"]["cont_mult"] if "cont_mult" in spec["gt"] else spec["gt"]["v_scale"]


def bases(scene_root, shard=None):
    """{base: variant specs sorted by k}, every n-th base from the k-th for shard 'k/n'."""
    specs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(scene_root).glob("*.json"))]
    names = sorted({s["id"].rsplit("_x", 1)[0] for s in specs})
    if shard:
        k, n = map(int, shard.split("/"))
        names = names[k - 1::n]
    return {b: sorted((s for s in specs if s["id"].rsplit("_x", 1)[0] == b), key=mult) for b in names}


def summarize(df):
    rows = []
    for base, g in df.groupby("base"):
        k = implied_multiplier(g.k.to_numpy(), g.loss.to_numpy())
        true = g.loss[g.k == 1.0]
        alt = g[g.k != 1.0]
        rows.append(dict(base=base, k_implied=k, sym_mra=float(metrics.sym_mra(k, 1.0)),
                         pref_acc=float((alt.loss > true.iloc[0]).mean()) if len(true) and len(alt) else np.nan))
    return pd.DataFrame(rows)


def run(lo, cfg, scene_root, render_root, out, n_sigmas=8, n_noise=4, shard=None, frames_fn=None):
    sigmas = lo.sigmas(n_sigmas)
    n_cond_latent = 1 + (cfg["cond_frames"] - 1) // cfg["vae_stride"][0]
    done = set(pd.read_csv(out).base) if out.exists() else set()
    for base, variants in bases(scene_root, shard).items():
        if base in done:
            continue
        t0, emb, noises, rows = time.time(), lo.text(variants[0]["caption"]), None, []
        gen = torch.Generator(lo.device).manual_seed(0)
        for spec in variants:
            frames = frames_fn(spec) if frames_fn else list(load_video(Path(render_root) / f"{spec['id']}.mp4")[0])
            assert len(frames) >= cfg["num_frames"], (spec["id"], len(frames))
            x0 = lo.latents(frames[: cfg["num_frames"]])
            if noises is None:                                  # same draws for every k
                noises = [torch.randn(x0.shape, generator=gen, device=lo.device) for _ in range(n_noise)]
            rows.append(dict(base=base, id=spec["id"], k=mult(spec), in_frame=spec["gt"].get("in_frame", 1.0),
                             loss=lo.loss(x0, emb, sigmas, noises, n_cond_latent), n_sigmas=n_sigmas, n_noise=n_noise))
            print(f"likelihood {spec['id']}: loss {rows[-1]['loss']:.5f}", flush=True)
        pd.DataFrame(rows).assign(wall_s=round(time.time() - t0, 1)).to_csv(out, mode="a", header=not out.exists(),
                                                                             index=False)
    df = pd.read_csv(out)
    summ = summarize(df)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cosmos2b", choices=sorted(LOSSES))
    ap.add_argument("--set", default="kcont", choices=sorted(RENDER))
    ap.add_argument("--shard", help="k/n: every n-th scene base from the k-th")
    ap.add_argument("--sigmas", type=int, default=8)
    ap.add_argument("--noise", type=int, default=4)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--write-shard", type=Path, help="only copy this shard's scene JSONs here (to render), then exit")
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    run_cfg = yaml.safe_load(open(a.config))
    data = Path(run_cfg["paths"]["data"])
    if a.write_shard:
        shutil.rmtree(a.write_shard, ignore_errors=True)
        a.write_shard.mkdir(parents=True)
        specs = [s for v in bases(data / "scenes" / a.set, a.shard).values() for s in v]
        for s in specs:
            shutil.copy(data / "scenes" / a.set / f"{s['id']}.json", a.write_shard)
        print(f"{len(specs)} scenes -> {a.write_shard}", flush=True)
        return
    import diffusers
    cfg = run_cfg["models"][a.model]
    kw = dict(torch_dtype=torch.bfloat16) | ({"revision": cfg["revision"]} if cfg.get("revision") else {})
    pipe = getattr(diffusers, cfg["pipeline"]).from_pretrained(cfg["repo"], **kw)
    if cfg.get("cpu_offload"):        # Cosmos: its text encoder and guardrail do not fit next to it on a 24 GB card
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df, summ = run(LOSSES[a.model](pipe, cfg, "cuda", torch.bfloat16), cfg, data / "scenes" / a.set,
                   data / "renders" / a.set, a.out, a.sigmas, a.noise, a.shard)
    print(summ.to_string(index=False))
    print(f"{len(summ)} bases: k_implied median {summ.k_implied.median():.3f}, sym MRA {summ.sym_mra.mean():.3f}, "
          f"pref_acc {summ.pref_acc.mean():.3f}")


if __name__ == "__main__":
    main()
