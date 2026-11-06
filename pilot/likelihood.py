"""C2 / C2-g: which continuation does the model find most likely? Denoising loss as a likelihood (LikePhys-style).

Each scene base has variants that share the input window and differ after it:
  kcont (C2-g, Cosmos): gravity switches to k x g at the end of the window, k in {0.25 ... 4}
  kcontl / kcontv (C2-g, LTX / VACE): the same with the switch at the end of that model's window
  kvelc / kvell / kvelv / kveln (C2, Cosmos-2B and -14B / LTX / VACE / Cosmos 3 Nano): the velocity is scaled by k in {0.5, 0.8, 1, 1.25, 2} (VEL_SETS)
  kturnc / kturnl / kturnv (C2-turn, Cosmos / LTX / VACE): the velocity turns by k in {0, 30, 180} degrees, same speed (TURN_SETS)
For each variant: loss ||v_pred - (eps - x0)||^2 on the continuation latent frames (window frames clean for Cosmos and
LTX; VACE noises all frames and gets the window through its context), averaged over noise levels and fixed noise draws (identical across a base's variants). Per base: the implied multiplier (minimum
of a parabola in ln k), its symmetric MRA against the true k = 1, and the share of altered variants with a higher loss
than the true one (pref_acc; chance 0.5). The implied multiplier is left out for the turn sets.

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
from PIL import Image

from pilot import metrics
from pilot.models import _pil
from pilot.track import load_video

RENDER = {"kcont": (16, 5.8125), "kvelc": (16, 5.8125), "kturnc": (16, 5.8125),     # fps, seconds: >= the model's clip
          "kvell": (24, 5.0417), "kturnl": (24, 5.0417), "kcontl": (24, 5.0417), "kvelv": (16, 5.0625),
          "kveln": (24, 5.0417), "kturnv": (16, 5.0625), "kcontv": (16, 5.0625)}


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


class VaceLoss:
    """WanVACEPipeline's conditioning: every latent frame is noised and the window enters through the VACE context
    (window frames then grey, mask 0 on the window), as pilot.models._vace_mask generates; output v = eps - x0."""
    sigmas = CosmosLoss.sigmas

    def __init__(self, pipe, cfg, device, dtype):
        self.pipe, self.cfg, self.device, self.dtype = pipe, cfg, device, dtype

    @torch.no_grad()
    def latents(self, frames):
        """Also sets self.ctrl, the VACE context of these frames' input window, for the next loss call."""
        c, p = self.cfg, self.pipe
        pil = _pil(frames, c["width"], c["height"])
        x = p.video_processor.preprocess_video(pil, c["height"], c["width"])
        z = p.vae.encode(x.to(self.device, p.vae.dtype)).latent_dist.mode().float()
        mean = torch.tensor(p.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
        std = torch.tensor(p.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
        n, size = len(pil), pil[0].size
        grey = Image.new("RGB", size, (127, 127, 127))
        video = pil[: c["cond_frames"]] + [grey] * (n - c["cond_frames"])
        mask = [Image.new("L", size, 0)] * c["cond_frames"] + [Image.new("L", size, 255)] * (n - c["cond_frames"])
        video, mask, refs = p.preprocess_conditions(video, mask, None, 1, c["height"], c["width"], n, torch.float32,
                                                    self.device)
        ctrl = torch.cat([p.prepare_video_latents(video, mask, refs, None, self.device), p.prepare_masks(mask, refs)], 1)
        self.ctrl = ctrl.to(self.dtype)
        return (z - mean) / std

    @torch.no_grad()
    def text(self, prompt):
        return self.pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device=self.device)[0]

    @torch.no_grad()
    def loss(self, x0, emb, sigmas, noises, n_cond_latent):
        out = []
        for s in sigmas:
            for eps in noises:
                xs = (1 - s) * x0 + s * eps
                t = torch.tensor([s * 1000.0], device=self.device)
                v = self.pipe.transformer(hidden_states=xs.to(self.dtype), timestep=t,
                                          encoder_hidden_states=emb.to(self.dtype), control_hidden_states=self.ctrl,
                                          return_dict=False)[0].float()
                out.append(((v - (eps - x0)) ** 2)[:, :, n_cond_latent:].mean().item())
        return float(np.mean(out))


class NanoLoss:
    """Cosmos3OmniPipeline (Cosmos 3 Nano): one packed sequence of text then vision tokens; the window's latent frames
    stay clean (condition_frame_indexes_vision), the rest are noised at timestep s x 1000, as its denoising loop does;
    output v = eps - x0 (flow matching). Text-conditional pass only, as for the other models."""

    def __init__(self, pipe, cfg, device, dtype):
        self.pipe, self.cfg, self.device, self.dtype = pipe, cfg, device, dtype

    def sigmas(self, n):
        p, steps = self.pipe, self.cfg["steps"]
        if p.config.use_native_flow_schedule:                           # as the pipeline sets its schedule
            s = np.linspace(1.0 - 1.0 / p.scheduler.config.num_train_timesteps, 0.0, steps + 1)[:-1]
            p.scheduler.set_timesteps(steps, sigmas=s)
        else:
            p.scheduler.set_timesteps(steps)
        s = p.scheduler.sigmas.cpu().numpy()[:steps]
        return [float(v) for v in s[np.linspace(0, len(s) - 1, n).astype(int)]]

    @torch.no_grad()
    def latents(self, frames):
        c, p = self.cfg, self.pipe
        x = p.video_processor.preprocess_video(_pil(frames, c["width"], c["height"]), c["height"], c["width"])
        return p._encode_video(x.to(self.device, torch.float32)).float()             # (1, C, T, H, W), normalised

    @torch.no_grad()
    def text(self, prompt):
        c, p = self.cfg, self.pipe
        p.maybe_free_model_hooks()
        p.safety_checker.to(self.device)                                 # NVIDIA guardrail stays in the loop
        if not p.safety_checker.check_text_safety(prompt):
            raise ValueError("prompt rejected by the Cosmos guardrail")
        p.safety_checker.to("cpu")
        ids, _ = p.tokenize_prompt(prompt, None, num_frames=c["num_frames"], height=c["height"], width=c["width"],
                                   fps=c["fps"], use_system_prompt=None, add_resolution_template=True,
                                   add_duration_template=True, action_mode=None, action_view_point=None)
        return p._prepare_text_segment(ids, device=self.device)

    @torch.no_grad()
    def loss(self, x0, emb, sigmas, noises, n_cond_latent):
        p, T = self.pipe, x0.shape[2]
        cond = list(range(n_cond_latent))
        vis = p._prepare_vision_segment(input_vision_tokens=x0, has_image_condition=True,
                                        mrope_offset=emb["vision_start_temporal_offset"], vision_fps=float(self.cfg["fps"]),
                                        curr=emb["und_len"], device=self.device, condition_frame_indexes=cond)
        ind = torch.zeros(1, 1, T, 1, 1, device=self.device)
        ind[:, :, :n_cond_latent] = 1.0
        n_train = p.scheduler.config.num_train_timesteps
        out = []
        for s in sigmas:
            for eps in noises:
                xs = ind * x0 + (1 - ind) * ((1 - s) * x0 + s * eps)
                with p.transformer.cache_context("cond", step_index=0, sigma=s, num_inference_steps=self.cfg["steps"]):
                    v = p.transformer(input_ids=emb["input_ids"], text_indexes=emb["text_indexes"],
                                      position_ids=torch.cat([emb["text_mrope_ids"], vis["vision_mrope_ids"]], dim=1),
                                      und_len=emb["und_len"], sequence_length=emb["und_len"] + vis["num_vision_tokens"],
                                      vision_tokens=[xs.to(self.dtype)], vision_token_shapes=vis["vision_token_shapes"],
                                      vision_sequence_indexes=vis["vision_sequence_indexes"],
                                      vision_mse_loss_indexes=vis["vision_mse_loss_indexes"],
                                      vision_timesteps=torch.full((vis["num_noisy_vision_tokens"],), s * n_train,
                                                                  device=self.device),
                                      vision_noisy_frame_indexes=vis["vision_noisy_frame_indexes"],
                                      return_dict=False)[0][0]
                v = v.float().reshape(x0.shape)
                if not getattr(self, "_signed", False):                 # smoke check of the target's sign, once
                    self._signed = True
                    err = lambda y: ((v - y) ** 2)[:, :, n_cond_latent:].mean().item()
                    print(f"nano sign check at sigma {s:.3f}: eps - x0 {err(eps - x0):.4f}, x0 - eps {err(x0 - eps):.4f}", flush=True)
                out.append(((v - (eps - x0)) ** 2)[:, :, n_cond_latent:].mean().item())
        return float(np.mean(out))


LOSSES = {"cosmos2b": CosmosLoss, "cosmos14b": CosmosLoss, "ltx2b": LTXLoss, "vace13": VaceLoss,
          "cosmos3_nano": NanoLoss}


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
    gt = spec["gt"]
    return gt["cont_mult"] if "cont_mult" in gt else gt["turn_deg"] if "turn_deg" in gt else gt["v_scale"]


def is_true(spec):
    return mult(spec) == (0 if "turn_deg" in spec["gt"] else 1.0)


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
        t = g.true.astype(bool) if "true" in g else g.k == 1.0
        k = implied_multiplier(g.k.to_numpy(), g.loss.to_numpy()) if (g.k > 0).all() else np.nan
        true, alt = g.loss[t], g[~t]
        rows.append(dict(base=base, k_implied=k, sym_mra=float(metrics.sym_mra(k, 1.0)) if k == k else np.nan,
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
            rows.append(dict(base=base, id=spec["id"], k=mult(spec), true=is_true(spec), in_frame=spec["gt"].get("in_frame", 1.0),
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
    from pilot.models import from_pretrained
    pipe = from_pretrained(getattr(diffusers, cfg["pipeline"]), cfg["repo"], **kw)
    if cfg.get("scheduler"):          # Nano: the sampler and flow shift it generates with (pilot.models.Generator)
        pipe.scheduler = getattr(diffusers, cfg["scheduler"]).from_config(pipe.scheduler.config, flow_shift=cfg["flow_shift"])
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
