"""Generator wrappers with activation capture (diffusers 0.41).

Each wrapper: generate(cond_frames, prompt, seed, obj_box) -> dict(video, final_latent, acts, vae, meta).
cond_frames: list of RGB uint8 arrays at the model's fps (empty list = text only).
obj_box: [x0, y0, x1, y1] of the object on the input frames, in input-frame pixels (None = no object pooling).

Activations: forward hooks on every transformer block, conditional pass only, at the denoising steps
closest to the configured fractions. Two poolings per block and step: mean over object tokens in the
input (conditioning) latent frames, and mean over all tokens.
"""
import time

import numpy as np
import torch
from PIL import Image

DTYPE = torch.bfloat16


def _pil(frames, w, h):
    return [Image.fromarray(f).resize((w, h), Image.BICUBIC) for f in frames]


class Capture:
    """Forward hooks that pool block outputs at selected denoising steps."""

    def __init__(self, cfg, n_steps, fractions, token_grid, obj_mask):
        self.cfg, self.grid, self.obj = cfg, token_grid, obj_mask            # obj_mask: bool (F, H, W) tokens
        self.steps = sorted({int(round(f * (n_steps - 1))) for f in fractions})
        self.step, self.seen, self.acts, self.handles = 0, set(), {}, []
        self.final_latent = None
        self.n_steps = n_steps

    def attach(self, transformer):
        for group in self.cfg["blocks"]:
            for i, blk in enumerate(getattr(transformer, group)):
                self.handles.append(blk.register_forward_hook(self._hook(group, i)))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()

    def _hook(self, group, i):
        def fn(_module, _inputs, out):
            key = (group, i, self.step)
            if self.step not in self.steps or key in self.seen:               # first call per step = cond pass
                return
            self.seen.add(key)
            h = out[0] if isinstance(out, (tuple, list)) else out
            if self.cfg.get("cfg") == "batched":
                h = h[self.cfg["cond_batch_index"]]
            else:
                h = h[0]
            h = h.reshape(-1, h.shape[-1]).float()                             # (N, D)
            n_grid = int(np.prod(self.grid))
            pooled = {"all": h[:n_grid].mean(0)}
            if self.obj is not None and h.shape[0] >= n_grid and self.obj.any():
                m = torch.from_numpy(self.obj.reshape(-1)).to(h.device)
                pooled["obj"] = h[:n_grid][m].mean(0)
            self.acts[key] = {k: v.cpu().numpy().astype(np.float16) for k, v in pooled.items()}
        return fn

    def callback(self, _pipe, i, _t, kwargs):
        self.step = i + 1
        if i == self.n_steps - 1 and "latents" in kwargs:
            self.final_latent = kwargs["latents"].detach().float().cpu()
        return kwargs

    def arrays(self):
        """{'blocks/obj': (n_blocks, n_steps, D), ...} for npz storage."""
        out = {}
        for group in self.cfg["blocks"]:
            idx = sorted({k[1] for k in self.acts if k[0] == group})
            for pool in ("obj", "all"):
                if all(pool in self.acts.get((group, i, s), {}) for i in idx for s in self.steps) and idx:
                    out[f"{group}/{pool}"] = np.stack([np.stack([self.acts[(group, i, s)][pool] for s in self.steps])
                                                       for i in idx])
        out["steps"] = np.array(self.steps)
        return out


def token_grid(cfg):
    vt, vs, _ = cfg["vae_stride"]
    pt, ph, pw = cfg["patch"]
    f_lat = (cfg["num_frames"] - 1) // vt + 1
    return (f_lat // pt, cfg["height"] // (cfg["vae_stride"][1] * ph), cfg["width"] // (cfg["vae_stride"][2] * pw))


def object_token_mask(cfg, grid, box, in_wh):
    """Tokens covering the object box in the latent frames that hold the input frames."""
    if box is None:
        return None
    n_cond_latent = 1 + (cfg["cond_frames"] - 1) // cfg["vae_stride"][0]   # causal VAE: frame 0, then groups
    F, H, W = grid
    sx, sy = W / in_wh[0], H / in_wh[1]
    x0, y0, x1, y1 = box
    m = np.zeros(grid, bool)
    m[:n_cond_latent, max(int(y0 * sy), 0):max(int(np.ceil(y1 * sy)), 1),
      max(int(x0 * sx), 0):max(int(np.ceil(x1 * sx)), 1)] = True
    return m


def _cosmos25(pipe, c, pil, kw):
    if pil:
        kw |= dict(video=pil, num_latent_conditional_frames=2)
    return pipe(**kw).frames[0]


def _vace_mask(pipe, c, pil, kw):
    if pil:
        n = c["num_frames"]
        grey = Image.new("RGB", (c["width"], c["height"]), (127, 127, 127))
        known, unknown = Image.new("L", grey.size, 0), Image.new("L", grey.size, 255)
        kw |= dict(video=pil + [grey] * (n - len(pil)), mask=[known] * len(pil) + [unknown] * (n - len(pil)))
    return pipe(**kw).frames[0]


def _ltx_conditions(pipe, c, pil, kw):
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    kw |= dict(frame_rate=c["fps"], decode_timestep=0.05, decode_noise_scale=0.025)
    if pil:
        kw["conditions"] = [LTXVideoCondition(video=pil, frame_index=0)]
    return pipe(**kw).frames[0]


# cfg["input"] -> fn(pipe, cfg, pil_frames_or_None, common_kwargs) -> output frames (F, H, W, 3) that start with
# the input window (frame k = time k/fps from the input start)
INPUTS = {"cosmos25": _cosmos25, "vace_mask": _vace_mask, "ltx_conditions": _ltx_conditions}


class Generator:
    def __init__(self, name, cfg, run_cfg, device="cuda", pipe=None, dtype=DTYPE):
        """pipe: an already-built pipeline (tests use tiny random-weight ones); otherwise loaded from the Hub."""
        import diffusers
        self.name, self.cfg, self.run_cfg, self.device, self.dtype = name, cfg, run_cfg, device, dtype
        if pipe is None:
            kw = dict(torch_dtype=dtype)
            if cfg.get("revision"):
                kw["revision"] = cfg["revision"]
            pipe = getattr(diffusers, cfg["pipeline"]).from_pretrained(cfg["repo"], **kw)
        self.pipe = pipe
        if cfg.get("cpu_offload"):            # each component goes to the GPU only while it runs (48 GB cards)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        if cfg.get("vae_tiling") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()     # decode in tiles: the 720p decode is the memory peak
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def vae_features(self, frames, box, in_wh):
        """VAE latent of the input frames, pooled over the object region and overall (baseline probe)."""
        c = self.cfg
        x = self.pipe.video_processor.preprocess_video(_pil(frames, c["width"], c["height"]), c["height"], c["width"])
        vae = self.pipe.vae
        home = next(vae.parameters()).device              # CPU when model offload is on
        vae.to(self.device)
        lat = vae.encode(x.to(self.device, self.dtype)).latent_dist.mean[0].float()   # (C, F, h, w)
        vae.to(home)
        out = {"all": lat.mean((1, 2, 3)).cpu().numpy()}
        if box is not None:
            _, F, h, w = lat.shape
            sx, sy = w / in_wh[0], h / in_wh[1]
            x0, y0, x1, y1 = box
            sub = lat[:, :, int(y0 * sy):max(int(np.ceil(y1 * sy)), int(y0 * sy) + 1),
                      int(x0 * sx):max(int(np.ceil(x1 * sx)), int(x0 * sx) + 1)]
            out["obj"] = sub.mean((1, 2, 3)).cpu().numpy()
        return out

    def _call(self, frames, prompt, gen, cb):
        c = self.cfg
        common = dict(prompt=prompt, height=c["height"], width=c["width"], num_frames=c["num_frames"],
                      num_inference_steps=c["steps"], guidance_scale=c["guidance"], generator=gen,
                      output_type="np", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        pil = _pil(frames, c["width"], c["height"]) if frames else None
        return INPUTS[c["input"]](self.pipe, c, pil, common)

    def blocks(self, group=None):
        return getattr(self.pipe.transformer, group or self.cfg["blocks"][0])

    def generate(self, cond_frames, prompt, seed, obj_box=None, in_wh=(854, 480), capture=True, hooks=()):
        """hooks: extra (module, forward_hook) pairs active during this call only (used for steering)."""
        c = self.cfg
        grid = token_grid(c)
        cap = Capture(c, c["steps"], self.run_cfg["capture_fractions"], grid,
                      object_token_mask(c, grid, obj_box, in_wh))
        cap.handles += [m.register_forward_hook(fn) for m, fn in hooks]   # first: capture then sees steered output
        if capture:
            cap.attach(self.pipe.transformer)
        gen = torch.Generator(self.device).manual_seed(seed)
        t0 = time.time()
        try:
            video = np.asarray(self._call(list(cond_frames), prompt, gen, cap.callback))
        finally:
            cap.detach()
        wall = time.time() - t0
        assert len(video) == c["num_frames"], f"{self.name}: {len(video)} frames, expected {c['num_frames']}"
        video = (np.clip(video, 0, 1) * 255).astype(np.uint8) if video.dtype != np.uint8 else video
        res = dict(video=video, final_latent=cap.final_latent, acts=cap.arrays() if capture else {},
                   meta=dict(model=self.name, repo=c["repo"], revision=c.get("revision"), seed=seed, prompt=prompt,
                             cond_frames=len(cond_frames), cond_seconds=len(cond_frames) / c["fps"], fps=c["fps"],
                             height=c["height"], width=c["width"], num_frames=c["num_frames"], steps=c["steps"],
                             guidance=c["guidance"], wall_s=round(wall, 1), token_grid=list(grid)))
        if cond_frames and capture:
            res["vae"] = self.vae_features(cond_frames, obj_box, in_wh)
        return res
