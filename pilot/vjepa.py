"""V-JEPA 2, a predictor that never makes pixels: C1 probes, C2 / C2-g surprise, C3 gamma per layer.

    python -m pilot.vjepa acts --model vjepa2l --exp b1          # runs/{exp}/{model}/{scene}_s0/acts.npz + meta.json
    python -m pilot.vjepa surprise --model vjepa2l --set kvelv --out results/c2_kvelv_vjepa2l_ltx_vjepa2.csv
    python -m pilot.vjepa gamma --model vjepa2l --out results/c3_vjepa2l_gamma_ltx_vjepa2.csv

- Frames are squashed to the crop size: the renders are 832x480 and objects cross the full width, so the processor's
  centre crop would cut them out. Normalisation is the processor's.
- The encoder sees only the context frames. VJEPA2Model.forward encodes the whole clip before masking, which would let
  the future into the context tokens, so the encoder and predictor are called separately here.
- acts (C1): the context's tokens at every encoder layer, pooled over the object box ("obj") or all tokens ("all"), as
  the generators' acts, so pilot.probe reads them unchanged. "vae/*" holds the patch embeddings (hidden state 0), the
  input-level baseline that VAE latents are for the generators. "predictor/all": the predicted future tokens, pooled.
- surprise (C2, C2-g): mean L1 between the predictor's future tokens and the encoder's tokens of the real clip at those
  positions, V-JEPA 2's own pairwise test. Rows match pilot.likelihood, so its summary and scripts/c2_report.py apply.
- gamma (C3): per layer, the slope of the out-of-fold probe prediction of ln g on the true ln g over the kgrav scenes
  (1: follows gravity, 0: one prior value), scene-split as every probe.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pilot import metrics, probe

MODELS = {"vjepa2l": "facebook/vjepa2-vitl-fpc64-256", "vjepa2g": "facebook/vjepa2-vitg-fpc64-384"}
N_FRAMES = 64                                                   # frames per clip in training (fpc64)
SURPRISE_SETS = {"kvelv": 16, "kturnl": 24, "kcontl": 24}   # context frames: 1 s at the render fps, before the switch
ACT_FPS, ACT_FRAMES = 16, 16                                    # C1 context: the first 1.0 s


class VJEPA:
    def __init__(self, model, mean, std, device):
        self.model, self.device = model.eval(), device
        c = model.config
        self.size, self.tub, self.patch = c.crop_size, c.tubelet_size, c.patch_size
        self.mean = torch.tensor(mean, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 1, 3, 1, 1)
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def load(cls, name, device="cuda"):
        from transformers import AutoVideoProcessor, VJEPA2Model
        p = AutoVideoProcessor.from_pretrained(MODELS[name])
        m = VJEPA2Model.from_pretrained(MODELS[name], torch_dtype=torch.bfloat16).to(device)
        return cls(m, p.image_mean, p.image_std, device)

    def pixels(self, frames):
        """(T, H, W, 3) uint8 -> (1, T, 3, S, S), squashed and normalised."""
        x = torch.as_tensor(np.stack(frames), device=self.device).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        return ((x[None] - self.mean) / self.std).to(self.dtype)

    def per_t(self):
        return (self.size // self.patch) ** 2

    @torch.no_grad()
    def acts(self, frames, box, in_wh):
        pix = self.pixels(frames)
        o = self.model(pixel_values_videos=pix, skip_predictor=True, output_hidden_states=True)   # pix: context only
        hs = o.hidden_states                                            # embeddings, then each layer (before the final norm)
        g = self.size // self.patch
        m = np.zeros((len(frames) // self.tub, g, g), bool)
        x0, y0, x1, y1 = box
        sx, sy = g / in_wh[0], g / in_wh[1]
        m[:, max(int(y0 * sy), 0):max(int(np.ceil(y1 * sy)), 1), max(int(x0 * sx), 0):max(int(np.ceil(x1 * sx)), 1)] = True
        obj = torch.as_tensor(m.reshape(-1), device=self.device)
        pool = lambda h: (h[0, obj].float().mean(0).cpu().numpy(), h[0].float().mean(0).cpu().numpy())
        layers = [pool(h) for h in hs]
        pred = self.predict(o.last_hidden_state, len(frames), N_FRAMES)
        return {"vae/obj": layers[0][0], "vae/all": layers[0][1],
                "encoder/obj": np.stack([o for o, _ in layers[1:]])[:, None], "encoder/all": np.stack([a for _, a in layers[1:]])[:, None],
                "predictor/all": pred[0].float().mean(0).cpu().numpy()[None, None]}

    def predict(self, ctx, n_ctx, n_total):
        nc, n = n_ctx // self.tub * self.per_t(), n_total // self.tub * self.per_t()
        idx = torch.arange(n, device=self.device)[None]
        return self.model.predictor(encoder_hidden_states=ctx, context_mask=[idx[:, :nc]],
                                    target_mask=[idx[:, nc:]]).last_hidden_state

    @torch.no_grad()
    def surprise(self, frames, n_ctx):
        pix = self.pixels(frames)
        ctx = self.model.encoder(pixel_values_videos=pix[:, :n_ctx]).last_hidden_state
        tgt = self.model.encoder(pixel_values_videos=pix).last_hidden_state[:, n_ctx // self.tub * self.per_t():]
        return float((self.predict(ctx, n_ctx, len(frames)).float() - tgt.float()).abs().mean())


def write_acts(vj, name, exp, cfg):
    from pilot.generate import rendered_inputs
    e = cfg["experiments"][exp]
    data = Path(cfg["paths"]["data"])
    specs = sorted((data / "scenes" / e["scenes"]).glob("*.json"))
    for f in specs:
        spec = json.loads(f.read_text(encoding="utf-8"))
        out = Path("runs") / exp / name / f"{spec['id']}_s0"
        if (out / "acts.npz").exists():
            continue
        frames, box = rendered_inputs(spec, data / "renders" / e["scenes"], ACT_FPS, ACT_FRAMES)
        in_wh = (frames[0].shape[1], frames[0].shape[0])
        acts = vj.acts(frames, box, in_wh)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "acts.npz", **acts)
        (out / "meta.json").write_text(json.dumps(dict(scene=spec["id"], exp=exp, model=name, seed=0, fps=ACT_FPS,
                                                       cond_frames=ACT_FRAMES, box=list(box), in_wh=list(in_wh))))
        print(f"acts {exp}/{spec['id']}", flush=True)


def run_surprise(vj, scene_root, render_root, out, n_ctx, shard=None, frames_fn=None):
    from pilot.likelihood import bases, is_true, mult, summarize
    from pilot.track import load_video
    out = Path(out)
    done = set(pd.read_csv(out).base) if out.exists() else set()
    for base, variants in bases(scene_root, shard).items():
        if base in done:
            continue
        rows = []
        for spec in variants:
            frames = frames_fn(spec) if frames_fn else list(load_video(Path(render_root) / f"{spec['id']}.mp4")[0])
            assert len(frames) >= N_FRAMES, (spec["id"], len(frames))
            rows.append(dict(base=base, id=spec["id"], k=mult(spec), true=is_true(spec),
                             in_frame=spec["gt"].get("in_frame", 1.0), loss=vj.surprise(frames[:N_FRAMES], n_ctx)))
            print(f"surprise {spec['id']}: {rows[-1]['loss']:.5f}", flush=True)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    df = pd.read_csv(out)
    summ = summarize(df)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    return df, summ


def gamma(name, out, runs="runs"):
    feats, y, g = probe.load(runs, name, ["b5"], "g")
    rows = []
    for key in ("encoder/obj", "encoder/all"):
        for layer in range(feats[key].shape[0]):
            pred = np.exp(probe.cv_predict(feats[key][layer, 0], np.log(y), g))
            sl, (a, b) = metrics.gamma_fit(pred, y, groups=g)
            rows.append(dict(feature=key, layer=layer, gamma=sl, lo=a, hi=b, sym_mra=float(metrics.sym_mra(pred, y).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acts", "surprise", "gamma"])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--exp", help="acts: b1 or b5")
    ap.add_argument("--set", choices=sorted(SURPRISE_SETS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--config", default="configs/pilot.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.cmd == "gamma":
        print(gamma(a.model, a.out).sort_values("sym_mra").tail(5).to_string(index=False))
        return
    vj = VJEPA.load(a.model)
    if a.cmd == "acts":
        write_acts(vj, a.model, a.exp, cfg)
        return
    data = Path(cfg["paths"]["data"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df, summ = run_surprise(vj, data / "scenes" / a.set, data / "renders" / a.set, a.out, SURPRISE_SETS[a.set])
    print(f"{len(summ)} bases: pref_acc {summ.pref_acc.mean():.3f}, k_implied median {summ.k_implied.median():.3f}")


if __name__ == "__main__":
    main()
