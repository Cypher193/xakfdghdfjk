"""Tiny random-weight versions of the pilot generators (configs from diffusers' own tests), for CPU checks."""
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, \
    Qwen2Tokenizer, T5EncoderModel

import diffusers as D


class DummySafetyChecker(D.ModelMixin, D.ConfigMixin):
    """Test-only stand-in. Real runs always load NVIDIA's cosmos_guardrail."""
    def __init__(self):
        super().__init__()
        self.register_buffer("_t", torch.zeros(1), persistent=False)

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, frames):
        return frames


def _t5():
    cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
    return T5EncoderModel(cfg).eval(), AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")


def _wan_vae():
    return D.AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                              temperal_downsample=[False, True, True])


def vace():
    torch.manual_seed(0)
    te, tok = _t5()
    tr = D.WanVACETransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                                     out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=3,
                                     cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
                                     vace_layers=[0, 2], vace_in_channels=96)
    pipe = D.WanVACEPipeline(transformer=tr, vae=_wan_vae(), scheduler=D.FlowMatchEulerDiscreteScheduler(shift=7.0),
                             text_encoder=te, tokenizer=tok, transformer_2=None)
    cfg = dict(repo="tiny", input="vace_mask", fps=16, num_frames=9, height=32, width=32, steps=8, guidance=5.0,
               cond_frames=5, blocks=["blocks", "vace_blocks"], patch=[1, 2, 2], vae_stride=[4, 8, 8], cfg="separate")
    return pipe, cfg


def ltx():
    torch.manual_seed(0)
    te, tok = _t5()
    tr = D.LTXVideoTransformer3DModel(in_channels=8, out_channels=8, patch_size=1, patch_size_t=1, num_attention_heads=4,
                                      attention_head_dim=8, cross_attention_dim=32, num_layers=2, caption_channels=32)
    vae = D.AutoencoderKLLTXVideo(in_channels=3, out_channels=3, latent_channels=8, block_out_channels=(8, 8, 8, 8),
                                  decoder_block_out_channels=(8, 8, 8, 8), layers_per_block=(1, 1, 1, 1, 1),
                                  decoder_layers_per_block=(1, 1, 1, 1, 1),
                                  spatio_temporal_scaling=(True, True, False, False),
                                  decoder_spatio_temporal_scaling=(True, True, False, False),
                                  decoder_inject_noise=(False, False, False, False, False),
                                  upsample_residual=(False, False, False, False), upsample_factor=(1, 1, 1, 1),
                                  timestep_conditioning=False, patch_size=1, patch_size_t=1,
                                  encoder_causal=True, decoder_causal=False)
    vae.use_framewise_encoding = vae.use_framewise_decoding = False
    pipe = D.LTXConditionPipeline(transformer=tr, vae=vae, scheduler=D.FlowMatchEulerDiscreteScheduler(),
                                  text_encoder=te, tokenizer=tok)
    cfg = dict(repo="tiny", input="ltx_conditions", fps=24, num_frames=9, height=32, width=32, steps=8, guidance=3.0,
               cond_frames=5, blocks=["transformer_blocks"], patch=[1, 1, 1], vae_stride=[4, 4, 4], cfg="batched",
               cond_batch_index=1)
    return pipe, cfg


def cosmos():
    torch.manual_seed(0)
    tr = D.CosmosTransformer3DModel(in_channels=17, out_channels=16, num_attention_heads=2, attention_head_dim=16,
                                    num_layers=2, mlp_ratio=2, text_embed_dim=32, adaln_lora_dim=4, max_size=(4, 32, 32),
                                    patch_size=(1, 2, 2), rope_scale=(2.0, 1.0, 1.0), concat_padding_mask=True,
                                    extra_pos_embed_type="learnable")
    qcfg = Qwen2_5_VLConfig(
        text_config={"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 2, "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default",
                                                                "type": "default"}, "rope_theta": 1000000.0},
        vision_config={"depth": 2, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 16},
        hidden_size=16, vocab_size=152064, vision_end_token_id=151653, vision_start_token_id=151652,
        vision_token_id=151654)
    pipe = D.Cosmos2_5_PredictBasePipeline(
        transformer=tr, vae=_wan_vae(), scheduler=D.UniPCMultistepScheduler(),
        text_encoder=Qwen2_5_VLForConditionalGeneration(qcfg),
        tokenizer=Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"),
        safety_checker=DummySafetyChecker())
    cfg = dict(repo="tiny", input="cosmos25", fps=16, num_frames=9, height=32, width=32, steps=8, guidance=7.0,
               cond_frames=5, blocks=["transformer_blocks"], patch=[1, 2, 2], vae_stride=[4, 8, 8], cfg="separate")
    return pipe, cfg


BUILDERS = {"vace13": vace, "ltx2b": ltx, "cosmos2b": cosmos}


def frames(n, wh=(64, 48), seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (wh[1], wh[0], 3), dtype=np.uint8) for _ in range(n)]


def wan22():
    """Tiny WanPipeline with Wan2.2-TI2V's per-token timesteps (expand_timesteps)."""
    torch.manual_seed(0)
    te, tok = _t5()
    tr = D.WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                                 out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2,
                                 cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32)
    pipe = D.WanPipeline(transformer=tr, vae=_wan_vae(), scheduler=D.FlowMatchEulerDiscreteScheduler(shift=7.0),
                         text_encoder=te, tokenizer=tok, transformer_2=None, expand_timesteps=True)
    cfg = dict(repo="tiny", fps=24, num_frames=9, height=32, width=32, steps=8, blocks=["blocks"], patch=[1, 2, 2],
               vae_stride=[4, 8, 8])
    return pipe, cfg


def ltx_t2v():
    pipe, cfg = ltx()
    p = D.LTXPipeline(**{k: getattr(pipe, k) for k in ("transformer", "vae", "scheduler", "text_encoder", "tokenizer")})
    return p, cfg
