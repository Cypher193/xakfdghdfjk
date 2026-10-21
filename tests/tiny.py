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


def skyreels():
    """Tiny SkyReels-V2 DF V2V (components from diffusers v0.41.0 tests/pipelines/skyreels_v2)."""
    torch.manual_seed(0)
    te, tok = _t5()
    tr = D.SkyReelsV2Transformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                        in_channels=16, out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32,
                                        num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                        rope_max_seq_len=32)
    pipe = D.SkyReelsV2DiffusionForcingVideoToVideoPipeline(
        transformer=tr, vae=_wan_vae(), scheduler=D.UniPCMultistepScheduler(use_flow_sigmas=True),
        text_encoder=te, tokenizer=tok)
    cfg = dict(repo="tiny", input="skyreels_df", scheduler="UniPCMultistepScheduler", flow_shift=8.0, fps=24,
               num_frames=9, height=32, width=32, steps=8, guidance=6.0, cond_frames=5, blocks=["blocks"],
               patch=[1, 2, 2], vae_stride=[4, 8, 8], cfg="separate")
    return pipe, cfg


def ltx2():
    """Tiny LTX-2 condition pipeline (diffusers v0.41.0 tests/pipelines/ltx2/testing_utils.py). The test VAE is
    swapped for a 3-level spatiotemporal one: the test's VAE does not compress time but its decoder doubles it."""
    from transformers import Gemma3ForConditionalGeneration
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-gemma3")
    te = Gemma3ForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-gemma3")
    hid, n_layers = te.config.text_config.hidden_size, te.config.text_config.num_hidden_layers
    torch.manual_seed(0)
    tr = D.LTX2VideoTransformer3DModel(
        in_channels=4, out_channels=4, patch_size=1, patch_size_t=1, num_attention_heads=2, attention_head_dim=8,
        cross_attention_dim=16, audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
        audio_attention_head_dim=4, audio_cross_attention_dim=8, num_layers=2, qk_norm="rms_norm_across_heads",
        caption_channels=hid, rope_double_precision=False, rope_type="split")
    conn = LTX2TextConnectors(
        caption_channels=hid, text_proj_in_factor=n_layers + 1, video_connector_num_attention_heads=4,
        video_connector_attention_head_dim=8, video_connector_num_layers=1, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=4, audio_connector_attention_head_dim=8, audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=None, connector_rope_base_seq_len=32, rope_theta=10000.0,
        rope_double_precision=False, causal_temporal_positioning=False, rope_type="split")
    vae = D.AutoencoderKLLTX2Video(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 8, 8),
        decoder_block_out_channels=(8, 8, 8), layers_per_block=(1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1),
        spatio_temporal_scaling=(True, True, True), decoder_spatio_temporal_scaling=(True, True, True),
        decoder_inject_noise=(False,) * 4, downsample_type=("spatiotemporal",) * 3, upsample_residual=(False,) * 3,
        upsample_factor=(1,) * 3, timestep_conditioning=False, patch_size=1, patch_size_t=1, encoder_causal=True,
        decoder_causal=False)
    vae.use_framewise_encoding = vae.use_framewise_decoding = False
    avae = D.AutoencoderKLLTX2Audio(
        base_channels=4, output_channels=2, ch_mult=(1,), num_res_blocks=1, attn_resolutions=None, in_channels=2,
        resolution=32, latent_channels=2, norm_type="pixel", causality_axis="height", dropout=0.0,
        mid_block_add_attention=False, sample_rate=16000, mel_hop_length=160, is_causal=True, mel_bins=8)
    voc = LTX2Vocoder(
        in_channels=avae.config.output_channels * avae.config.mel_bins, hidden_channels=32, out_channels=2,
        upsample_kernel_sizes=[4, 4], upsample_factors=[2, 2], resnet_kernel_sizes=[3], resnet_dilations=[[1, 3, 5]],
        leaky_relu_negative_slope=0.1, output_sampling_rate=16000)
    pipe = D.LTX2ConditionPipeline(scheduler=D.FlowMatchEulerDiscreteScheduler(), vae=vae, audio_vae=avae,
                                   text_encoder=te, tokenizer=tok, connectors=conn, transformer=tr, vocoder=voc)
    cfg = dict(repo="tiny", input="ltx2_conditions", fps=24, num_frames=17, height=32, width=32, steps=8, guidance=3.0,
               cond_frames=9, blocks=["transformer_blocks"], patch=[1, 1, 1], vae_stride=[8, 8, 8], cfg="batched",
               cond_batch_index=1)
    return pipe, cfg


def helios():
    """Tiny Helios (diffusers v0.41.0 tests/pipelines/helios/test_helios.py). Needs >= 33 input frames."""
    torch.manual_seed(0)
    te, tok = _t5()
    tr = D.HeliosTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                                    out_channels=16, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2,
                                    cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_dim=(4, 4, 4),
                                    has_multi_term_memory_patch=True, guidance_cross_attn=True,
                                    zero_history_timestep=True, is_amplify_history=False)
    pipe = D.HeliosPipeline(transformer=tr, vae=_wan_vae(), text_encoder=te, tokenizer=tok,
                            scheduler=D.HeliosScheduler(stage_range=[0, 1], stages=1, use_dynamic_shifting=True))
    cfg = dict(repo="tiny", input="helios_v2v", fps=24, num_frames=66, height=32, width=32, steps=8, guidance=5.0,
               cond_frames=33, blocks=["blocks"], tokens="last", grid_frames=33, obj_tokens=False, patch=[1, 2, 2],
               vae_stride=[4, 8, 8], cfg="separate")
    return pipe, cfg


def cosmos3():
    """Tiny Cosmos 3 Omni (diffusers v0.41.0 tests/pipelines/cosmos/test_cosmos3.py). The scheduler takes the
    nvidia/Cosmos3-Nano flow config: the test's default epsilon UniPC lets the clean conditioning latents drift."""
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosSafetyChecker

    class Checker(D.ModelMixin, D.ConfigMixin, CosmosSafetyChecker):
        """Test-only stand-in; subclasses CosmosSafetyChecker because the pipeline gates both checks on isinstance."""
        def __init__(self):
            D.ModelMixin.__init__(self)
            self.register_buffer("_t", torch.zeros(1), persistent=False)

        def check_text_safety(self, prompt):
            return True

        def check_video_safety(self, frames):
            return frames

    torch.manual_seed(0)
    tr = D.Cosmos3OmniTransformer(head_dim=6, hidden_act="relu2", hidden_size=6, intermediate_size=12,
                                  latent_channel=16, latent_patch_size=1, num_attention_heads=1, num_hidden_layers=1,
                                  num_key_value_heads=1, patch_latent_dim=16, qk_norm_for_text=False, rms_norm_eps=1e-5,
                                  rope_axes_dim=[1, 1, 1], vocab_size=151657)
    sched = D.UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, use_karras_sigmas=True,
                                      sigma_min=0.147, sigma_max=200.0, final_sigmas_type="zero",
                                      time_shift_type="exponential")
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-cosmos3-modular-pipe", subfolder="text_tokenizer")
    pipe = D.Cosmos3OmniPipeline(transformer=tr, text_tokenizer=tok, vae=_wan_vae(), scheduler=sched,
                                 sound_tokenizer=None, safety_checker=Checker(), enable_safety_checker=True)
    cfg = dict(repo="tiny", input="cosmos3_omni", scheduler="UniPCMultistepScheduler", flow_shift=10.0, fps=24,
               num_frames=9, height=32, width=32, steps=8, guidance=6.0, cond_frames=5, blocks=["layers"], stream=1,
               patch=[1, 1, 1], vae_stride=[4, 8, 8], cfg="separate")
    return pipe, cfg


BUILDERS = {"vace13": vace, "ltx2b": ltx, "cosmos2b": cosmos}
SCREEN_BUILDERS = {"skyreels13": skyreels, "ltx2": ltx2, "helios": helios, "cosmos3_nano": cosmos3}


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
