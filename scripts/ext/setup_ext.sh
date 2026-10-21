#!/usr/bin/env bash
# External-repo models for the screen (not in diffusers), each in its own venv under $EXT_ROOT (/workspace/ext):
#   bash scripts/ext/setup_ext.sh magi selfforcing      -> $EXT_ROOT/{name}/{repo,venv,...}
# Repos and weights are pinned. Each venv gets torch 2.7.1 cu128 (the pod driver supports CUDA 12.8; an unpinned
# torch once pulled 2.14 and CUDA became unavailable) plus the flash-attn 2.8.3 wheel built for it. Weights are
# downloaded with the main environment's huggingface_hub. Re-running skips finished parts.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXT=${EXT_ROOT:-/workspace/ext}
TORCH_PINS="torch==2.7.1 torchvision==0.22.1"
# Triton compiles a C helper against the venv's base Python (system 3.10): it needs Python.h
[ -f /usr/include/python3.10/Python.h ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.10-dev; }
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3

clone() {  # url sha dir
  [ -d "$3/.git" ] || git clone -q "$1" "$3"
  git -C "$3" fetch -q origin "$2" 2>/dev/null || true
  git -C "$3" checkout -q "$2"
}

venv() {   # dir: python 3.10 venv with pinned cu128 torch, flash-attn for torch's C++ ABI, then "$@" packages
  local d=$1; shift
  [ -x "$d/venv/bin/python" ] || uv venv -q --python 3.10 --seed "$d/venv"
  printf '%s\n' $TORCH_PINS > "$d/constraints.txt"
  uv pip install -q --python "$d/venv/bin/python" $TORCH_PINS --index-url https://download.pytorch.org/whl/cu128
  local abi
  abi=$("$d/venv/bin/python" -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
  uv pip install -q --python "$d/venv/bin/python" -c "$d/constraints.txt" --no-deps \
    "$FA/flash_attn-2.8.3+cu12torch2.7cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
  uv pip install -q --python "$d/venv/bin/python" -c "$d/constraints.txt" "$@" imageio==2.36.1 imageio-ffmpeg==0.5.1
  "$d/venv/bin/python" -c "import torch, flash_attn
assert torch.__version__.startswith('2.7.1') and torch.cuda.is_available(), (torch.__version__, torch.cuda.is_available())
print('venv ok: torch', torch.__version__, 'cuda', torch.version.cuda, 'flash_attn', flash_attn.__version__)"
}

magi() {   # SandAI-org/MAGI-1 (Apache 2.0), 4.5B-distill; MagiAttention is not needed on A100/A6000 (sm < 90)
  local d=$EXT/magi
  clone https://github.com/SandAI-org/MAGI-1.git 0fcefdef8ce2df37a3b8890979433c06eb003328 "$d/repo"
  venv "$d" numpy==1.26.4 einops==0.8.0 ffmpeg-python==0.2.0 safetensors==0.4.5 tqdm transformers==4.42.3 \
    huggingface_hub==0.25.2 sentencepiece==0.2.0 protobuf==5.28.3 ftfy==6.2.0 beautifulsoup4==4.13.4 timm==1.0.15 \
    diffusers==0.29.2 accelerate==0.32.1
  python - "$d/ckpt" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
t5 = ["config.json", "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
      "pytorch_model.bin.index.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"]
dit = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"]
files = ["ckpt/vae/config.json", "ckpt/vae/diffusion_pytorch_model.safetensors"] + \
        [f"ckpt/t5/t5-v1_1-xxl/{f}" for f in t5] + [f"ckpt/magi/4.5B_distill/inference_weight.distill/{f}" for f in dit]
for f in files:   # explicit list: the repo also holds other sizes and a stray T5 shard (~100 GB we do not need)
    hf_hub_download("sand-ai/MAGI-1", f, revision="ca8bdfb28b1d88a29ec2a3ea423029f801a891bd", local_dir=sys.argv[1])
EOF
}

selfforcing() {   # guandeh17/Self-Forcing (Apache 2.0): causal Wan2.1-T2V-1.3B, DMD checkpoint
  local d=$EXT/selfforcing
  clone https://github.com/guandeh17/Self-Forcing.git 33593df3e81fa3ec10239271dd2c100facac6de1 "$d/repo"
  venv "$d" diffusers==0.31.0 transformers==4.49.0 tokenizers==0.21.0 numpy==1.24.4 omegaconf==2.3.0 einops==0.8.0 \
    ftfy==6.3.1 regex==2024.11.6 easydict==1.13 sentencepiece==0.2.0 tqdm
  python - "$d/repo" <<'EOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
repo = sys.argv[1]
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", revision="37ec512624d61f7aa208f7ea8140a131f93afc9a",
                  local_dir=f"{repo}/wan_models/Wan2.1-T2V-1.3B")
hf_hub_download("gdhe17/Self-Forcing", "checkpoints/self_forcing_dmd.pt",
                revision="2f8b779212da279d212c22a509b66ad6552f350e", local_dir=repo)
EOF
}

[ $# -gt 0 ] || { echo "usage: setup_ext.sh magi|selfforcing ..."; exit 2; }
for m in "$@"; do
  case $m in
    magi|selfforcing) echo "== $m"; $m ;;
    *) echo "unknown external model: $m"; exit 2 ;;
  esac
done
df -h / | tail -1
