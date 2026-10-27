# One image for every pod: rendering (Blender) and generation (torch 2.8 / CUDA 12.8).
# Holds dependencies only; the code is cloned on the pod (/workspace/cvpr_vail). Every new vast.ai instance pulls it, so
# it stays small: the official PyTorch runtime base (no CUDA toolkit: torch's wheels carry the CUDA libraries they use;
# a common image, so hosts often have it cached) plus sshd for vast's SSH launch mode. The RunPod base it replaced
# (2026-10-27) was 11 GB compressed with one 4 GB torch layer that stalled on cold hosts.
# Needs a host driver that supports CUDA 12.8 (vast: cuda_vers >= 12.8).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV HF_HOME=/workspace/hf \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    BPY_PYTHON=/opt/bpy/bin/python

# sshd (vast.ai's SSH mode starts it; the image's own entrypoint is replaced), ffmpeg for video I/O, git for the clone,
# X/GL libraries the headless bpy wheel links against
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server ffmpeg git ca-certificates \
        libxrender1 libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libsm6 libice6 libgl1 libegl1 \
    && mkdir -p /run/sshd && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/cvpr_vail/requirements.txt
RUN pip install --no-cache-dir -r /opt/cvpr_vail/requirements.txt

# Blender as a Python module: bpy 5.2 needs Python 3.13, so it gets its own venv
RUN pip install --no-cache-dir uv \
    && uv venv --python 3.13 /opt/bpy \
    && uv pip install --python /opt/bpy --no-cache bpy==5.2.2 numpy pyyaml \
    && /opt/bpy/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"

RUN python -c "import torch, diffusers, transformers; print('torch', torch.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)" \
    && which sshd
