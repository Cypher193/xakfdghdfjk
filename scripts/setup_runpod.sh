#!/usr/bin/env bash
# Fresh RunPod pod or vast.ai instance (image ghcr.io/cypher193/cvpr_vail, >= 200 GB disk at /workspace).
# Data persists on the private HF dataset repo (scripts/sync.py); weights are re-downloaded per pod.
# Set HF_TOKEN as a RunPod secret / vast.ai account env var. Never write the token into the repo.
# SCREEN_TIER set (scripts/screen_pod.sh): only SAM2 is pre-downloaded (each screen step fetches its own model),
# sync.py pulls/pushes only this tier's files, and the sync log carries the tier suffix.
set -euo pipefail
mkdir -p /workspace && cd /workspace          # pod volume disk or container disk; no network volume needed
[ -d cvpr_vail ] || git clone https://github.com/Cypher193/xakfdghdfjk.git cvpr_vail
cd cvpr_vail
export HF_HOME=/workspace/hf
pip install -q -r requirements.txt
python - <<'EOF'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
# ~100 GB in total; the text encoders are most of it
repos = [("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"),
         ("Lightricks/LTX-Video-0.9.5", None),
         ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", None),
         ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None),
         ("facebook/sam2.1-hiera-large", None)]
for repo, rev in repos[-1:] if os.environ.get("SCREEN_TIER") else repos:
    print("downloading", repo, flush=True)
    snapshot_download(repo, revision=rev)
EOF
python -m pilot.metrics
python scripts/sync.py pull                     # renders, runs, results from earlier pods
python -m pilot.download
mkdir -p data/logs
pkill -f "scripts/sync.py push" || true
nohup python scripts/sync.py push --every 900 > "data/logs/sync${SCREEN_TIER:+_screen_$SCREEN_TIER}.log" 2>&1 &   # every 15 min
echo "Setup done. Background sync running (data/logs/sync.log). Before terminating a pod: python scripts/sync.py push"
