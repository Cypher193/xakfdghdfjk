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
from huggingface_hub import snapshot_download   # HF_TOKEN is read from the environment; login() would call /whoami-v2,
                                                  # whose strict limit failed relaunched pods' setup (2026-11-06)
# ~100 GB in total; the text encoders are most of it
repos = [("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"),
         ("Lightricks/LTX-Video-0.9.5", None),
         ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", None),
         ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None),
         ("facebook/sam2.1-hiera-large", None)]
if os.environ.get("SCREEN_TIER"):
    repos = repos[-1:]
elif os.environ.get("LTX_JOB"):            # job pods: the job's models + SAM2 only (disk sized for that, downloads billed per GB)
    import sys; sys.path.insert(0, "scripts")
    from run_pilot import job_repos
    repos = job_repos(os.environ["LTX_JOB"])
import time
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
for repo, rev in repos:
    print("downloading", repo, flush=True)
    for attempt in range(12):             # HF's 1,000 API calls per 5 min are shared by every pod (2026-11-06)
        try:
            snapshot_download(repo, revision=rev)
            break
        except (HfHubHTTPError, LocalEntryNotFoundError) as e:
            if "429" not in str(e) or attempt == 11:
                raise
            print(f"HF rate limit, retry {attempt + 1} in 90 s", flush=True)
            time.sleep(90)
EOF
python -m pilot.metrics
python scripts/sync.py pull                     # renders, runs, results from earlier pods
python -m pilot.download
mkdir -p data/logs
pkill -f "scripts/sync.py push" || true
nohup python scripts/sync.py push --every "${SYNC_EVERY:-900}" > "data/logs/sync${SCREEN_TIER:+_screen_$SCREEN_TIER}${LTX_JOB:+_ltx_$LTX_JOB}.log" 2>&1 &   # every 15 min
echo "Setup done. Background sync running (data/logs/sync.log). Before terminating a pod: python scripts/sync.py push"
