#!/usr/bin/env bash
# P11 timing for the two models used only for budgeting: Cosmos-Predict2.5-14B and Cosmos 3 Nano, both in the main
# environment (diffusers 0.41, model CPU offload, guardrail on). Frees the Wan caches first (their runs are done by now).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/pod_env.sh
rm -rf "$HF_HOME"/hub/models--Wan-AI--* && echo "freed Wan caches"; df -h / | tail -1
timeout 2400 python -m pilot.generate --model cosmos14b --exp b1 --limit 1 --seeds 0 --no-capture; echo "cosmos14b exit $?"
rm -rf "$HF_HOME"/hub/models--nvidia--Cosmos-Predict2.5-14B
timeout 2400 python -m pilot.generate --model cosmos3_nano --exp b1 --limit 1 --seeds 0 --no-capture; echo "cosmos3_nano exit $?"
