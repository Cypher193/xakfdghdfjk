#!/usr/bin/env bash
# Generator screen on one fresh pod (template cvpr-vail-pilot: image ghcr.io/cypher193/cvpr_vail, 300 GB container
# disk, no volume): setup (outside the budget), then the screen plan for one tier, then push, verify, stop the pod.
#   cd /workspace && git clone https://github.com/Cypher193/xakfdghdfjk.git cvpr_vail && cd cvpr_vail
#   setsid nohup bash scripts/screen_pod.sh A 10 > /workspace/screen.out 2>&1 < /dev/null &      # tier, budget (h)
# Tiers A and B can run on two pods at once: SCREEN_TIER makes sync.py pull the inputs plus this tier's files and
# push only those, and every log, budget and table file carries the _screen_{tier} suffix.
set -uo pipefail
cd "$(dirname "$0")/.."
TIER=${1:?usage: screen_pod.sh A|B|all BUDGET_H}
BUDGET=${2:?usage: screen_pod.sh A|B|all BUDGET_H}
source scripts/pod_env.sh
export SCREEN_TIER=$TIER
S=_screen_$TIER
mkdir -p data/logs
utc_in() { python -c "import datetime as d; print((d.datetime.now(d.timezone.utc) + d.timedelta(hours=$1)).strftime('%Y-%m-%dT%H:%MZ'))"; }
# Crash guard: if this script or its children die without stopping the pod, push to HF and stop the pod
setsid nohup python scripts/guard.py --tier "$TIER" --hard-at "$(utc_in "$BUDGET + 2")" > /workspace/guard.out 2>&1 < /dev/null &
bash scripts/setup_runpod.sh > "data/logs/setup$S.log" 2>&1 || { echo "setup failed"; exit 1; }
if [ "$TIER" != B ]; then                        # only Cosmos 3 Nano (tier A) uses the guardrail
  python -c "from cosmos_guardrail import CosmosSafetyChecker; CosmosSafetyChecker()" > "data/logs/guardrail$S.log" 2>&1 \
    || { echo "guardrail failed (see data/logs/guardrail$S.log)"; exit 1; }
fi
EXT=$(python -c "import sys; sys.path.insert(0, 'scripts'); from run_pilot import screen_ext; print(*screen_ext('$TIER'))")
if [ -n "$EXT" ]; then
  bash scripts/ext/setup_ext.sh $EXT > "data/logs/setup_ext$S.log" 2>&1 || { echo "external setup failed"; exit 1; }
fi
STOP=$(utc_in "$BUDGET + 0.25")      # backstop only: run_pilot already skips steps that do not fit the budget
HARD=$(utc_in "$BUDGET + 1.0")
python scripts/run_pilot.py --plan screen --tier "$TIER" --budget-h "$BUDGET" "${@:3}" &   # e.g. --skip ltx2
# Waits for run_pilot (or STOP), uploads, verifies on HF, then stops the pod (finalize.stop_pod).
python scripts/finalize.py --stop-at "$STOP" --hard-at "$HARD" --plan screen --tier "$TIER"
echo "SCREEN POD DONE ($TIER)"
