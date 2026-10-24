#!/usr/bin/env bash
# One job of the LTX-2B full plan on one fresh pod (a 24 GB card is enough; gen1..gen4 and probe run on separate pods):
# setup (outside the budget), the job's steps (the last one scores), push, verify, stop the pod.
#   cd /workspace && git clone https://github.com/Cypher193/xakfdghdfjk.git cvpr_vail && cd cvpr_vail
#   setsid nohup bash scripts/ltx_pod.sh gen1 6 > /workspace/ltx.out 2>&1 < /dev/null &      # job gen1|gen2|gen3|gen4|probe, budget (h)
# LTX_JOB makes sync.py pull the inputs plus the job's files (probe: every gen job's LTX runs) and push only the job's own.
# The probe job must start after gen1..gen4 have pushed. Logs carry the _ltx_{job} suffix. Heartbeats go to $MONITOR_URL if set.
set -uo pipefail
cd "$(dirname "$0")/.."
JOB=${1:?usage: ltx_pod.sh gen1|gen2|gen3|gen4|probe BUDGET_H}
BUDGET=${2:?usage: ltx_pod.sh gen1|gen2|gen3|gen4|probe BUDGET_H}
source scripts/pod_env.sh
export LTX_JOB=$JOB
S=_ltx_$JOB
mkdir -p data/logs
utc_in() { python -c "import datetime as d; print((d.datetime.now(d.timezone.utc) + d.timedelta(hours=$1)).strftime('%Y-%m-%dT%H:%MZ'))"; }
# Crash guard: if this script or its children die without stopping the pod, push to HF and stop the pod
setsid nohup python scripts/guard.py --tag "$S" --hard-at "$(utc_in "$BUDGET + 2")" > /workspace/guard.out 2>&1 < /dev/null &
bash scripts/setup_runpod.sh > "data/logs/setup$S.log" 2>&1 || { echo "setup failed"; exit 1; }
STOP=$(utc_in "$BUDGET + 0.25")      # backstop only: run_pilot already skips steps that do not fit the budget
HARD=$(utc_in "$BUDGET + 1.0")
python scripts/run_pilot.py --plan ltx --job "$JOB" --budget-h "$BUDGET" &
# Waits for run_pilot (or STOP), uploads, verifies on HF, then stops the pod (finalize.stop_pod).
python scripts/finalize.py --stop-at "$STOP" --hard-at "$HARD" --plan ltx --job "$JOB"
echo "LTX POD DONE ($JOB)"
