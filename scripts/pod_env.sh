# Source in SSH sessions: they do not inherit the container's environment (HF_TOKEN secret, image ENV).
set -a
source <(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|HF_TOKEN|HF_HOME|BPY_PYTHON|UV_PYTHON_INSTALL_DIR|PYTHONUNBUFFERED)=')
set +a
export HF_HOME=${HF_HOME:-/workspace/hf}
