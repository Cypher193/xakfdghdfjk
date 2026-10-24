# Source in SSH sessions: they do not inherit the container's environment (HF_TOKEN secret, image ENV).
# vast.ai SSH mode: the template's onstart also writes them to /etc/environment (CONTAINER_* = this instance's ID and key)
set -a
source <( { tr '\0' '\n' < /proc/1/environ; cat /etc/environment 2>/dev/null; } \
  | grep -E '^(RUNPOD_POD_ID|CONTAINER_ID|CONTAINER_API_KEY|HF_TOKEN|HF_HOME|BPY_PYTHON|UV_PYTHON_INSTALL_DIR|PYTHONUNBUFFERED)=')
set +a
export HF_HOME=${HF_HOME:-/workspace/hf}
