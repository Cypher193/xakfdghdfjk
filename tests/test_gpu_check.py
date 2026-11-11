"""gpu_check verdicts on faked nvidia-smi readings and benchmark results (no GPU needed).
    python -m tests.test_gpu_check"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_check  # noqa: E402

OS_RAM = 128e9


def run(tflops, power_limit=300, draw=290, clock=1700, slowdown="Not Active", ram=OS_RAM, min_ram=0):
    row = {"name": "NVIDIA A40", "power.limit": power_limit, "power.default_limit": 300.0, "power.draw": draw,
           "clocks.sm": clock, "clocks.max.sm": 1740.0, "temperature.gpu": 70.0}
    row |= {f"clocks_throttle_reasons.{r}": slowdown for r in gpu_check.BAD_REASONS}
    gpu_check.smi = lambda: dict(row)
    gpu_check.bench = lambda s: (time.sleep(0.2), tflops)[1]     # long enough for one sample
    gpu_check.os.sysconf = lambda k: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": ram / 4096}[k]
    return gpu_check.check(0.1, min_ram)


healthy = run(110)
assert healthy["ok"], healthy["fails"]
assert not run(40)["ok"], "slow matmul must fail"
assert not run(110, power_limit=200)["ok"], "host-lowered power limit must fail"
assert not run(110, clock=900, draw=200)["ok"], "low SM clock while under the power limit must fail"
assert run(110, clock=900, draw=295)["ok"], "low SM clock at the power limit is normal power capping"
assert not run(110, draw=120)["ok"], "low power draw under load must fail"
assert not run(110, slowdown="Active")["ok"], "thermal or hardware slowdown must fail"
assert not run(110, ram=64e9, min_ram=80)["ok"], "too little host RAM must fail"
assert gpu_check.spec_for("NVIDIA RTX A6000") == 154.8 and gpu_check.spec_for("NVIDIA A100 80GB PCIe") == 312.0
print("gpu_check tests passed")
