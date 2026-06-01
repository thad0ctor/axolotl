#!/usr/bin/env bash
# Reusable NVFP4 training benchmark that always lands on the correct path.
#
# Two traps this script exists to avoid:
#   1. `axolotl train` defaults to the `accelerate` launcher, which spawns a
#      NEW python (often base conda, no MSLK) -> the FP4 quant silently uses the
#      slow torchao path and numbers are meaningless. We force `--launcher python`
#      (in-process) so the venv that runs this script (with MSLK) is the one used.
#   2. A fast-but-diverging run is worse than useless. We parse the loss and
#      REFUSE to report a run whose loss diverged (here: any logged loss > 5 or
#      non-finite), printing DIVERGED instead of a throughput number.
#
# Usage:
#   scripts/bench_nvfp4.sh [--gpu N] [--steps N] [--venv PATH] cfg1.yaml [cfg2.yaml ...]
#
# Each cfg is run once after a short warmup (Triton/inductor disk cache). Steady
# s/step is the median of steps after the warmup window; memory is the peak
# device-reserved / max-active logged by axolotl.
set -euo pipefail

GPU=6                # 5090 (sm_120). NEVER the RTX 6000 Blackwell (idx 0,3) for timing.
STEPS=40
VENV="${VENV:-/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_nvfp4}"
CFGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    *) CFGS+=("$1"); shift;;
  esac
done
[[ ${#CFGS[@]} -gt 0 ]] || { echo "usage: $0 [--gpu N] [--steps N] cfg.yaml ..."; exit 2; }

# Repo src that owns this script -> PYTHONPATH, so the in-process run imports the
# branch under test (the venv's editable install may point elsewhere).
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src"
PY="$VENV/bin/python"
AX="$VENV/bin/axolotl"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 9B+ needs this to not OOM
export PYTHONPATH="$SRC"

# Self-check: is MSLK present in the venv? Without it the FP4 path is unrepresentative.
if "$PY" -c "import mslk" 2>/dev/null; then
  echo "[ok] MSLK present in venv ($("$PY" -c 'import torch;print(torch.__version__)'))"
else
  echo "[WARN] MSLK NOT importable in $VENV — FP4 path will use the slow torchao quant; numbers not representative."
fi
echo "[ok] launcher=python  gpu=$GPU(PCI_BUS_ID)  src=$SRC"

LOGDIR="$(mktemp -d /tmp/bench_nvfp4.XXXXXX)"
# Warm the Triton/inductor disk cache with the first config (6 steps, throwaway).
WARM="$LOGDIR/warm.yaml"; sed "s/max_steps: .*/max_steps: 6/" "${CFGS[0]}" > "$WARM"
echo "[warmup] ${CFGS[0]} ..."; "$AX" train "$WARM" --launcher python > "$LOGDIR/warm.log" 2>&1 || true

printf '\n%-28s %10s %10s %10s %10s\n' CONFIG "s/step" "active GiB" "resv GiB" "loss"
for cfg in "${CFGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  log="$LOGDIR/$name.log"
  "$AX" train "$cfg" --launcher python > "$log" 2>&1 || true
  "$PY" - "$name" "$log" "$STEPS" <<'PY'
import re, sys, statistics
name, log, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(log).read()
# Hard failure first: a crash/OOM leaves a traceback and no train summary. Check
# this before anything else (model-load "shards/s" bars also match the it/s regex,
# so step-count alone can't tell a failed run from a real one).
failed = ("OutOfMemoryError" in s or "Traceback (most recent call last)" in s) \
    and "'train_loss'" not in s
losses = [float(x) for x in re.findall(r"'loss': '([0-9.eE+-]+)'", s)]
# Count only TRAINING step rates: slice from the first loss log onward, so the
# model-load "Loading checkpoint shards … it/s" bars (which precede training) are
# excluded without needing the rate and the loss to share a line.
train = s[s.find("'loss':"):] if "'loss':" in s else ""
its = [1/float(x) for x in re.findall(r"([0-9.]+)it/s\]", train)] + \
      [float(x) for x in re.findall(r"([0-9.]+)s/it\]", train)]
warm = max(10, steps // 3)
steady = its[warm:] if len(its) > warm + 2 else its
resv = [float(x) for x in re.findall(r"device_reserved \(GiB\)': '([0-9.]+)'", s)]
act  = [float(x) for x in re.findall(r"max_active \(GiB\)': '([0-9.]+)'", s)]
diverged = any((l != l) or l > 5.0 for l in losses) if losses else False
if failed:
    err = next((ln for ln in s.splitlines()
                if "OutOfMemory" in ln or ("Error" in ln and "telemetry" not in ln)),
               "run failed")
    print(f"{name:<28} {'FAILED':>10}  ({err.strip()[:60]})")
elif diverged:
    print(f"{name:<28} {'DIVERGED':>10}  (loss up to {max(losses):.1f} — invalid; check config)")
elif steady:
    sps = statistics.median(steady)
    print(f"{name:<28} {sps:>10.3f} {max(act) if act else 0:>10.2f} "
          f"{max(resv) if resv else 0:>10.2f} {losses[-1] if losses else 0:>10.3f}")
else:
    print(f"{name:<28} {'NO DATA':>10}")
PY
done
echo
echo "logs: $LOGDIR"
