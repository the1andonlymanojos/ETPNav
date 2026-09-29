#!/bin/bash
# Run a command politely on a shared machine.
#   1. Pre-flight: refuse to start if RAM/GPU/load is already tight (exit 75) or another compute job holds the GPU.
#   2. Run at lowest CPU/IO priority; under systemd-run hard caps (RAM + CPU quota) when the user manager is available.
#   3. Report this command's peak RAM / VRAM footprint (system delta vs. the pre-flight baseline).
# Knobs (env): MIN_AVAIL_GB=8  MAX_GPU_USED_MB=5000  MAX_LOAD=16  RAM_CAP=8G  CPU_QUOTA=300%
# Usage: scripts/guarded_run.sh python run.py ...
set -u
MIN_AVAIL_GB=${MIN_AVAIL_GB:-8}; MAX_GPU_USED_MB=${MAX_GPU_USED_MB:-5000}; MAX_LOAD=${MAX_LOAD:-16}
RAM_CAP=${RAM_CAP:-8G}; CPU_QUOTA=${CPU_QUOTA:-300%}

avail=$(free -g | awk '/Mem/{print $7}')
used_mb0=$(free -m | awk '/Mem/{print $3}')
gpu0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
echo "[guard] avail RAM ${avail}G | GPU used ${gpu0}MiB, compute procs ${gpu_procs} | load ${load}"
if [ "$avail" -lt "$MIN_AVAIL_GB" ] || [ "$gpu0" -gt "$MAX_GPU_USED_MB" ] || [ "$gpu_procs" -gt 0 ] || [ "$load" -gt "$MAX_LOAD" ]; then
  echo "[guard] machine looks busy -> NOT starting (exit 75)"; exit 75
fi

SAMP=$(mktemp)
( while true; do echo "$(free -m | awk '/Mem/{print $3}') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')" >> "$SAMP"; sleep 2; done ) &
SPID=$!
trap 'kill $SPID 2>/dev/null; rm -f "$SAMP"' EXIT

WRAP=""
if command -v systemd-run >/dev/null && systemd-run --user --scope -q -p MemoryMax=1G true >/dev/null 2>&1; then
  WRAP="systemd-run --user --scope -q -p MemoryMax=$RAM_CAP -p MemorySwapMax=0 -p CPUQuota=$CPU_QUOTA"
  echo "[guard] hard caps on: RAM<=$RAM_CAP (no swap), CPU<=$CPU_QUOTA"
else
  echo "[guard] systemd-run unavailable -> nice/ionice only (no hard caps)"
fi

start=$(date +%s)
$WRAP nice -n 19 ionice -c3 "$@"; rc=$?
dur=$(( $(date +%s) - start ))

peak_ram=$(awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m}' "$SAMP")
peak_gpu=$(awk 'BEGIN{m=0} {if($2>m)m=$2} END{print m}' "$SAMP")
echo "[guard] finished rc=$rc in ${dur}s | peak footprint vs baseline: RAM +$(( peak_ram - used_mb0 ))MiB, VRAM +$(( peak_gpu - gpu0 ))MiB"
exit $rc
