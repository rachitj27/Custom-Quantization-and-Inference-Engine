#!/usr/bin/env bash
# Score both weight-quantization schemes over the test set.
#
#   ./run_both.sh
#
# Produces preds_pt/detections.csv (per-tensor weights) and
# preds_pc/detections.csv (per-channel weights).
#
# ENGINE and KERNEL override the binary and the convolution kernel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMG_DIR="$ROOT/YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/test/images"
ENGINE="${ENGINE:-$HERE/build/custom_engine}"

run_one() {
  local prefix="$1" outdir="$2"
  local csv="$outdir/detections.csv"
  mkdir -p "$outdir"
  : > "$csv"
  echo "=== $prefix ==="
  local start
  start=$(date +%s)
  local n=0
  while IFS= read -r img; do
    "$ENGINE" "$img" ${KERNEL:+--kernel "$KERNEL"} \
      --model-json "$ROOT/quantization/$prefix.json" \
      --model-bin  "$ROOT/quantization/$prefix.bin" \
      -o "$outdir/$(basename "${img%.*}")_pred.jpg" \
      --csv-append "$csv" > /dev/null
    n=$((n + 1))
    if (( n % 10 == 0 )); then echo "  ...$n"; fi
  done < <(find "$IMG_DIR" -maxdepth 1 -type f -iname '*.jpg' | sort)
  echo "  $n images in $(( $(date +%s) - start ))s, $(wc -l < "$csv") detections"
}

run_one model_int8    "$HERE/build/preds_pt"
run_one model_int8_pc "$HERE/build/preds_pc"
