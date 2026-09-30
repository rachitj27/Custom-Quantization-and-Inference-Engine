#!/usr/bin/env bash
# Run the custom engine over a directory of images.
#
#   ./run_testset.sh <image_dir> <output_dir> [max_images]
#
# ENGINE overrides the binary (e.g. build-cuda/custom_engine) and KERNEL
# picks the convolution kernel. Without KERNEL the engine uses its own
# default, which is vnni-int8 -- so a GPU run scored without it silently
# measures the CPU path.
#
# Writes annotated JPEGs into <output_dir> plus a detections.csv of every box in
# "image,class_id,class_name,conf,x1,y1,x2,y2" form (640x640 network pixels),
# which quantization/eval_map.py consumes to score the engine.
set -euo pipefail

IMG_DIR="${1:?usage: run_testset.sh <image_dir> <output_dir> [max_images]}"
OUT_DIR="${2:?usage: run_testset.sh <image_dir> <output_dir> [max_images]}"
MAX="${3:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${ENGINE:-$HERE/build/custom_engine}"

if [[ ! -x "$ENGINE" ]]; then
  echo "Engine not built: $ENGINE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/detections.csv"
: > "$CSV"

mapfile -t IMAGES < <(find "$IMG_DIR" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)

if (( MAX > 0 )); then
  IMAGES=("${IMAGES[@]:0:MAX}")
fi

echo "Running engine over ${#IMAGES[@]} images..."
start=$(date +%s)

for i in "${!IMAGES[@]}"; do
  img="${IMAGES[$i]}"
  base="$(basename "$img")"
  stem="${base%.*}"
  "$ENGINE" "$img" ${KERNEL:+--kernel "$KERNEL"} -o "$OUT_DIR/${stem}_pred.jpg" --csv-append "$CSV" > /dev/null
  if (( (i + 1) % 10 == 0 )); then
    echo "  ...$((i + 1))/${#IMAGES[@]}"
  fi
done

end=$(date +%s)
n=$(wc -l < "$CSV")
echo "Processed ${#IMAGES[@]} images in $((end - start))s -- $n detections"
echo "Wrote $CSV"
