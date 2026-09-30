#!/usr/bin/env bash
# Set up a Colab runtime to build and run the CUDA engine.
#
#   !bash colab_bootstrap.sh [branch]
#
# Then, per iteration:
#   !cd /content/engine && git pull && cmake --build cpp_engine/build-cuda -j4
#
# Colab wipes /content on disconnect, so the git branch is the source of
# truth. A clone alone cannot run the engine -- .gitignore excludes *.bin and
# *.pt -- so the model, test input and best.pt come from the release tarball.
set -euo pipefail

# step out of anything this deletes, or a notebook %cd'd into the clone
# loses its working directory when ENGINE_DIR goes
cd / 2>/dev/null || true

BRANCH="${1:-cuda-gemm-speedup}"
REPO="https://github.com/rachitj27/Custom-Quantization-and-Inference-Engine.git"
ASSET_URL="https://github.com/rachitj27/Custom-Quantization-and-Inference-Engine/releases/download/engine-assets-v1/engine-assets.tar.gz"
UPSTREAM="https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection.git"

ENGINE_DIR=/content/engine
SRC_DIR=/content/src

echo "=== GPU ==="
if ! command -v nvidia-smi > /dev/null; then
  echo "No nvidia-smi. This is a CPU runtime -- switch to a GPU runtime" >&2
  echo "(Runtime > Change runtime type > T4 GPU) and re-run." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="${CAP//./}"
if [[ -z "$ARCH" ]]; then
  echo "Could not read the compute capability" >&2
  exit 1
fi
# dp4a needs sm_61
if (( ARCH < 61 )); then
  echo "Compute capability $CAP is below the sm_61 floor for dp4a" >&2
  exit 1
fi
echo "Building for sm_$ARCH"

echo
echo "=== toolchain ==="
nvcc --version | tail -2
# Colab's gcc is already one CUDA accepts, so nothing to pin here
apt-get -qq install -y nlohmann-json3-dev > /dev/null
echo "nlohmann-json installed, cmake $(cmake --version | head -1 | awk '{print $3}')"

echo
echo "=== source ==="
rm -rf "$ENGINE_DIR"
git clone -q --branch "$BRANCH" --depth 1 "$REPO" "$ENGINE_DIR"
echo "cloned $BRANCH at $(git -C "$ENGINE_DIR" rev-parse --short HEAD)"

echo
echo "=== model assets ==="
curl -sSL "$ASSET_URL" | tar xz -C "$ENGINE_DIR"
for f in quantization/model_int8_pc.bin quantization/model_int8_pc.json test_input.bin best.pt; do
  if [[ ! -f "$ENGINE_DIR/$f" ]]; then
    echo "missing after unpack: $f" >&2
    exit 1
  fi
done
echo "model_int8_pc.bin $(stat -c%s "$ENGINE_DIR/quantization/model_int8_pc.bin") bytes"
echo "test_input.bin    $(stat -c%s "$ENGINE_DIR/test_input.bin") bytes"

echo
echo "=== dataset ==="
# setup_colab.py remaps the 3-class labels to 2 and drops six images.
# skipping it gives mAP 0.0000.
rm -rf "$SRC_DIR"
git clone -q --depth 1 "$UPSTREAM" "$SRC_DIR"
python "$ENGINE_DIR/benchmarks/setup_colab.py"

# setup_colab.py writes /content/fire-8, eval_map.py wants it under the repo
mkdir -p "$ENGINE_DIR/YOLOv8-Fire-and-Smoke-Detection/datasets"
ln -sfn /content/fire-8 "$ENGINE_DIR/YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8"

N_TEST="$(ls "$ENGINE_DIR/YOLOv8-Fire-and-Smoke-Detection/datasets/fire-8/test/images" | wc -l)"
echo "test split visible to eval_map.py: $N_TEST images"
if [[ "$N_TEST" != "49" ]]; then
  echo "expected 49 -- mAP will not be comparable to the recorded rows" >&2
fi

echo
echo "=== python deps ==="
pip install -q ultralytics
echo "ultralytics installed"

echo
echo "=== build ==="
cmake -S "$ENGINE_DIR/cpp_engine" -B "$ENGINE_DIR/cpp_engine/build-cuda" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" > /dev/null
cmake --build "$ENGINE_DIR/cpp_engine/build-cuda" -j4

# CPU baseline. Colab has no AVX-VNNI, so this falls back to scalar-int8 and
# the 233.5 ms VNNI figure is not reproducible here.
cmake -S "$ENGINE_DIR/cpp_engine" -B "$ENGINE_DIR/cpp_engine/build" \
  -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$ENGINE_DIR/cpp_engine/build" -j4 > /dev/null

echo
echo "=== ready ==="
echo "cd $ENGINE_DIR"
echo "  ./cpp_engine/build-cuda/gemm_selftest"
echo "  ./cpp_engine/build-cuda/custom_engine --input-bin test_input.bin --dump-dir dumps_cuda --kernel cuda-fp32"
