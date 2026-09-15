import numpy as np
import os

BUILD_DIR = r"C:\Users\rachi\OneDrive\coding\EdgeAI"

layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 19, 21]

print(f"{'Layer':<8} {'N':>10} {'Exact':>8} {'Within±1':>10} {'Within±2':>10} {'MAE':>8} {'Max|diff|':>10}")
print("-" * 70)

for idx in layers:
    ref_path = os.path.join(BUILD_DIR, f"test_L{idx:02d}_ref.bin")
    cpp_path = os.path.join(BUILD_DIR, f"test_L{idx:02d}_cpp.bin")

    if not (os.path.exists(ref_path) and os.path.exists(cpp_path)):
        print(f"L{idx:02d}    (missing files)")
        continue

    ref = np.fromfile(ref_path, dtype=np.int8).astype(np.int32)
    cpp = np.fromfile(cpp_path, dtype=np.int8).astype(np.int32)

    if ref.size != cpp.size:
        print(f"L{idx:02d}    SIZE MISMATCH ref={ref.size} cpp={cpp.size}")
        continue

    diff = np.abs(ref - cpp)
    exact = (diff == 0).mean() * 100
    within1 = (diff <= 1).mean() * 100
    within2 = (diff <= 2).mean() * 100
    mae = diff.mean()
    maxdiff = diff.max()

    print(f"L{idx:02d}     {ref.size:>10} {exact:>7.2f}% {within1:>9.2f}% {within2:>9.2f}% {mae:>8.3f} {maxdiff:>10}")