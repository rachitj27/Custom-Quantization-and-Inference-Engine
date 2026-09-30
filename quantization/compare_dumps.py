"""Diff two directories of engine layer dumps, byte for byte.

compare_layers.py answers "how close is the engine to PyTorch", which is a
human-read number that is never 100% at 8 bits. This answers a different and
sharper question: did swapping the kernel change anything at all. It exits
nonzero when it did, so it works as a gate.

For the INT8 kernels the answer must be zero difference. Integer accumulation
is order-independent, so a kernel that computes the same dot products in a
different order still lands on the same int32, and every float operation after
that runs on the host. Anything else is an indexing bug.

The FP32 kernels are a different story: the reduction order genuinely differs,
so pass --max-abs-diff 1 there.

Usage:
    python quantization/compare_dumps.py --a dumps_scalar_int8 --b dumps_cuda_int8
    python quantization/compare_dumps.py --a dumps_scalar_fp32 --b dumps_cuda_fp32 \
        --max-abs-diff 1
"""

import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DUMP_RE = re.compile(r"^L(\d+)_cpp\.bin$")


def dump_index(path):
    """Map L00_cpp.bin .. L21_cpp.bin to {layer_index: filename}."""
    if not os.path.isdir(path):
        return None
    found = {}
    for name in os.listdir(path):
        m = DUMP_RE.match(name)
        if m:
            found[int(m.group(1))] = name
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference dump dir")
    ap.add_argument("--b", required=True, help="dump dir under test")
    ap.add_argument("--max-abs-diff", type=int, default=0,
                    help="largest tolerated per-element INT8 difference (default 0)")
    args = ap.parse_args()

    dir_a = args.a if os.path.isabs(args.a) else os.path.join(ROOT, args.a)
    dir_b = args.b if os.path.isabs(args.b) else os.path.join(ROOT, args.b)

    index_a = dump_index(dir_a)
    index_b = dump_index(dir_b)
    for label, path, index in (("--a", dir_a, index_a), ("--b", dir_b, index_b)):
        if index is None:
            print("{} is not a directory: {}".format(label, path))
            return 2
        if not index:
            print("{} holds no L*_cpp.bin dumps: {}".format(label, path))
            return 2

    print("a: {}".format(dir_a))
    print("b: {}".format(dir_b))
    print("tolerance: |diff| <= {}".format(args.max_abs_diff))
    print()

    only_a = sorted(set(index_a) - set(index_b))
    only_b = sorted(set(index_b) - set(index_a))
    if only_a or only_b:
        print("layer sets differ -- only in a: {}, only in b: {}".format(only_a, only_b))

    print("{:<6} {:>10} {:>10} {:>10} {:>6} {:>9}".format(
        "Layer", "N", "exact", "differing", "max", "bias"))

    failures = []
    shared = sorted(set(index_a) & set(index_b))
    for i in shared:
        a = np.fromfile(os.path.join(dir_a, index_a[i]), dtype=np.int8)
        b = np.fromfile(os.path.join(dir_b, index_b[i]), dtype=np.int8)

        if a.size != b.size:
            print("{:<6} {:>10} {:>10} {:>10} {:>6}  SIZE MISMATCH (b={})".format(
                "L{:02d}".format(i), a.size, "-", "-", "-", b.size))
            failures.append(i)
            continue

        # int16 so the subtraction cannot wrap at the int8 boundary.
        signed = a.astype(np.int16) - b.astype(np.int16)
        d = np.abs(signed)
        n_diff = int((d != 0).sum())
        worst = int(d.max()) if d.size else 0
        bad = worst > args.max_abs_diff

        # Mean signed difference separates rounding from a real error. Reduction
        # order that rounds differently scatters symmetrically about zero; a
        # wrong scale, a missing bias or a bad index pulls the mean off it.
        bias = float(signed.mean())

        print("{:<6} {:>10} {:>9.4f}% {:>10} {:>6} {:>+9.5f}  {}".format(
            "L{:02d}".format(i), a.size, 100.0 * float((d == 0).mean()), n_diff, worst,
            bias, "FAIL" if bad else ""))
        if bad:
            failures.append(i)

    print()
    if failures or only_a or only_b:
        if failures:
            print("FAIL: {} layer(s) outside tolerance: {}".format(
                len(failures), ", ".join("L{:02d}".format(i) for i in failures)))
        if only_a or only_b:
            print("FAIL: dump directories do not cover the same layers")
        return 1

    print("PASS: {} layers within |diff| <= {}".format(len(shared), args.max_abs_diff))
    return 0


if __name__ == "__main__":
    sys.exit(main())
