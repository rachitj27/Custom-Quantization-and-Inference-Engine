// Host check of the INT8 GEMM formulation against direct convolution.
// No GPU needed; integer throughout, so exact.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

struct Case { int ic, oc, kh, kw, in_h, in_w, stride, pad, zp; };

static int rnd(unsigned& s, int lo, int hi) {
    s = s * 1664525u + 1013904223u;
    return lo + (int)((s >> 8) % (unsigned)(hi - lo + 1));
}

static bool run(const Case& t, unsigned seed) {
    unsigned s = seed;
    const int out_h = (t.in_h + 2 * t.pad - t.kh) / t.stride + 1;
    const int out_w = (t.in_w + 2 * t.pad - t.kw) / t.stride + 1;
    const int m_real = out_h * out_w;
    const int icp = round_up(t.ic, 16);
    const int kp = t.kh * t.kw * icp;

    // input, CHW int8
    std::vector<int8_t> in((size_t)t.ic * t.in_h * t.in_w);
    for (auto& v : in) v = (int8_t)rnd(s, -128, 127);

    // weights OIHW, then packed as prepare_kernel does
    std::vector<int8_t> w((size_t)t.oc * t.ic * t.kh * t.kw);
    for (auto& v : w) v = (int8_t)rnd(s, -127, 127);

    std::vector<int8_t> whwc((size_t)t.oc * t.kh * t.kw * icp, 0);
    std::vector<int32_t> wsum((size_t)t.oc, 0);
    for (int o = 0; o < t.oc; o++) {
        int32_t sum = 0;
        for (int r = 0; r < t.kh; r++) {
            for (int c = 0; c < t.kw; c++) {
                const size_t dst = (((size_t)o * t.kh + r) * t.kw + c) * icp;
                for (int i = 0; i < t.ic; i++) {
                    const int8_t v = w[(((size_t)o * t.ic + i) * t.kh + r) * t.kw + c];
                    whwc[dst + i] = v;
                    sum += v;
                }
            }
        }
        wsum[o] = sum;
    }

    // A: [m_real][kp], holes = zero_point
    std::vector<int8_t> A((size_t)m_real * kp, (int8_t)t.zp);
    for (int mm = 0; mm < m_real; mm++) {
        for (int k = 0; k < kp; k++) {
            const int i = k % icp;
            const int tap = k / icp;
            const int c = tap % t.kw;
            const int r = tap / t.kw;
            int8_t v = (int8_t)t.zp;
            if (i < t.ic && r < t.kh) {
                const int oh = mm / out_w;
                const int ow = mm - oh * out_w;
                const int ih = oh * t.stride + r - t.pad;
                const int iw = ow * t.stride + c - t.pad;
                if (ih >= 0 && ih < t.in_h && iw >= 0 && iw < t.in_w) {
                    v = in[((size_t)i * t.in_h + ih) * t.in_w + iw];
                }
            }
            A[(size_t)mm * kp + k] = v;
        }
    }

    // B: K-major, grouped in fours
    const int np = t.oc;
    std::vector<int8_t> B((size_t)(kp / 4) * np * 4, 0);
    for (int o = 0; o < t.oc; o++) {
        for (int k = 0; k < kp; k++) {
            const int i = k % icp;
            const int tap = k / icp;
            const int c = tap % t.kw;
            const int r = tap / t.kw;
            if (i >= t.ic) continue;
            const size_t src = (((size_t)o * t.kh + r) * t.kw + c) * icp + i;
            B[((size_t)(k / 4) * np + o) * 4 + (k % 4)] = whwc[src];
        }
    }

    // acc_raw as dp4a accumulates it, then the correction
    int bad = 0;
    for (int mm = 0; mm < m_real; mm++) {
        for (int o = 0; o < t.oc; o++) {
            int32_t raw = 0;
            for (int k4 = 0; k4 < kp / 4; k4++) {
                for (int j = 0; j < 4; j++) {
                    const int k = k4 * 4 + j;
                    raw += (int32_t)A[(size_t)mm * kp + k] *
                           (int32_t)B[((size_t)k4 * np + o) * 4 + j];
                }
            }
            const int32_t got = raw - t.zp * wsum[o];

            // direct convolution, as conv_scalar_int8 computes it
            int32_t want = 0;
            for (int i = 0; i < t.ic; i++) {
                for (int r = 0; r < t.kh; r++) {
                    const int oh = mm / out_w;
                    const int ih = oh * t.stride + r - t.pad;
                    if (ih < 0 || ih >= t.in_h) continue;
                    for (int c = 0; c < t.kw; c++) {
                        const int ow = mm % out_w;
                        const int iw = ow * t.stride + c - t.pad;
                        if (iw < 0 || iw >= t.in_w) continue;
                        const int q = in[((size_t)i * t.in_h + ih) * t.in_w + iw];
                        const int wv = w[(((size_t)o * t.ic + i) * t.kh + r) * t.kw + c];
                        want += (int32_t)(q - t.zp) * (int32_t)wv;
                    }
                }
            }

            if (got != want) {
                if (bad < 3) {
                    std::printf("    m=%d oc=%d got=%d want=%d\n", mm, o, got, want);
                }
                bad++;
            }
        }
    }

    const long long worst = (long long)kp * 255 * 127;
    std::printf("  ic=%-3d oc=%-3d %dx%d s%d p%d  %dx%d -> %dx%d  K=%-5d %s"
                "  (|acc| bound %lld, %.1f%% of int32)\n",
                t.ic, t.oc, t.kh, t.kw, t.stride, t.pad, t.in_h, t.in_w, out_h,
                out_w, kp, bad ? "MISMATCH" : "exact", worst,
                100.0 * (double)worst / 2147483647.0);
    return bad == 0;
}

int main() {
    // the shapes the model uses, plus a thin layer 0 and odd zero points
    const Case cases[] = {
        {3, 16, 3, 3, 32, 32, 2, 1, -128},
        {16, 32, 3, 3, 16, 16, 2, 1, -128},
        {32, 64, 3, 3, 10, 10, 1, 1, -116},
        {64, 64, 1, 1, 8, 8, 1, 0, -125},
        {48, 32, 1, 1, 7, 7, 1, 0, 0},
        {96, 64, 3, 3, 5, 5, 1, 1, 42},
        {128, 2, 1, 1, 5, 5, 1, 0, -100},
        {3, 8, 3, 3, 4, 4, 1, 1, 127},
    };
    bool ok = true;
    std::printf("INT8 GEMM formulation vs direct convolution\n");
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (!run(cases[i], 12345u + i * 7919u)) ok = false;
    }
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
