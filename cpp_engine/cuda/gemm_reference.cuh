#ifndef GEMM_REFERENCE_CUH
#define GEMM_REFERENCE_CUH

// Launches the verbatim vendored kernel. Selftest only.
void sgemm_reference_launch(int M, int N, int K, float alpha, const float* dA,
                            const float* dB, float beta, float* dC);

#endif  // GEMM_REFERENCE_CUH
