// matrix_multiply.cpp
#include <vector>
extern "C" {
    void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                C[i * K + j] = 0;
                for (int k = 0; k < N; ++k) {
                    C[i * K + j] += A[i * N + k] * B[k * K + j];
                }
            }
        }
    }
}
