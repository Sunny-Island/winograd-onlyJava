#include <assert.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>



const float G[4][3] = {
    {1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, {0.0, 0.5, -0.5, 0.0}, {0.0, 0.5, 0.5, 1.0}};
const float B[4][4] = {
    {1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
const float B_T[4][4] = {
    {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
const float A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
const float A_T[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

void dot1x1(const int K, float *A, int N,float *B, float *out, int i, int j) {
  float p = *(out + i * N + j); 
  for(int k = 0; k < K; k++) {
    p += A[i * K + k] * B[k * N + j];
  }
  *(out + i * N + j) = p;
}

void avx_dot_4x4(const int K, float *A, int N, float *B, float *out, int i, int j) {
  __m128 
    c_00_01_02_03,
    c_10_11_12_13,
    c_20_21_22_23,
    c_30_31_32_33,

    b_00_01_02_03,

    a_00_01_02_03,
    a_10_11_12_13,
    a_20_21_22_23,
    a_30_31_32_33;

    c_00_01_02_03 = _mm_setzero_ps();
    c_10_11_12_13 = _mm_setzero_ps();
    c_20_21_22_23 = _mm_setzero_ps();
    c_30_31_32_33 = _mm_setzero_ps();

    for(int k = 0; k<K; ++k) {
      b_00_01_02_03 = _mm_load_ps(B + k * N + j);
      a_00_01_02_03 = _mm_broadcast_ss(A + i * K + k);
      a_10_11_12_13 = _mm_broadcast_ss(A + (i + 1) * K + k); 
      a_20_21_22_23 = _mm_broadcast_ss(A + (i + 2) * K + k); 
      a_30_31_32_33 = _mm_broadcast_ss(A + (i + 3) * K + k); 

      c_00_01_02_03 += _mm_mul_ps(a_00_01_02_03, b_00_01_02_03);
      c_10_11_12_13 += _mm_mul_ps(a_10_11_12_13, b_00_01_02_03);
      c_20_21_22_23 += _mm_mul_ps(a_20_21_22_23, b_00_01_02_03);
      c_30_31_32_33 += _mm_mul_ps(a_30_31_32_33, b_00_01_02_03);
    }
    _mm_store_ps(out + i * N + j, _mm_add_ps(_mm_load_ps(out + i * N + j), c_00_01_02_03));
    _mm_store_ps(out + (i + 1) * N + j, _mm_add_ps(_mm_load_ps(out + (i + 1) * N + j), c_10_11_12_13));
    _mm_store_ps(out + (i + 2) * N + j, _mm_add_ps(_mm_load_ps(out + (i + 2) * N + j), c_20_21_22_23));
    _mm_store_ps(out + (i + 3) * N + j, _mm_add_ps(_mm_load_ps(out + (i + 3) * N + j), c_30_31_32_33));

}

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function
// For rookies: this sgemm is the worst sgemm I've ever written throughout my
// career.
//      If you don't know where to start, optimize this function as a good
//      starting point.
void sgemm(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  for (int i = 0; i < M * N; ++i) {
    out[i] = 0.0f;
  }
  
  //#pragma omp parallel for
  for (int i = 0; i < M; i += 4)
    for (int j = 0; j < N; j += 4){
      // dot1x1(K, A, N, B, out, i, j);
      // dot1x1(K, A, N, B, out, i, j + 1);
      // dot1x1(K, A, N, B, out, i, j + 2);
      // dot1x1(K, A, N, B, out, i, j + 3);
      avx_dot_4x4(K, A, N, B, out, i, j);
    }
}

// User API for winograd F(2,3)
// image: [batch * C * inHeight * inWidth]
// filter: [K * C * 3 * 3]
// result: [batch * K * outHeight * outWidth]
void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M) {
  // m = 2; r = 3; alpha = 4
  const int outHeight = inHeight - 2;
  const int outWidth = inWidth - 2;
  const int sizeI = inHeight * inWidth;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  const int P = outHeight / 2 * outWidth / 2 * N;

  float tmp_u[12];  // 4 * 3
  float u[16];      // 4 * 4;
  // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float *filters_ptr = filter + (k * C + c) * sizeF;
      sgemm(&G[0][0], filters_ptr, tmp_u, 4, 3, 3);
      sgemm(tmp_u, &G_T[0][0], u, 4, 3, 4);
      for (int xi = 0; xi < 4; ++xi)
        for (int nu = 0; nu < 4; ++nu)
          U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu];
    }
  }
  // V[:, :, c, p] = B_T * image[c, b, :, :] * B
  float tmp_v[16];
  float d[16];  // d: [4 * 4];
  float v[16];  // v: [4 * 4];

  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < outHeight / 2; ++y) {
        for (int x = 0; x < outWidth / 2; ++x) {
          // Generate d_cb
          for (int iy = 0; iy < 4; ++iy)
            for (int ix = 0; ix < 4; ++ix)
              d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                     (y * 2 + iy) * inWidth + (x * 2 + ix)];
          sgemm(&B_T[0][0], d, tmp_v, 4, 4, 4);
          sgemm(tmp_v, &B[0][0], v, 4, 4, 4);
          int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
          for (int xi = 0; xi < 4; ++xi)
            for (int nu = 0; nu < 4; ++nu)
              V[((xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu];
        }
      }
    }

  // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
  for (int xi = 0; xi < 4; ++xi) {
    for (int nu = 0; nu < 4; ++nu) {
      float *M_ptr = M + (xi * 4 + nu) * K * P;
      float *U_ptr = U + (xi * 4 + nu) * K * C;
      float *V_ptr = V + (xi * 4 + nu) * C * P;
      sgemm(U_ptr, V_ptr, M_ptr, K, C, P);
    }
  }

  // Y = A_T * m * A
  float mm[16];       // 4 * 4
  float temp_out[4];  // 2 * 2
  float tmp_m[8];     // 2 * 4
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      for (int y = 0; y < outHeight / 2; ++y) {
        for (int x = 0; x < outWidth / 2; ++x) {
          int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
          for (int xi = 0; xi < 4; ++xi) {
            for (int nu = 0; nu < 4; ++nu) {
              mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
            }
          }
          sgemm(&A_T[0][0], mm, tmp_m, 2, 4, 4);
          sgemm(tmp_m, &A[0][0], temp_out, 2, 4, 2);
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              out[((n * K + k) * outHeight + y * 2 + i) * outWidth + x * 2 +
                  j] = temp_out[i * 2 + j];
        }
      }
    }
}