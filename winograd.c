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

      c_00_01_02_03 = _mm_fmadd_ps(a_00_01_02_03, b_00_01_02_03, c_00_01_02_03);
      c_10_11_12_13 = _mm_fmadd_ps(a_10_11_12_13, b_00_01_02_03, c_10_11_12_13);
      c_20_21_22_23 = _mm_fmadd_ps(a_20_21_22_23, b_00_01_02_03, c_20_21_22_23);
      c_30_31_32_33 = _mm_fmadd_ps(a_30_31_32_33, b_00_01_02_03, c_30_31_32_33);
    }
    _mm_store_ps(out + i * N + j, _mm_add_ps(_mm_load_ps(out + i * N + j), c_00_01_02_03));
    _mm_store_ps(out + (i + 1) * N + j, _mm_add_ps(_mm_load_ps(out + (i + 1) * N + j), c_10_11_12_13));
    _mm_store_ps(out + (i + 2) * N + j, _mm_add_ps(_mm_load_ps(out + (i + 2) * N + j), c_20_21_22_23));
    _mm_store_ps(out + (i + 3) * N + j, _mm_add_ps(_mm_load_ps(out + (i + 3) * N + j), c_30_31_32_33));
}

void avx_dot_8x8(const int K, float *A, int N, float *B, float *out, int i, int j) {
  __m256
    c_00_01_02_03_04_05_06_07,
    c_10_11_12_13_14_15_16_17,
    c_20_21_22_23_24_25_26_27,
    c_30_31_32_33_34_35_36_37,
    c_40_41_42_43_44_45_46_47,
    c_50_51_52_53_54_55_56_57,
    c_60_61_62_63_64_65_66_67,
    c_70_71_72_73_74_75_76_77,

    b_00_01_02_03_04_05_06_07,

    a_00_01_02_03_04_05_06_07,
    a_10_11_12_13_14_15_16_17,
    a_20_21_22_23_24_25_26_27,
    a_30_31_32_33_34_35_36_37,
    a_40_41_42_43_44_45_46_47,
    a_50_51_52_53_54_55_56_57,
    a_60_61_62_63_64_65_66_67,
    a_70_71_72_73_74_75_76_77;

  c_00_01_02_03_04_05_06_07 = _mm256_setzero_ps();
  c_10_11_12_13_14_15_16_17 = _mm256_setzero_ps();
  c_20_21_22_23_24_25_26_27 = _mm256_setzero_ps();
  c_30_31_32_33_34_35_36_37 = _mm256_setzero_ps();
  c_40_41_42_43_44_45_46_47 = _mm256_setzero_ps();
  c_50_51_52_53_54_55_56_57 = _mm256_setzero_ps();
  c_60_61_62_63_64_65_66_67 = _mm256_setzero_ps();
  c_70_71_72_73_74_75_76_77 = _mm256_setzero_ps();

    for(int k = 0; k<K; ++k) {
      b_00_01_02_03_04_05_06_07 = _mm256_load_ps(B + k * N + j);

      a_00_01_02_03_04_05_06_07 = _mm256_broadcast_ss(A + i * K + k);
      a_10_11_12_13_14_15_16_17 = _mm256_broadcast_ss(A + (i + 1) * K + k); 
      a_20_21_22_23_24_25_26_27 = _mm256_broadcast_ss(A + (i + 2) * K + k); 
      a_30_31_32_33_34_35_36_37 = _mm256_broadcast_ss(A + (i + 3) * K + k);
      a_40_41_42_43_44_45_46_47 = _mm256_broadcast_ss(A + (i + 4) * K + k);
      a_50_51_52_53_54_55_56_57 = _mm256_broadcast_ss(A + (i + 5) * K + k);
      a_60_61_62_63_64_65_66_67 = _mm256_broadcast_ss(A + (i + 6) * K + k);
      a_70_71_72_73_74_75_76_77 = _mm256_broadcast_ss(A + (i + 7) * K + k); 

      c_00_01_02_03_04_05_06_07 = _mm256_fmadd_ps(a_00_01_02_03_04_05_06_07, b_00_01_02_03_04_05_06_07, c_00_01_02_03_04_05_06_07);
      c_10_11_12_13_14_15_16_17 = _mm256_fmadd_ps(a_10_11_12_13_14_15_16_17, b_00_01_02_03_04_05_06_07, c_10_11_12_13_14_15_16_17);
      c_20_21_22_23_24_25_26_27 = _mm256_fmadd_ps(a_20_21_22_23_24_25_26_27, b_00_01_02_03_04_05_06_07, c_20_21_22_23_24_25_26_27);
      c_30_31_32_33_34_35_36_37 = _mm256_fmadd_ps(a_30_31_32_33_34_35_36_37, b_00_01_02_03_04_05_06_07, c_30_31_32_33_34_35_36_37);
      c_40_41_42_43_44_45_46_47 = _mm256_fmadd_ps(a_40_41_42_43_44_45_46_47, b_00_01_02_03_04_05_06_07, c_40_41_42_43_44_45_46_47);
      c_50_51_52_53_54_55_56_57 = _mm256_fmadd_ps(a_50_51_52_53_54_55_56_57, b_00_01_02_03_04_05_06_07, c_50_51_52_53_54_55_56_57);
      c_60_61_62_63_64_65_66_67 = _mm256_fmadd_ps(a_60_61_62_63_64_65_66_67, b_00_01_02_03_04_05_06_07, c_60_61_62_63_64_65_66_67);
      c_70_71_72_73_74_75_76_77 = _mm256_fmadd_ps(a_70_71_72_73_74_75_76_77, b_00_01_02_03_04_05_06_07, c_70_71_72_73_74_75_76_77);
    }
    _mm256_store_ps(out + i * N + j, _mm256_add_ps(_mm256_load_ps(out + i * N + j), c_00_01_02_03_04_05_06_07));
    _mm256_store_ps(out + (i + 1) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 1) * N + j), c_10_11_12_13_14_15_16_17));
    _mm256_store_ps(out + (i + 2) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 2) * N + j), c_20_21_22_23_24_25_26_27));
    _mm256_store_ps(out + (i + 3) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 3) * N + j),   c_30_31_32_33_34_35_36_37));
    _mm256_store_ps(out + (i + 4) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 4) * N + j),   c_40_41_42_43_44_45_46_47));
    _mm256_store_ps(out + (i + 5) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 5) * N + j),   c_50_51_52_53_54_55_56_57));
    _mm256_store_ps(out + (i + 6) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 6) * N + j),   c_60_61_62_63_64_65_66_67));
    _mm256_store_ps(out + (i + 7) * N + j, _mm256_add_ps(_mm256_load_ps(out + (i + 7) * N + j),   c_70_71_72_73_74_75_76_77));
}

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function
// For rookies: this sgemm is the worst sgemm I've ever written throughout my
// career.
//      If you don't know where to start, optimize this function as a good
//      starting point.
void sgemm(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {

  memset(out,0,M*N*sizeof(float));
  if(M % 4!=0 || N % 4 !=0) {
    for (int i = 0; i < M; i += 1)
      for (int j = 0; j < N; j += 1) {
        dot1x1(K, A, N, B, out, i, j);      
      }
    return;
  }
  for (int i = 0; i < M; i += 4)
    for (int j = 0; j < N; j += 4){
      avx_dot_4x4(K, A, N, B, out, i, j);
    }
}

void sgemm_parallel(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
  memset(out,0,M*N*sizeof(float));
  #pragma omp parallel for num_threads(24)
    for (int j = 0; j < N; j += 8)
      for (int i = 0; i < M; i += 8)
        avx_dot_8x8(K, A, N, B, out, i, j);
    
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
        for (int nu = 0; nu < 4; ++nu){
          int tmp = xi * 4 + nu;
          U[(tmp * K + k) * C + c] = u[tmp];
        }
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
            for (int nu = 0; nu < 4; ++nu){
              int tmp = xi * 4 + nu;
              V[(tmp * C + c) * P + b] = v[tmp];
            }
        }
      }
    }

  // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
  for (int xi = 0; xi < 4; ++xi) {
    for (int nu = 0; nu < 4; ++nu) {
      int tmp = xi * 4 + nu;
      float *M_ptr = M + tmp * K * P;
      float *U_ptr = U + tmp * K * C;
      float *V_ptr = V + tmp * C * P;
      sgemm_parallel(U_ptr, V_ptr, M_ptr, K, C, P);
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
              int tmp = xi * 4 + nu;
              mm[tmp] = M[(tmp * K + k) * P + b];
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