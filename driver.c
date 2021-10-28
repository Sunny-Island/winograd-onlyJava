#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <mkl.h>
#include <omp.h>

#include "config.h"

float* t_filter;    
float* t_image;    
float* c_out;  

#if 1
const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
long ISTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 
long OSTRIDE = (MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
#endif

#ifdef __DEBUG
#define inline
#endif
#define LOOP_NUM 100

inline double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1.e-6;
}

inline void no4k_aligned(long *num) {
  long flag = *num;
  if (flag % 4096 == 0) (*num) += 128;
}

// setup scratch memory used in the algorithm 
void winconv_init_lib()
{
    int ret;

    t_filter = (float *)mkl_malloc(36*FSTRIDE*sizeof(float), 64);
    //std::cout << 36 * FSTRIDE * sizeof(float) << std::endl;
    assert(t_filter != NULL); 
    t_image = (float *)mkl_malloc(36*ISTRIDE*sizeof(float), 64);
    //std::cout << 36 * ISTRIDE * sizeof(float) << std::endl;
    assert(t_image != NULL); 
    c_out = (float *)mkl_malloc(36*OSTRIDE*sizeof(float), 64);
    //std::cout << 36 * OSTRIDE * sizeof(float) << std::endl;
    assert(c_out != NULL); 
}

// free up 
void winconv_free_lib()
{
    mkl_free(t_filter);    
    mkl_free(t_image);    
    mkl_free(c_out);    
}


void winograd_init(const int layer_num, const int Batch[], const int C[],
                   const int H[], const int W[], const int K[]) {
  int tmp;
  /* Compute max stride for input, filter and output. */
  long istride, fstride, ostride;
  istride = fstride = ostride = 0;

  for (int i = 0; i < layer_num; i++) {
    tmp = Batch[i] * (H[i] + 1) / 4 * (W[i] + 1) / 4 * C[i];
    if (tmp > istride) istride = tmp;

    tmp = C[i] * K[i];
    if (tmp > fstride) fstride = tmp;

    tmp = Batch[i] * (H[i] + 1) / 4 * (W[i] + 1) / 4 * K[i];
    if (tmp > ostride) ostride = tmp;
  }

  no4k_aligned(&istride);
  no4k_aligned(&fstride);
  no4k_aligned(&ostride);
  ISTRIDE = istride;
  FSTRIDE = fstride;
  OSTRIDE = ostride;
}

// API. The implementation is in winograd.cpp
void winconv_2x3(float *__restrict__ image, const int irows, const int icols,
                 const int C, float *__restrict__ filter, const int K,
                 const int batch, float *__restrict__ out);

int naive_conv(float *in, float *kn, float *out, const int N, const int C,
               const int H, const int W, const int K) {
  int inpos, knpos, outpos;

  int dimIn[4] = {N, C, H, W};
  int dimKn[4] = {K, C, 3, 3};
  int dimOut[4] = {N, K, H - 2, W - 2};

  int ingap[3] = {dimIn[1] * dimIn[2] * dimIn[3], dimIn[2] * dimIn[3],
                  dimIn[3]};
  int kngap[3] = {dimKn[1] * dimKn[2] * dimKn[3], dimKn[2] * dimKn[3],
                  dimKn[3]};
  int outgap[3] = {dimOut[1] * dimOut[2] * dimOut[3], dimOut[2] * dimOut[3],
                   dimOut[3]};

#pragma omp parallel for private(inpos, knpos, outpos)
  for (int inn = 0; inn < dimIn[0]; inn++)
    for (int knn = 0; knn < dimKn[0]; knn++)
      for (int inc = 0; inc < dimIn[1]; inc++) {
        for (int outh = 0; outh < dimOut[2]; outh++)
          for (int outw = 0; outw < dimOut[3]; outw++) {
            outpos =
                inn * outgap[0] + knn * outgap[1] + outh * outgap[2] + outw;
            for (int knh = 0; knh < dimKn[2]; knh++)
              for (int knw = 0; knw < dimKn[3]; knw++) {
                inpos = inn * ingap[0] + inc * ingap[1] +
                        (outh + knh) * ingap[2] + (outw + knw);
                // knpos = knn*kngap[0] + inc*kngap[1] + 8 - (knh*kngap[2] +
                // knw);
                knpos = knn * kngap[0] + inc * kngap[1] + knh * kngap[2] + knw;
                out[outpos] += in[inpos] * kn[knpos];
              }
          }
      }

  return 0;
}

void winograd_conv(const int layer_idx, const int validation_mode,
                   const int irows, const int icols, const int C, const int K,
                   const int batch, long *total_flops, double *total_time) {
  long i, j, n;
  const int outHeight = irows - 2;
  const int outWidth = icols - 2;
  const int sizeI = irows * icols;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  const int P = batch * outHeight / 2 * outWidth / 2;

  int ret;

  float *image, *filter, *out;
  image = (float *)mkl_malloc(64, batch * C * sizeI * sizeof(float));
  assert(image != NULL);
  filter = (float *)mkl_malloc(64, K * C * sizeF * sizeof(float));
  assert(filter != NULL);
  out = (float *)mkl_malloc(64, batch * K * sizeO * sizeof(float));
  assert(out != NULL);

#pragma omp parallel for private(i)
  for (long i = 0; i < batch * C * sizeI; i++) image[i] = (float)(i % 10 + 1);
    // image[i] = rand()%10;
#pragma omp parallel for private(i)
  for (long i = 0; i < K * C * sizeF; i++) filter[i] = (float)(i / sizeF + 1);
  // filter[i] = rand()%10;

  // Warm up
  printf("%d",2);
  winconv_2x3(image, irows, icols, C, filter, K, batch, out);
  printf("%d",3);
  if (validation_mode) {  // Verify mode. Check the result
    float *out_ref = (float *)malloc(batch * K * sizeO * sizeof(float));
    memset(out_ref, 0, batch * K * sizeO * sizeof(float));

    naive_conv(image, filter, out_ref, batch, C, irows, icols, K);
    printf(
        "Layer %-2d: (Channel Height Weight Filter Batch) = "
        "(%-3d %-3d %-3d %-3d %-3d) : ",
        layer_idx, C, irows, icols, K, batch);
    long n;
    for (n = 0; n < batch * sizeO * K; n++)
      if (fabs((out[n] - out_ref[n]) / out_ref[n]) > 1e-4) {
        printf(
            "Validation Failed ! winogradConv[%d] = %f || directConv[%d] = %f "
            "\n",
            n, out[n], n, out_ref[n]);
        break;
      }
    if (n == batch * sizeO * K) printf("Validation Passed !\n");
    free(out_ref);
  } else {  // Benchmark mode
    printf("%d",4);
    double start_time = timestamp();
    for (int i = 0; i < LOOP_NUM; i++) {
      winconv_2x3(image, irows, icols, C, filter, K, batch, out);
    }
    double end_time = timestamp();

    double elapse_time_all = end_time - start_time;

    double elapse_time = elapse_time_all / LOOP_NUM;
    *total_time += elapse_time;

    long nflops = (long)batch * K * C * (irows - 2) * (icols - 2) * 3 * 3 * 2;
    double gflops = (double)nflops * 1.0e-9 / elapse_time;
    *total_flops += nflops;
    printf("Layer %-2d:  Elapse time %lf ms. ( %7.2lf GFlops) \n", layer_idx,
           elapse_time * 1000, gflops);
  }
  mkl_free(image);
  mkl_free(filter);
  mkl_free(out);
}

int main(int argc, char *argv[]) {
  printf("%d",11);
  if (argc < 2) {
    printf("Usage: %s layer.conf [validation=0/1] \n", argv[0]);
    // printf("Please provided layer configs. Aborting\n");
    exit(-1);
  }
  printf("%d",12);
  FILE *input = fopen(argv[1], "r");
  if (!input) {
    printf("File open failed. Aborting...\n");
    exit(-1);
  }
  printf("%d",13);
  int validation_mode = 0;  // 0 : benchmark mode, 1: validation mode
  if (argc > 2) validation_mode = atoi(argv[2]);

  int layer_num, batch;
  fscanf(input, "%d", &layer_num);
  if (layer_num <= 0) {
    printf("Invalid layer num %d. Aborting\n", layer_num);
    fclose(input);
    exit(1);
  }
  printf("%d",14);
  int *C_arr = (int *)malloc(sizeof(int) * layer_num);      // Channel
  int *H_arr = (int *)malloc(sizeof(int) * layer_num);      // Image Height
  int *W_arr = (int *)malloc(sizeof(int) * layer_num);      // Image Width
  int *K_arr = (int *)malloc(sizeof(int) * layer_num);      // Filters
  int *Batch_arr = (int *)malloc(sizeof(int) * layer_num);  // Batch
  printf("%d",15);

  for (int l = 0; l < layer_num; ++l) {
    fscanf(input, "%d%d%d%d%d", &C_arr[l], &H_arr[l], &W_arr[l], &K_arr[l],
           &Batch_arr[l]);
  }
  fclose(input);

  printf("%d",16);

  // srand(time(NULL));
  srand(20210930);

  double total_time;
  long total_flops;

  printf("%d",0);

  winconv_init_lib();

  winograd_init(layer_num, Batch_arr, C_arr, H_arr, W_arr, K_arr);

  printf("%d",1);

  for (int l = 0; l < layer_num; l++) {
    winograd_conv(l, validation_mode, H_arr[l], W_arr[l], C_arr[l], K_arr[l],
                  Batch_arr[l], &total_flops, &total_time);
  }

  if (!validation_mode)
    printf("Total elapse time: %lf. ( %7.2lf GFlops) \n", total_time,
           (double)total_flops * 1.0e-9 / total_time);

  winconv_free_lib();

  if (C_arr) free(C_arr);
  if (H_arr) free(H_arr);
  if (W_arr) free(W_arr);
  if (K_arr) free(K_arr);
  if (Batch_arr) free(Batch_arr);
  return 0;
}