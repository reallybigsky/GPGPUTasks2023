#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

#define TILE_SIZE 16
#define THREAD_WORK 4

__kernel void matrix_multiplication_naive(__global const float* a,
                                          __global const float* b,
                                          __global float* c,
                                          const unsigned int M,
                                          const unsigned int K,
                                          const unsigned int N)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= N || j >= M) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }

    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_local_mem(__global const float* a,
                                              __global const float* b,
                                              __global float* c,
                                              const unsigned int M,
                                              const unsigned int K,
                                              const unsigned int N)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= N || j >= M) {
        return;
    }

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        tileA[local_j][local_i] = a[j * K + tileK + local_i];
        tileB[local_j][local_i] = b[tileK * N + local_j * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_more_work_per_thr(__global const float* a,
                                                      __global const float* b,
                                                      __global float* c,
                                                      const unsigned int M,
                                                      const unsigned int K,
                                                      const unsigned int N)
{
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    const int i = get_group_id(0) * TILE_SIZE + local_i;
    const int j = get_group_id(1) * TILE_SIZE + local_j;

    if (i >= N || j >= M) {
        return;
    }

    const int RTS = TILE_SIZE / THREAD_WORK;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK];
    for (int w = 0; w < THREAD_WORK; ++w)
        sum[w] = 0.0f;

    for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        for (int w = 0; w < THREAD_WORK; ++w) {
            int tmp = w * RTS;
            tileA[local_j + w * RTS][local_i] = a[(j + tmp) * K + tileK + local_i];
            tileB[local_j + w * RTS][local_i] = b[(tileK + local_j + tmp) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp = tileB[k][local_i];
            for (int w = 0; w < THREAD_WORK; ++w) {
                sum[w] += tileA[local_j + w * RTS][k] * tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < THREAD_WORK; ++w) {
        c[(j + w * RTS) * N + i] = sum[w];
    }
}
