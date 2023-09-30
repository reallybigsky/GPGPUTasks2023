#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* from,
                               __global float* to,
                               const unsigned int M,
                               const unsigned int K)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int gr_i = get_group_id(0);
    const unsigned int gr_j = get_group_id(1);

    const unsigned int ls_i = get_local_size(0);
    const unsigned int ls_j = get_local_size(1);

    __local float tile[TILE_SIZE][TILE_SIZE * 2];

    if (i < M && j < K) {
        tile[local_j][local_i + local_j] = from[j * K + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < M && j < K) {
        to[M * (ls_i * gr_i + local_j) + (ls_j * gr_j + local_i)] = tile[local_i][local_i + local_j];
    }
}