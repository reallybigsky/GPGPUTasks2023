#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

__kernel void merge(__global const float* A,
                    __global float* B,
                    const unsigned int N,
                    const unsigned int M)
{
    const int gid = get_global_id(0);
    const float curr = gid < N ? A[gid] : FLT_MAX;
    float other;

    const int group_id = gid / M;
    const int left_bound = group_id * M;
    const int right_bound = left_bound + M;
    const int local_id = gid - left_bound;

    bool is_right = group_id % 2;

    int L = is_right ? left_bound - M : right_bound;
    int R = is_right ? left_bound : L + M;
    int med = 0;

    while (L < R) {
        med = (L + R) / 2;
        other = A[med];
        if ((!is_right && curr <= other) || curr < other) {
            R = med;
        } else {
            L = med + 1;
        }
    }

    const int index = local_id + R - !is_right * M;
    if (index < N)
        B[index] = curr;
}
