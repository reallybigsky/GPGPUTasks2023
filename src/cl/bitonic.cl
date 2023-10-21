#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

__kernel void bitonic(__global float *as,
                      const unsigned N,
                      const unsigned int K,
                      const unsigned int K_curr)
{
    const int gid = get_global_id(0);

    if (gid * 2 >= N)
        return;

    const int idx = (gid / K_curr) * K_curr * 2 + (gid % K_curr);
    const int pr = (gid / K) % 2 ? -1 : 1;
    float lhs = as[idx];
    float rhs = as[idx + K_curr];
    if (pr * lhs > pr * rhs) {
        as[idx] = rhs;
        as[idx + K_curr] = lhs;
    }
}
