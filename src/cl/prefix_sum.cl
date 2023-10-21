#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

__kernel void prefix_sum(__global const unsigned int *as,
                         __global unsigned int *bs,
                         const unsigned int N,
                         const unsigned int offset)
{
    const int gid = get_global_id(0);
    if (gid >= N)
        return;

    bs[gid] = gid >= offset ? as[gid] + as[gid - offset] : as[gid];
}