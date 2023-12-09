#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


#define GRAVITATIONAL_FORCE 0.0001

#define morton_t unsigned long

#define NBITS_PER_DIM 16
#define NBITS (NBITS_PER_DIM /*x dimension*/ + NBITS_PER_DIM /*y dimension*/ + 32 /*index augmentation*/)

int LBVHSize(int N) {
    return N + N-1;
}

morton_t getBits(morton_t morton_code, int bit_index, int prefix_size)
{
    morton_t one = 1;
    return (morton_code >> bit_index) & ((one << prefix_size) - one);
}

int getBit(morton_t morton_code, int bit_index)
{
    return (morton_code >> bit_index) & 1;
}

int getIndex(morton_t morton_code)
{
    morton_t mask = 1;
    mask = (mask << 32) - 1;
    return morton_code & mask;
}

int spreadBits(int word){
    word = (word ^ (word << 8 )) & 0x00ff00ff;
    word = (word ^ (word << 4 )) & 0x0f0f0f0f;
    word = (word ^ (word << 2 )) & 0x33333333;
    word = (word ^ (word << 1 )) & 0x55555555;
    return word;
}

struct __attribute__ ((packed)) BBox {

    int minx, maxx;
    int miny, maxy;

};

void clear(__global struct BBox *self)
{
    self->minx = INT_MAX;
    self->maxx = INT_MIN;
    self->miny = self->minx;
    self->maxy = self->maxx;
}

bool contains(__global const struct BBox *self, float fx, float fy)
{
    int x = fx + 0.5;
    int y = fy + 0.5;
    return x >= self->minx && x <= self->maxx &&
           y >= self->miny && y <= self->maxy;
}

bool empty(__global const struct BBox *self)
{
    return self->minx > self->maxx;
}

struct __attribute__ ((packed)) Node {

    int child_left, child_right;
    struct BBox bbox;

    // used only for nbody
    float mass;
    float cmsx;
    float cmsy;
};

bool hasLeftChild(__global const struct Node *self)
{
    return self->child_left >= 0;
}

bool hasRightChild(__global const struct Node *self)
{
    return self->child_right >= 0;
}

bool isLeaf(__global const struct Node *self)
{
    return !hasLeftChild(self) && !hasRightChild(self);
}

void growPoint(__global struct BBox *self, float fx, float fy)
{
    self->minx = min(self->minx, (int) (fx + 0.5));
    self->maxx = max(self->maxx, (int) (fx + 0.5));
    self->miny = min(self->miny, (int) (fy + 0.5));
    self->maxy = max(self->maxy, (int) (fy + 0.5));
}

void growBBox(__global struct BBox *self, __global const struct BBox *other)
{
    growPoint(self, other->minx, other->miny);
    growPoint(self, other->maxx, other->maxy);
}

bool equals(__global const struct BBox *lhs, __global const struct BBox *rhs)
{
    return lhs->minx == rhs->minx && lhs->maxx == rhs->maxx && lhs->miny == rhs->miny && lhs->maxy == rhs->maxy;
}

bool equalsPoint(__global const struct BBox *lhs, float fx, float fy)
{
    int x = fx + 0.5;
    int y = fy + 0.5;
    return lhs->minx == x && lhs->maxx == x && lhs->miny == y && lhs->maxy == y;
}

morton_t zOrder(float fx, float fy, int i){
    int x = fx + 0.5;
    int y = fy + 0.5;

    // у нас нет эксепшенов, но можно писать коды ошибок просто в консоль, и следить чтобы вывод был пустой

    if (x < 0 || x >= (1 << NBITS_PER_DIM)) {
        printf("098245490432590890\n");
//        return 0;
    }
    if (y < 0 || y >= (1 << NBITS_PER_DIM)) {
        printf("432764328764237823\n");
//        return 0;
    }

    // TODO
    int xx = spreadBits(x);
    int yy = spreadBits(y);
    morton_t morton_code = yy * 2 + xx;

    // augmentation
    return (morton_code << 32) | i;
}

__kernel void generateMortonCodes(__global const float *pxs, __global const float *pys,
                                  __global morton_t *codes,
                                  int N)
{
    int gid = get_global_id(0);
    if (gid >= N)
        return;

    codes[gid] = zOrder(pxs[gid], pys[gid], gid);
}

bool mergePathPredicate(morton_t val_mid, morton_t val_cur, bool is_right)
{
    return is_right ? val_mid <= val_cur : val_mid < val_cur;
}

void __kernel merge(__global const morton_t *as, __global morton_t *as_sorted, unsigned int n, unsigned int subarray_size)
{
    const int gid = get_global_id(0);
    if (gid >= n)
        return;

    const int subarray_id = gid / subarray_size;
    const int is_right_subarray = subarray_id & 1;

    const int base_cur = (subarray_id) * subarray_size;
    const int base_other = (subarray_id + 1 - 2 * is_right_subarray) * subarray_size;

    const int j = gid - base_cur;
    const morton_t val_cur = as[gid];

    int i0 = -1;
    int i1 = subarray_size;
    while (i1 - i0 > 1) {
        int mid = (i0 + i1) / 2;
        if (base_other + mid < n && mergePathPredicate(as[base_other + mid], val_cur, is_right_subarray)) {
            i0 = mid;
        } else {
            i1 = mid;
        }
    }
    const int i = i1;

    int idx = min(base_cur, base_other) + j + i;
    as_sorted[idx] = val_cur;
}

int findSplit(__global const morton_t *codes, int i_begin, int i_end, int bit_index)
{
    if (getBit(codes[i_begin], bit_index) == getBit(codes[i_end - 1], bit_index)) {
        return -1;
    }

    // TODO бинпоиск для нахождения разбиения области ответственности ноды
    while (i_begin + 1 != i_end) {
        int i_med = i_begin + (i_end - i_begin)/2;
        int bit_med = getBit(codes[i_med], bit_index);
        if (bit_med) {
            i_end = i_med;
        } else {
            i_begin = i_med;
        }
    }
    return i_end;
}

void findRegion(int *i_begin, int *i_end, int *bit_index, __global const morton_t *codes, int N, int i_node)
{
    // TODO
    if (i_node < 1 || i_node > N - 2) {
        printf("842384298293482\n");
    }

    int dir = 0;
    int i_bit = NBITS-1;
    for (; i_bit >= 0; --i_bit) {
        // TODO найти dir и значащий бит
        int lb = getBit(codes[i_node - 1], i_bit);
        int cb = getBit(codes[i_node], i_bit);
        int rb = getBit(codes[i_node + 1], i_bit);

        if (lb == 0 && rb == 1) {
            dir = cb * 2 - 1;
            break;
        }
    }

    if (dir == 0) {
        printf("8923482374983\n");
    }

    int K = NBITS - i_bit;
    morton_t pref0 = getBits(codes[i_node], i_bit, K);

    // TODO бинпоиск зоны ответственности
    int l = i_node;
    int r = dir > 0 ? N : -1;
    while (r - l != dir) {
        int m = l + (r - l)/2;
        morton_t pref1 = getBits(codes[m], i_bit, K);
        if (pref0 == pref1) {
            l = m;
        } else {
            r = m;
        }
    }
    int i_node_end = l;

    *bit_index = i_bit - 1;

    if (dir > 0) {
        *i_begin = i_node;
        *i_end = i_node_end + 1;
    } else {
        *i_begin = i_node_end;
        *i_end = i_node + 1;
    }
}


void initLBVHNode(__global struct Node *nodes, int i_node, __global const morton_t *codes, int N, __global const float *pxs, __global const float *pys, __global const float *mxs)
{
    // TODO
    __global struct Node *node = &nodes[i_node];

    clear(&node->bbox);
    node->mass = 0;
    node->cmsx = 0;
    node->cmsy = 0;

    if (i_node >= N-1) {
        node->child_left = -1;
        node->child_right = -1;
        int i_point = i_node - (N-1);

        int index = getIndex(codes[i_point]);

        float center_mass_x = pxs[index];
        float center_mass_y = pys[index];

        growPoint(&node->bbox, center_mass_x, center_mass_y);
        node->cmsx = center_mass_x;
        node->cmsy = center_mass_y;
        node->mass = mxs[index];

        return;
    }

    int i_begin = 0, i_end = N, bit_index = NBITS-1;

    if (i_node) {
        // TODO
        findRegion(&i_begin, &i_end, &bit_index, codes, N, i_node);
    }

    bool found = false;
    for (int i_bit = bit_index; i_bit >= 0; --i_bit) {
        // TODO
        int split = findSplit(codes, i_begin, i_end, i_bit);
        if (split < 0) continue;

        if (split < 1 || split < i_begin + 1 || i_end < split + 1) {
            printf("043204230042342\n");
        }

        // TODO проинициализировать nodes[i_node].child_left, nodes[i_node].child_right на основе i_begin, i_end, split
        //   не забудьте на N-1 сдвинуть индексы, указывающие на листья
        node->child_left = split - i_begin == 1 ? split - 1 + N - 1 : split - 1;
        node->child_right = i_end - split == 1 ? split + N - 1 : split;

        found = true;
        break;
    }

    if (!found) {
        printf("54356549645\n");
    }
}

__kernel void buildLBVH(__global const float *pxs, __global const float *pys, __global const float *mxs,
                       __global const morton_t *codes, __global struct Node *nodes,
                       int N)
{
    // TODO
    unsigned int gid = get_global_id(0);
    if (gid < LBVHSize(N)) {
        initLBVHNode(nodes, gid, codes, N, pxs, pys, mxs);
    }
}

void initFlag(__global int *flags, int i_node, __global const struct Node *nodes, int level)
{
    flags[i_node] = -1;

    __global const struct Node *node = &nodes[i_node];
    if (isLeaf(node)) {
        printf("9423584385834\n");
//        return;
    }

    if (!empty(&node->bbox)) {
        return;
    }

    __global const struct BBox *left = &nodes[node->child_left].bbox;
    __global const struct BBox *right = &nodes[node->child_right].bbox;

    if (!empty(left) && !empty(right)) {
        flags[i_node] = level;
    }
}

__kernel void initFlags(__global int *flags, __global const struct Node *nodes,
                       int N, int level)
{
    int gid = get_global_id(0);

    if (gid == N-1)
        flags[gid] = 0; // use last element as a n_updated counter in next kernel

    if (gid >= N-1) // инициализируем только внутренние ноды
        return;

    initFlag(flags, gid, nodes, level);
}

void growNode(__global struct Node *root, __global struct Node *nodes)
{
    __global const struct Node *left = &nodes[root->child_left];
    __global const struct Node *right = &nodes[root->child_right];

    growBBox(&root->bbox, &left->bbox);
    growBBox(&root->bbox, &right->bbox);

    double m0 = left->mass;
    double m1 = right->mass;

    root->mass = m0 + m1;

    if (root->mass <= 1e-8) {
        printf("04230420340322\n");
//        return;
    }

    root->cmsx = (left->cmsx * m0 + right->cmsx * m1) / root->mass;
    root->cmsy = (left->cmsy * m0 + right->cmsy * m1) / root->mass;
}

__kernel void growNodes(__global int *flags, __global struct Node *nodes,
                        int N, int level)
{
    int gid = get_global_id(0);

    if (gid >= N-1) // инициализируем только внутренние ноды
        return;

    __global struct Node *node = &nodes[gid];
    if (flags[gid] == level) {
        growNode(node, nodes);
        atomic_add(&flags[N-1], 1);
    }
}

// https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
bool barnesHutCondition(float x, float y, __global const struct Node *node)
{
    float dx = x - node->cmsx;
    float dy = y - node->cmsy;
    float s = max(node->bbox.maxx - node->bbox.minx, node->bbox.maxy - node->bbox.miny);
    float d2 = dx*dx + dy*dy;
    float thresh = 0.5;

    return s * s < d2 * thresh * thresh;
}

void calculateForce(float x0, float y0, float m0, __global const struct Node *nodes, __global float *force_x, __global float *force_y)
{
    float dfx = 0.0f;
    float dfy = 0.0f;

    // TODO
    int stack[2 * NBITS_PER_DIM];
    int stack_size = 0;
    // TODO кладем корень на стек
    stack[stack_size++] = 0;

    while (stack_size) {
        // TODO берем ноду со стека
        int i_node = stack[--stack_size];
        __global const struct Node *node = &nodes[i_node];

        if (isLeaf(node)) {
            continue;
        }

        {
            __global const struct Node *left = &nodes[node->child_left];
            __global const struct Node *right = &nodes[node->child_right];
            if (contains(&left->bbox, x0, y0) && contains(&right->bbox, x0, y0)) {
                if (!equals(&left->bbox, &right->bbox)) {
                    printf("42357987645432456547\n");
                }
                if (!equalsPoint(&left->bbox, x0, y0)) {
                    printf("5446456456435656\n");
                }
                continue;
            }
        }

        int children[] = {node->child_left, node->child_right};
        for (int i = 0; i < 2; ++i) {
            int i_child = children[i];
            __global const struct Node *child = &nodes[i_child];
            if (!contains(&child->bbox, x0, y0) && barnesHutCondition(x0, y0, child)) {
                // TODO посчитать взаимодействие точки с центром масс ноды
                float dx = child->cmsx - x0;
                float dy = child->cmsy - y0;
                float dr2 = max(100.f, dx * dx + dy * dy);

                float dr2_inv = 1.f / dr2;
                float dr_inv = sqrt(dr2_inv);

                float ex = dx * dr_inv;
                float ey = dy * dr_inv;

                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

                dfx += child->mass * fx;
                dfy += child->mass * fy;
            } else {
                // TODO кладем ребенка на стек
                stack[stack_size++] = i_child;
                if (stack_size >= 2 * NBITS_PER_DIM) {
                    printf("0420392384283\n");
                }
            }
        }
    }

    *force_x += dfx;
    *force_y += dfy;
}

__kernel void calculateForces(
        __global const float *pxs, __global const float *pys,
        __global const float *vxs, __global const float *vys,
        __global const float *mxs,
        __global const struct Node *nodes,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
    // TODO
    unsigned int gid = get_global_id(0);
    float x0 = pxs[gid];
    float y0 = pys[gid];
    float m0 = mxs[gid];

    __global float *dvx = dvx2d + t * N + gid;
    __global float *dvy = dvy2d + t * N + gid;

    calculateForce(x0, y0, m0, nodes, dvx, dvy);
}

__kernel void integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t,
        int coord_shift)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];

    // отражаем частицы от границ мира, чтобы не ломался подсчет мортоновского кода
    if (pxs[i] < 1) {
        vxs[i] *= -1;
        pxs[i] += vxs[i];
    }
    if (pys[i] < 1) {
        vys[i] *= -1;
        pys[i] += vys[i];
    }
    if (pxs[i] >= 2 * coord_shift - 1) {
        vxs[i] *= -1;
        pxs[i] += vxs[i];
    }
    if (pys[i] >= 2 * coord_shift - 1) {
        vys[i] *= -1;
        pys[i] += vys[i];
    }
}
