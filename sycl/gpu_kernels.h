#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>

// utility function to load from memory (try different cache hints)
#define USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS 0
#define USE_LDCS_FOR_WEIGHT_LOADS 0

__dpct_inline__ sycl::uint4 loadFromMem(const sycl::uint4 *ptr) {
    sycl::uint4 ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    /*
    DPCT1053:0: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(ret.x()), "=r"(ret.y()), "=r"(ret.z()), "=r"(ret.w())
                 : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__dpct_inline__ uint32_t loadFromMem(const uint32_t *ptr) {
    uint32_t ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    /*
    DPCT1053:1: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];"
                 : "=r"(ret)
                 : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__dpct_inline__ sycl::half loadFromMem(const sycl::half *ptr) {
    sycl::half ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    uint16_t temp;
    /*
    DPCT1053:2: Migration of device assembly code is not supported.
    */
    asm volatile("ld.global.L1::no_allocate.u16 %0, [%1];"
                 : "=h"(temp)
                 : "l"(ptr));
    ret = sycl::bit_cast<sycl::half>(temp);
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

// ----------------------------------------------------------------------------
// GPU kernels

void convert_fp16_to_fp32(float *out, sycl::half *in, int elements,
                          const sycl::nd_item<3> &item_ct1) {
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index < elements)
        out[index] = (float)in[index];
}

void copy_embedding_kernel(sycl::half *x, const sycl::half *__restrict__ table,
                           int size, int *tokens, int *pPos,
                           const sycl::nd_item<3> &item_ct1)
{
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= size) return;
    int pos = *pPos;
    int token = tokens[pos];
    int table_index = index + token * size;
    x[index] = table[table_index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
void rmsnorm_kernel(sycl::half *o, sycl::half *x, sycl::half *weight, int size,
                    int elementsPerThread, const sycl::nd_item<3> &item_ct1,
                    float &shared_ss) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = item_ct1.get_local_id(2) + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            ss += val * val;
        }
    }

    ss = sycl::reduce_over_group(item_ct1.get_group(), ss, sycl::plus<>());

    if (item_ct1.get_local_id(2) == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sycl::sqrt(ss);
        shared_ss = ss;
    }
    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = item_ct1.get_local_id(2) + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (sycl::half)val;
        }
    }
}

// Only used for the final linear layer to get logits (for most other layers we use the INT4 version below)
void mat_vec_kernel(sycl::half *op, const sycl::half *ip, const sycl::half *wt,
                    int n, int d, int numSerialLoads, int ip_stride,
                    int w_stride, int op_stride, int w_row_stride, float alpha,
                    const sycl::nd_item<3> &item_ct1) {
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
    if (index >= d)
        return;
    const sycl::half *__restrict__ input =
        ip + item_ct1.get_group(1) * ip_stride;
    const sycl::half *__restrict__ weight =
        wt + item_ct1.get_group(1) * w_stride;
    sycl::half *output = op + item_ct1.get_group(1) * op_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item_ct1.get_local_id(2)) * 8;
        if (j < n) {
            sycl::half w[8];
            sycl::half ip[8];
            *((sycl::uint4 *)(&w)) =
                loadFromMem((sycl::uint4 *)(&weight[index * w_row_stride + j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]));
            for (int el = 0; el < 8; el++)
                sum += float(w[el]) * float(ip[el]);
        }
    }

    sum =
        sycl::reduce_over_group(item_ct1.get_sub_group(), sum, sycl::plus<>());
    sum *= alpha;

    if (item_ct1.get_local_id(2) == 0)
        output[index] = (sycl::half)sum;
}

// Simpler version of the above - handles non multiple of 8 dimensions too (used only by MHA block)
void mat_vec_kernel_simple(sycl::half *op, sycl::half *ip, sycl::half *wt,
                           int n, int numSerialElements, int ip_stride,
                           int w_stride, int w_row_stride, float alpha,
                           int *pPos, int kv_mul,
                           const sycl::nd_item<3> &item_ct1) {

    int op_stride = *pPos + 1;
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
    if (index >= op_stride)
        return;

    const sycl::half *__restrict__ input =
        ip + item_ct1.get_group(1) * ip_stride;
    const sycl::half *__restrict__ weight =
        wt + (item_ct1.get_group(1) / kv_mul) * w_stride;
    sycl::half *output = op + item_ct1.get_group(1) * op_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + item_ct1.get_local_id(2);
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);
    }

    sum =
        sycl::reduce_over_group(item_ct1.get_sub_group(), sum, sycl::plus<>());
    sum *= alpha;

    if (item_ct1.get_local_id(2) == 0)
        output[index] = (sycl::half)sum;
}

// hardcoded for group-count = 128
__dpct_inline__ float
get_mat_vec_int4(int index, const sycl::half *__restrict__ input,
                 const uint32_t *__restrict__ q_weight,
                 const uint32_t *__restrict__ q_zeros,
                 const sycl::half *__restrict__ scales, int inputElements,
                 int opElements, int packed_zeros_height, int scales_height,
                 int packed_weights_height, const sycl::nd_item<3> &item_ct1) {

    float sum = 0;
    for (int ygq = 0;
         ygq * 128 + item_ct1.get_local_id(2) * 4 < packed_weights_height;
         ygq++) { // each iteration of this loop covers 8 x 128 elements in y
                  // dimension of weight matrix (weight matrix is column major)
        uint32_t packed_q_z = loadFromMem(&q_zeros[index * packed_zeros_height + ygq]);

        // load weights in one go (32 elements from weight matrix loaded by each thread in one read)
        uint32_t loaded_packed_wts[4];
        *((sycl::uint4 *)(&loaded_packed_wts[0])) = loadFromMem((
            sycl::uint4 *)(&q_weight[index * packed_weights_height + ygq * 128 +
                                     item_ct1.get_local_id(2) * 4]));

        int group_y = ygq * 8 + (item_ct1.get_local_id(2) / 4);
        float q_z =
            (float)(packed_q_z >> (4 * (item_ct1.get_local_id(2) / 4)) & 0xF);
        float scale = (float)loadFromMem(&scales[index * scales_height + group_y]);
        int y_base = ygq * 1024 + item_ct1.get_local_id(2) * 32;

        for (int qi = 0; qi < 4; qi++) {                 // each iteration of this loop covers 256 elements in y dimension of weight matrix
            int ys = y_base + qi * 8;
            if (ys < inputElements) {
                uint32_t packed_q_w = loaded_packed_wts[qi];
                sycl::half ip[8];
                *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[ys]));

                for (int i = 0; i < 8; i++) {
                    float q_wt = (float)(packed_q_w & 0xF);
                    float w = (q_wt - q_z) * scale;
                    sum += w * float(ip[i]);
                    packed_q_w = (packed_q_w >> 4);
                }
            }
        }
    }

    sum =
        sycl::reduce_over_group(item_ct1.get_sub_group(), sum, sycl::plus<>());

    return sum;
}

void mat_vec_int4(sycl::half *__restrict__ output,
                  const sycl::half *__restrict__ input,
                  const uint32_t *__restrict__ q_weight,
                  const uint32_t *__restrict__ q_zeros,
                  const sycl::half *__restrict__ scales, int inputElements,
                  int opElements, int packed_zeros_height, int scales_height,
                  int packed_weights_height, bool accum, int loff, int *pPos,
                  const sycl::nd_item<3> &item_ct1)
{
    int index = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
    if (index >= opElements)
        return;

    float sum = get_mat_vec_int4(
        index, input, q_weight, q_zeros, scales, inputElements, opElements,
        packed_zeros_height, scales_height, packed_weights_height, item_ct1);

    if (item_ct1.get_local_id(2) == 0) {
        if (loff != -1) {
            output += loff + (*pPos * opElements);
        }

        if (accum)
            sum += (float)output[index];
        output[index] = (sycl::half)sum;
    }
}

void mat_vec_kernel_int4(
    sycl::half *__restrict__ output, const sycl::half *__restrict__ input,
    const uint32_t *__restrict__ q_weight, const uint32_t *__restrict__ q_zeros,
    const sycl::half *__restrict__ scales, int inputElements, int opElements,
    int packed_zeros_height, int scales_height, int packed_weights_height,
    bool accum, int loff, int *pPos, const sycl::nd_item<3> &item_ct1)
{
    mat_vec_int4(output, input, q_weight, q_zeros, scales, inputElements,
                 opElements, packed_zeros_height, scales_height,
                 packed_weights_height, accum, loff, pPos, item_ct1);
}

void qkv_matvec_kernel(
    sycl::half *__restrict__ q, sycl::half *__restrict__ key_cache,
    sycl::half *__restrict__ value_cache, const sycl::half *__restrict__ input,
    const uint32_t *__restrict__ q_weight, const uint32_t *__restrict__ q_zeros,
    const sycl::half *__restrict__ q_scales,
    const uint32_t *__restrict__ k_weight, const uint32_t *__restrict__ k_zeros,
    const sycl::half *__restrict__ k_scales,
    const uint32_t *__restrict__ v_weight, const uint32_t *__restrict__ v_zeros,
    const sycl::half *__restrict__ v_scales, int inputElements, int opElements,
    int packed_zeros_height, int scales_height, int packed_weights_height,
    int loff, int *pPos, const sycl::nd_item<3> &item_ct1)
{
    if (item_ct1.get_group(1) == 0)
        mat_vec_int4(q, input, q_weight, q_zeros, q_scales, inputElements,
                     opElements, packed_zeros_height, scales_height,
                     packed_weights_height, false, -1, nullptr, item_ct1);
    else if (item_ct1.get_group(1) == 1)
        mat_vec_int4(key_cache, input, k_weight, k_zeros, k_scales,
                     inputElements, opElements, packed_zeros_height,
                     scales_height, packed_weights_height, false, loff, pPos,
                     item_ct1);
    else // if (blockIdx.y == 2)
        mat_vec_int4(value_cache, input, v_weight, v_zeros, v_scales,
                     inputElements, opElements, packed_zeros_height,
                     scales_height, packed_weights_height, false, loff, pPos,
                     item_ct1);
}

void ffn_matvec_silu_kernel(
    sycl::half *__restrict__ output, const sycl::half *__restrict__ input,
    const uint32_t *__restrict__ g_weight, const uint32_t *__restrict__ g_zeros,
    const sycl::half *__restrict__ g_scales,
    const uint32_t *__restrict__ u_weight, const uint32_t *__restrict__ u_zeros,
    const sycl::half *__restrict__ u_scales, int inputElements, int opElements,
    int packed_zeros_height, int scales_height, int packed_weights_height,
    const sycl::nd_item<3> &item_ct1) {

    int index = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
    if (index >= opElements)
        return;

    float g_val = get_mat_vec_int4(
        index, input, g_weight, g_zeros, g_scales, inputElements, opElements,
        packed_zeros_height, scales_height, packed_weights_height, item_ct1);
    float u_val = get_mat_vec_int4(
        index, input, u_weight, u_zeros, u_scales, inputElements, opElements,
        packed_zeros_height, scales_height, packed_weights_height, item_ct1);

    // apply silu and write the result
    if (item_ct1.get_local_id(2) == 0) {
        float val = g_val;
        val *= 1.0f / (1.0f + sycl::native::exp(-val));
        val *= u_val;
        output[index] = (sycl::half)val;
    }
}

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
void vec_mat_kernel(sycl::half *op, const sycl::half *__restrict__ ip,
                    const sycl::half *__restrict__ wt, int N, int *pPos,
                    int w_stride, int op_stride, int w_row_stride, int kv_mul,
                    const sycl::nd_item<3> &item_ct1,
                    sycl::local_accessor<sycl::half, 3> loaded_fragment) {
    int K = *pPos + 1;
    const sycl::half *__restrict__ input = ip + item_ct1.get_group(1) * K;
    const sycl::half *__restrict__ weight =
        wt + (item_ct1.get_group(1) / kv_mul) * w_stride;
    sycl::half *output = op + item_ct1.get_group(1) * op_stride;

    int start_n = item_ct1.get_group(2) * 32;
    int i = start_n + item_ct1.get_local_id(1);

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts

    // OOB check
    if (i >= N)
        return;

    // load the first 32x32 fragment
    int n = start_n + item_ct1.get_local_id(2);
    int k = item_ct1.get_local_id(1);
    int offset = k * w_row_stride + n;
    loaded_fragment[0][item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] =
        ((n < N) && (k < K)) ? weight[offset] : (sycl::half)0.0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; ;) {
        /*
        DPCT1118:3: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(
            sycl::access::fence_space::local_space); // wait for the load

        int start_k = e * 32;
        if (start_k >= K) break;
        k = start_k + item_ct1.get_local_id(2);
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][item_ct1.get_local_id(2)]
                                    [item_ct1.get_local_id(1)]) *
               ((k < K) ? (float)input[k] : 0.0f);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + item_ct1.get_local_id(2);
        k = start_k + item_ct1.get_local_id(1);
        int offset = k * w_row_stride + n;
        loaded_fragment[buf_i][item_ct1.get_local_id(1)][item_ct1.get_local_id(
            2)] = ((n < N) && (k < K)) ? weight[offset] : (sycl::half)0.0;
    }

    sum =
        sycl::reduce_over_group(item_ct1.get_sub_group(), sum, sycl::plus<>());

    if (item_ct1.get_local_id(2) == 0)
        output[i] = (sycl::half)sum;
}

// Each block processes a single head
void RoPERotation_kernel(sycl::half *sq, sycl::half *sk_base, int num_kv_heads,
                         int head_size, int *pPos, int loff, float rope_theta,
                         const sycl::nd_item<3> &item_ct1) {
    int pos = *pPos;

    int h = item_ct1.get_group(2);
    sycl::half *q = sq + h * head_size;
    int i = item_ct1.get_local_id(2);
    int head_dim = (i * 2) % head_size;
    float freq = 1.0f / sycl::pow(rope_theta, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = sycl::cos(val);
    float fci = sycl::sin(val);
    float q0 = q[i];
    float q1 = q[i + head_size / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    if (h < num_kv_heads) {
        sycl::half *sk = sk_base + loff + pos * num_kv_heads * head_size;
        sycl::half *k = sk + h * head_size;
        float k0 = k[i];
        float k1 = k[i + head_size / 2];
        k[i] = k0 * fcr - k1 * fci;
        k[i + head_size / 2] = k0 * fci + k1 * fcr;
    }
}

void softmax_kernel(sycl::half *__restrict__ arr, int num_heads, int *pPos,
                    const sycl::nd_item<3> &item_ct1, float *att,
                    float &shared_val) {

    int h = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    int step = item_ct1.get_local_range(2);
    int size = *pPos + 1;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float)arr[h * size + t];
    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val,
                                      sycl::maximum<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = max_val;
    /*
    DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = sycl::native::exp(att[i] - max_val);
        sum += att[i];
    }

    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = sum;
    /*
    DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (sycl::half)(att[t] / sum);
}

void softmax_kernel_no_smem(sycl::half *arr, int num_heads, int *pPos,
                            const sycl::nd_item<3> &item_ct1,
                            float &shared_val) {
    int h = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    int step = item_ct1.get_local_range(2);
    int size = *pPos + 1;

    // find max value (for numerical stability)
    float max_val = tid < size ? (float)arr[h * size + tid] : 0;
    for (int i = tid + step; i < size; i += step)
    {
        float val = (float)arr[h * size + i];
        if (val > max_val)
            max_val = val;
    }

    max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val,
                                      sycl::maximum<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = max_val;
    /*
    DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float val = (float)arr[h * size + i];
        val = sycl::native::exp(val - max_val);
        arr[h * size + i] = (sycl::half)val;
        sum += val;
    }

    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = sum;
    /*
    DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (sycl::half)(float(arr[h * size + t]) / sum);
}

void argmax_kernel(sycl::half *__restrict__ x, int size, int *result,
                   volatile int *pPos, int *pPosGpu, bool write_token,
                   const sycl::nd_item<3> &item_ct1, float &shared_val,
                   int &global_max_pos) {

    int tid = item_ct1.get_local_id(2);
    int step = item_ct1.get_local_range(2);

    // find local max value and its position
    float max_val = tid < size ? (float)x[tid] : -INFINITY;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if ((float)x[i] > max_val) {
            max_val = x[i];
            max_pos = i;
        }
    }

    // find the global max value
    float global_max_val;
    global_max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val,
                                             sycl::maximum<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = global_max_val;
    /*
    DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    global_max_val = shared_val;

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.

    if (max_val == global_max_val) {
        global_max_pos = max_pos;
    }
    /*
    DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write next token to the current token location
    if (item_ct1.get_local_id(2) == 0) {
        int token_pos = *pPos;
        token_pos++;

        if (write_token)
            result[token_pos] = global_max_pos;

        // update the token indices (unblocks the CPU)
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}

// This is used for Top-P sampling. We do the following:
// 1. Divide the logits by temperature
// 2. Compute softmax
// 3. Write the indices in an array
void softmax_logits_kernel(sycl::half *__restrict__ logits, int size,
                           float temperature, int *indices,
                           const sycl::nd_item<3> &item_ct1,
                           float &shared_val) {
    int tid = item_ct1.get_local_id(2);
    int step = item_ct1.get_local_range(2);

    for (int t = tid; t < size; t += step)
    {
        // first just write the indices array
        indices[t] = t;

        // divide by temperature
        float val = (float)logits[t];
        val /= temperature;
        logits[t] = (sycl::half)val;
    }
    /*
    DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // Compute the softmax
    // find max value (for numerical stability)
    float max_val = tid < size ? ((float)logits[tid]) : -FLT_MAX;
    for (int i = tid + step; i < size; i += step)
        if ((float)logits[i] > max_val)
            max_val = logits[i];

    max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val,
                                      sycl::maximum<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = max_val;
    /*
    DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float v = sycl::native::exp(float(logits[i]) - max_val);
        logits[i] = (sycl::half)v;
        sum += v;
    }

    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0)
        shared_val = sum;
    /*
    DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        logits[t] = (sycl::half)(float(logits[t]) / sum);
}

// ----------------------------------------------------------------------------

// find the index in the array that crosses top-p threshold
void sample_top_p_kernel(sycl::half *sorted_logits_prefix_sum, int *indices,
                         int n, float top_p_threshold, int *result,
                         volatile int *pPos, int *pPosGpu,
                         const sycl::nd_item<3> &item_ct1)
{
    int tid = item_ct1.get_local_id(2);
    int step = item_ct1.get_local_range(2);

    int min_index = n - 1;

    for (int t = tid; t < n; t += step) {
        if ((float)(sorted_logits_prefix_sum[t]) >= top_p_threshold) {
            if (t < min_index) {
                min_index = t;
            }
        }
    }

    // find the min across the block
    int min_index_global = sycl::reduce_over_group(
        item_ct1.get_group(), min_index, sycl::minimum<>());
    if (item_ct1.get_local_id(2) == 0)
    {
        int token_pos = *pPos;
        token_pos++;
        result[token_pos] = indices[min_index_global];

        // update the token indices
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}

