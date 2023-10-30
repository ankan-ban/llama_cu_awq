#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>


// utility function to load from memory (try different cache hints)
#define USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS 1
#define USE_LDCS_FOR_WEIGHT_LOADS 0

__forceinline__ __device__ uint4 loadFromMem(const uint4* ptr) {
    uint4 ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__forceinline__ __device__ uint32_t loadFromMem(const uint32_t* ptr) {
    uint32_t ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__forceinline__ __device__ half loadFromMem(const half* ptr) {
    half ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    uint16_t temp;
    asm volatile("ld.global.L1::no_allocate.u16 %0, [%1];" : "=h"(temp) : "l"(ptr));
    ret = __ushort_as_half(temp);
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}




// ----------------------------------------------------------------------------
// GPU kernels

__global__ void convert_fp16_to_fp32(float* out, half* in, int elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}

__global__ void copy_embedding_kernel(half* x, const half* __restrict__ table, int size, int* tokens, int* pPos)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    int pos = *pPos;
    int token = tokens[pos];
    int table_index = index + token * size;
    x[index] = table[table_index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            ss += val * val;
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}


// Only used for the final linear layer to get logits (for most other layers we use the INT4 version below)
__global__ void mat_vec_kernel(half* op, const half* ip, const half* wt, int n, int d, int numSerialLoads,
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + threadIdx.x) * 8;
        if (j < n) {
            half w[8];
            half ip[8];
            *((uint4*)(&w)) = loadFromMem((uint4*)(&weight[index * w_row_stride + j]));
            *((uint4*)(&ip)) = *((uint4*)(&input[j]));
            for (int el = 0; el < 8; el++)
                sum += float(w[el]) * float(ip[el]);
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// Simpler version of the above - handles non multiple of 8 dimensions too (used only by MHA block)
__global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int numSerialElements,
    int ip_stride, int w_stride, int w_row_stride, float alpha, int* pPos, int kv_mul) {

    int op_stride = *pPos + 1;
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= op_stride)
        return;

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// hardcoded for group-count = 128
__forceinline__ __device__ float get_mat_vec_int4(int index, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height) {

    float sum = 0;
    for (int ygq = 0; ygq * 128 + threadIdx.x * 4 < packed_weights_height; ygq++) {   // each iteration of this loop covers 8 x 128 elements in y dimension of weight matrix (weight matrix is column major)
        uint32_t packed_q_z = loadFromMem(&q_zeros[index * packed_zeros_height + ygq]);

        // load weights in one go (32 elements from weight matrix loaded by each thread in one read)
        uint32_t loaded_packed_wts[4];
        *((uint4*)(&loaded_packed_wts[0])) = loadFromMem((uint4*)(&q_weight[index * packed_weights_height + ygq * 128 + threadIdx.x * 4]));

        int group_y = ygq * 8 + (threadIdx.x / 4);
        float q_z = (float)(packed_q_z >> (4 * (threadIdx.x / 4)) & 0xF);
        float scale = (float)loadFromMem(&scales[index * scales_height + group_y]);
        int y_base = ygq * 1024 + threadIdx.x * 32;

        for (int qi = 0; qi < 4; qi++) {                 // each iteration of this loop covers 256 elements in y dimension of weight matrix
            int ys = y_base + qi * 8;
            if (ys < inputElements) {
                uint32_t packed_q_w = loaded_packed_wts[qi];
                half ip[8];
                *((uint4*)(&ip)) = *((uint4*)(&input[ys]));

                for (int i = 0; i < 8; i++) {
                    float q_wt = (float)(packed_q_w & 0xF);
                    float w = (q_wt - q_z) * scale;
                    sum += w * float(ip[i]);
                    packed_q_w = (packed_q_w >> 4);
                }
            }
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    return sum;
}


__device__ void mat_vec_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;


    float sum = get_mat_vec_int4(index, input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    if (threadIdx.x == 0) {
        if (loff != -1) {
            output += loff + (*pPos * opElements);
        }

        if (accum)
            sum += (float)output[index];
        output[index] = (half)sum;
    }
}

__global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    mat_vec_int4(output, input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, accum, loff, pPos);
}

__global__ void qkv_matvec_kernel(half* __restrict__ q, half* __restrict__ key_cache, half* __restrict__ value_cache, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ q_scales,
    const uint32_t* __restrict__ k_weight, const uint32_t* __restrict__ k_zeros, const half* __restrict__ k_scales,
    const uint32_t* __restrict__ v_weight, const uint32_t* __restrict__ v_zeros, const half* __restrict__ v_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, int loff, int* pPos)
{
    if (blockIdx.y == 0)
        mat_vec_int4(q, input, q_weight, q_zeros, q_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, -1, nullptr);
    else if (blockIdx.y == 1)
        mat_vec_int4(key_cache, input, k_weight, k_zeros, k_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, loff, pPos);
    else // if (blockIdx.y == 2)
        mat_vec_int4(value_cache, input, v_weight, v_zeros, v_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, loff, pPos);
}

__global__ void  ffn_matvec_silu_kernel(half* __restrict__ output, const half* __restrict__ input, 
    const uint32_t* __restrict__ g_weight, const uint32_t* __restrict__ g_zeros, const half* __restrict__ g_scales, 
    const uint32_t* __restrict__ u_weight, const uint32_t* __restrict__ u_zeros, const half* __restrict__ u_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height) {

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;

    float g_val = get_mat_vec_int4(index, input, g_weight, g_zeros, g_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);
    float u_val = get_mat_vec_int4(index, input, u_weight, u_zeros, u_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    // apply silu and write the result
    if (threadIdx.x == 0) {
        float val = g_val;
        val *= 1.0f / (1.0f + expf(-val));
        val *= u_val;
        output[index] = (half)val;
    }
}

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
__global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt, int N, int* pPos, int w_stride, int op_stride, int w_row_stride, int kv_mul) {
    int K = *pPos + 1;
    const half* __restrict__ input = ip + blockIdx.y * K;
    const half* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;
    half* output = op + blockIdx.y * op_stride;

    int start_n = blockIdx.x * 32;
    int i = start_n + threadIdx.y;

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts
    __shared__ half loaded_fragment[2][32][32 + 2];

    // OOB check
    if (i >= N)
        return;

    // load the first 32x32 fragment
    int n = start_n + threadIdx.x;
    int k = threadIdx.y;
    int offset = k * w_row_stride + n;
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; ;) {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        if (start_k >= K) break;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float)input[k] : 0.0f);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        int offset = k * w_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[i] = (half)sum;
}

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sq, half* sk_base, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
    int pos = *pPos;

    int h = blockIdx.x;
    half* q = sq + h * head_size;
    int i = threadIdx.x;
    int head_dim = (i * 2) % head_size;
    float freq = 1.0f / powf(rope_theta, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float q0 = q[i];
    float q1 = q[i + head_size / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    if (h < num_kv_heads) {
        half* sk = sk_base + loff + pos * num_kv_heads * head_size;
        half* k = sk + h * head_size;
        float k0 = k[i];
        float k1 = k[i + head_size / 2];
        k[i] = k0 * fcr - k1 * fci;
        k[i + head_size / 2] = k0 * fci + k1 * fcr;
    }
}

__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int* pPos) {
    __shared__ float att[MAX_SEQ_LEN_SMEM_KERNEL];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = *pPos + 1;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float)arr[h * size + t];
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);
        sum += att[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half)(att[t] / sum);
}

__global__ void softmax_kernel_no_smem(half* arr, int num_heads, int* pPos) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = *pPos + 1;

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? (float)arr[h * size + tid] : 0;
    for (int i = tid + step; i < size; i += step)
    {
        float val = (float)arr[h * size + i];
        if (val > max_val)
            max_val = val;
    }

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float val = (float)arr[h * size + i];
        val = expf(val - max_val);
        arr[h * size + i] = (half)val;
        sum += val;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half)(float(arr[h * size + t]) / sum);
}

__global__ void argmax_kernel(half* __restrict__ x, int size, int* result, volatile int* pPos, int* pPosGpu, bool write_token) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

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
    global_max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.
    __shared__ int global_max_pos;
    if (max_val == global_max_val) {
        global_max_pos = max_pos;
    }
    __syncthreads();

    // write next token to the current token location
    if (threadIdx.x == 0) {
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
__global__ void softmax_logits_kernel(half* __restrict__ logits, int size, float temperature, int *indices) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    
    for (int t = tid; t < size; t += step)
    {
        // first just write the indices array
        indices[t] = t;

        // divide by temperature
        float val = (float)logits[t];
        val /= temperature;
        logits[t] = (half)val;
    }
    __syncthreads();

    // Compute the softmax
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? ((float)logits[tid]) : -FLT_MAX;
    for (int i = tid + step; i < size; i += step)
        if ((float)logits[i] > max_val)
            max_val = logits[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float v = expf(float(logits[i]) - max_val);
        logits[i] = (half)v;
        sum += v;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        logits[t] = (half)(float(logits[t]) / sum);
}

// ----------------------------------------------------------------------------

// find the index in the array that crosses top-p threshold
__global__ void sample_top_p_kernel(half* sorted_logits_prefix_sum, int* indices, int n, float top_p_threshold, int* result, volatile int* pPos, int* pPosGpu)
{
    int tid = threadIdx.x;
    int step = blockDim.x;

    int min_index = n - 1;

    for (int t = tid; t < n; t += step) {
        if ((float)(sorted_logits_prefix_sum[t]) >= top_p_threshold) {
            if (t < min_index) {
                min_index = t;
            }
        }
    }

    // find the min across the block
    using BlockReduce = cub::BlockReduce<int, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    int min_index_global = BlockReduce(temp).Reduce(min_index, cub::Min());
    if (threadIdx.x == 0)
    {
        int token_pos = *pPos;
        token_pos++;
        result[token_pos] = indices[min_index_global];

        // update the token indices
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}

