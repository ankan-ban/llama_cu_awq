/*
Inference for Llama-2 Transformer model in pure Cuda.

### INT4 - AWQ quantization version ###

1. First generate AWQ int-4 quantized weights following steps in https://github.com/mit-han-lab/llm-awq
 E.g:
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-chat-metadata.pt
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-chat-metadata.pt --q_backend real --dump_quant awq_weights/llama2-7b-awq.pt
 Note - AWQ scripts doesn't run on Windows. Use Linux or WSL.

2. Convert AWQ weights into individual weight binary files using convert_awq_to_bin.py

3. Convert/repack the weight binary files using the weight_packer.cpp utility.

4. Run this program pointing to the final weight file.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

constexpr int group_size = 128; // hardcoded for this implementation
#define DUMP_PER_TOKEN_TIMINGS 0

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void convert_fp32_to_fp16(half* out, float* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (half)in[index];
}

__global__ void convert_fp16_to_fp32(float* out, half* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
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
        ss += 1e-6f;
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


// Note that ~95% of total time is spent here, so optimizing this is important
// 1. One output generated per warp so that we can parallelize the dot product across the warp
// 2. We load 8 elements at a time for efficiency (assume dimensions to be multiple of 8)
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
            *((uint4 *)(&w)) = *((uint4 *)(&weight[index * w_row_stride + j]));
            *((uint4 *)(&ip)) = *((uint4 *)(&input[j]));
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

// Simpler version of the above - to handle non multiple of 8 dimensions too (needed for MHA block)
__global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int d, int numSerialElements,
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

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
__global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;

    float sum = 0;
    for (int ygq = 0; ygq < packed_zeros_height; ygq++) {   // each iteration of this loop covers 8 x 128 elements in y dimension of weight matrix (weight matrix is column major)
        uint32_t packed_q_z = q_zeros[index * packed_zeros_height + ygq];
        for (int qi = 0; qi < 8; qi += 2) {                 // each iteration of this loop covers 256 elements in y dimension of weight matrix
            int group_y_base = ygq * 8 + qi;
            int group_y = group_y_base + (threadIdx.x / 16);                    // First 16 threads of warp use first of the 2 groups, second set of 16 threads use the other
            float q_z = (float)(packed_q_z >> (4 * (threadIdx.x / 16)) & 0xF);
            packed_q_z = (packed_q_z >> 8);
            float scale = (float)scales[index * scales_height + group_y];

            // no loop here but because all warps are working in parallel, we cover 256 elements here (32 x 8 elements per thread)
            int ys = group_y_base * 128 + (threadIdx.x * 8);
            if (ys < inputElements) {
                uint32_t packed_q_w = q_weight[index * packed_weights_height + ys / 8];
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

    if (threadIdx.x == 0) {
        if (accum)
            sum += (float)output[index];
        output[index] = (half)sum;
    }
}

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
__global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt, int N, int K, int elementsPerThread,
    int ip_stride, int w_stride, int op_stride, int w_row_stride) {

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
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
    for (int e = 0; e < elementsPerThread;) {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float) input[k] : 0.0f);

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
__global__ void RoPERotation_kernel(half* sq, half* sk, int num_heads, int head_size, int pos) {
    int h = blockIdx.x;
    half* q = sq + h * head_size;
    half* k = sk + h * head_size;
    int i = threadIdx.x;
    int head_dim = (i * 2) % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float q0 = q[i];
    float q1 = q[i + head_size/2];
    float k0 = k[i];
    float k1 = k[i + head_size / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + head_size / 2] = k0 * fci + k1 * fcr;
}

#define MAX_SEQ_LEN 8192
__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int size) {
    __shared__ float att[MAX_SEQ_LEN];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float) arr[h * size + t];
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
        arr[h * size + t] = (half) (att[t] / sum);
}

__global__ void silu_element_wise_mul_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i];
        dest[i] = (half)val;
    }
}

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

struct QWeight {
    uint32_t* weight;
    uint32_t* zeros;
    half* scales;
};

struct PerLayerWeight {
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    QWeight wq_q;
    QWeight wq_k;
    QWeight wq_v;
    QWeight wq_o;
    QWeight wq_gate;
    QWeight wq_up;
    QWeight wq_down;
};

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)
    // classifier weights for the logits, on the last layer
    half* wcls;
    // final rmsnorm
    half* rms_final_weight; // (dim,)
    // Per layer weights
    PerLayerWeight* layers;
    int num_layers;
} TransformerWeights;

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)
    half* xb; // same, but inside a residual branch (dim,)
    half* xb2; // an additional buffer just for convenience (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; // query (dim,)
    half* att; // buffer for scores/attention values (n_heads, seq_len)
    half* logits_gpu; // output logits
    float* logits_temp; // logits in GPU memory converted to float
    float* logits; // logits copied CPU side
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * p->dim * sizeof(half));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_temp, p->vocab_size * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->logits || !s->key_cache
        || !s->value_cache || !s->logits_gpu || !s->logits_temp) {
        printf("malloc failed for allocaing run state!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits_gpu);
    cudaFree(s->logits_temp);
    free(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
}

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

void allocQWeight(QWeight* pWeight, size_t height, size_t width) {
    size_t packed_wt_height = divUp(height, 8);
    size_t scales_height = divUp(height, group_size);
    size_t packed_zeros_height = divUp(scales_height, 8);

    cudaMalloc((void**)&pWeight->weight, packed_wt_height * width * sizeof(uint32_t));
    cudaMalloc((void**)&pWeight->zeros, packed_zeros_height * width * sizeof(uint32_t));
    cudaMalloc((void**)&pWeight->scales, scales_height * width * sizeof(half));
}

void freeQWeight(QWeight* pWeight) {
    cudaFree(pWeight->weight);
    cudaFree(pWeight->zeros);
    cudaFree(pWeight->scales);
}

void malloc_weights(TransformerWeights* w, Config* p) {
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half));
    w->layers = (PerLayerWeight*)malloc(p->n_layers * sizeof(PerLayerWeight));
    w->num_layers = p->n_layers;
    for (int l = 0; l < p->n_layers; l++)
    {
        PerLayerWeight* layer = &(w->layers[l]);
        cudaMalloc((void**)&layer->rms_att_weight,  p->dim * sizeof(half));
        cudaMalloc((void**)&layer->rms_ffn_weight,  p->dim * sizeof(half));
        allocQWeight(&layer->wq_q, p->dim, p->dim);
        allocQWeight(&layer->wq_k, p->dim, p->dim);
        allocQWeight(&layer->wq_v, p->dim, p->dim);
        allocQWeight(&layer->wq_o, p->dim, p->dim);
        allocQWeight(&layer->wq_gate, p->dim, p->hidden_dim);
        allocQWeight(&layer->wq_up, p->dim, p->hidden_dim);
        allocQWeight(&layer->wq_down, p->hidden_dim, p->dim);
    }

    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(half));
    int head_size = p->dim / p->n_heads;
    cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->layers ||
        !w->rms_final_weight || !w->wcls) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_weights(TransformerWeights* w) {
    cudaFree(w->token_embedding_table);
    cudaFree(w->rms_final_weight);
    cudaFree(w->wcls);
    for (int l = 0; l < w->num_layers; l++) {
        PerLayerWeight* layer = &(w->layers[l]);
        cudaFree(layer->rms_att_weight);
        cudaFree(layer->rms_ffn_weight);
        freeQWeight(&layer->wq_q);
        freeQWeight(&layer->wq_k);
        freeQWeight(&layer->wq_v);
        freeQWeight(&layer->wq_o);
        freeQWeight(&layer->wq_gate);
        freeQWeight(&layer->wq_up);
        freeQWeight(&layer->wq_down);
    }
    free(w->layers);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint
void readWeight(void* op, FILE* fp, size_t bytes, void* scratch) {
    if (fread(scratch, 1, bytes, fp) != bytes) { printf("error reading weights");  exit(1); }
    cudaMemcpyAsync(op, scratch, bytes, cudaMemcpyHostToDevice);
}

void uploadQWeight(QWeight& weight, FILE* fp, size_t height, size_t width, void* scratch) {
    int meta_height = divUp(height, group_size);
    int packed_wt_height = divUp(height, 8);
    int packed_zeros_height = divUp(meta_height, 8);

    readWeight(weight.weight, fp, packed_wt_height * width * sizeof(uint32_t), scratch);
    readWeight(weight.zeros,  fp, packed_zeros_height * width * sizeof(uint32_t), scratch);
    readWeight(weight.scales, fp, meta_height * width * sizeof(half), scratch);
}

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f) {
    size_t scratch_size = std::max(p->vocab_size, p->hidden_dim) * p->dim;
    scratch_size *= sizeof(half);
    void* scratchCpu = malloc(scratch_size);

    readWeight(w->token_embedding_table, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    readWeight(w->wcls, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    readWeight(w->rms_final_weight, f, p->dim * sizeof(half), scratchCpu);

    // upload decoder block weight for each layer
    for (int i = 0; i < p->n_layers; i++) {
        uploadQWeight(w->layers[i].wq_q, f, p->dim, p->dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_k, f, p->dim, p->dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_v, f, p->dim, p->dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_o, f, p->dim, p->dim, scratchCpu);

        uploadQWeight(w->layers[i].wq_up  , f, p->dim, p->hidden_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_gate, f, p->dim, p->hidden_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_down, f, p->hidden_dim, p->dim, scratchCpu);

        readWeight(w->layers[i].rms_att_weight, f, p->dim * sizeof(half), scratchCpu);
        readWeight(w->layers[i].rms_ffn_weight, f, p->dim * sizeof(half), scratchCpu);
    }

    printf("\nloaded weights\n");
    free(scratchCpu);
    return 0;
}


// ----------------------------------------------------------------------------
// neural net blocks

void rmsnorm(half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<< 1, 1024 >>> (o, x, weight, size, elementsPerThread);
}

void matmul(half* xout, half* x, half* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;
    if ((n & 7) || (d & 7))
        mat_vec_kernel_simple <<<grid_dim, block_dim >>> (xout, x, w, n, d, serialElements, x_stride, w_stride, op_stride, w_row_stride, alpha);
    else
        mat_vec_kernel <<<grid_dim, block_dim >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha);
}

void matmul(half* xout, half* x, QWeight &w, int inpSize, int opSize, bool accum = false) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(1); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = divUp(inpSize, 8);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    mat_vec_kernel_int4 <<<grid_dim, block_dim >>> (xout, x, w.weight, w.zeros, w.scales, inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, accum);
}

void RoPERotation(half *q, half *k, int num_heads, int head_size, int pos) {
    RoPERotation_kernel <<<num_heads, head_size / 2 >>> (q, k, num_heads, head_size, pos);
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half * value_cache, half *att, int num_heads, int head_size, int seq_len) {
    int dim = head_size * num_heads;
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(seq_len, 32), num_heads);
    mat_vec_kernel_simple <<< grid_dim1, block_dim >>> (att, q, key_cache, head_size, seq_len, serialElements, head_size, head_size, seq_len, dim, 1.0 / sqrt(head_size));

    // 2. Run softmax kernel
    softmax_kernel <<< num_heads, 1024 >>> (att, num_heads, seq_len);

    // 3. weighted sum of the values to get the final result
    serialElements = divUp(seq_len, 32);    
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim >>> (output, att, value_cache, head_size, seq_len, serialElements, seq_len, head_size, head_size, dim);
}

void siluElementwiseMul(half *hb, half *hb2, int size) {
   silu_element_wise_mul_kernel <<< divUp(size, 256), 256 >>> (hb, hb2, size);
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    // copy the token embedding into x
    half* content_row = &(w->token_embedding_table[token * dim]);
    cudaMemcpyAsync(x, content_row, dim * sizeof(half), cudaMemcpyDeviceToDevice);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_att_weight, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        half* key_cache_row = s->key_cache + loff + pos * dim;
        half* value_cache_row = s->value_cache + loff + pos * dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->layers[l].wq_q, dim, dim);
        matmul(key_cache_row, s->xb, w->layers[l].wq_k, dim, dim);
        matmul(value_cache_row, s->xb, w->layers[l].wq_v, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(s->q, key_cache_row, p->n_heads, head_size, pos);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, pos+1);

        // final matmul to get the output of the attention fused with residual connection back into x
        matmul(s->x, s->xb, w->layers[l].wq_o, dim, dim, true);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_ffn_weight, dim);

        // apply gate and up proj (opt: can be done in single kernel as batch of 2)
        matmul(s->hb, s->xb, w->layers[l].wq_gate, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->layers[l].wq_up, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        matmul(s->x, s->hb, w->layers[l].wq_down, hidden_dim, dim, true);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    // copy logits from GPU->CPU
    convert_fp16_to_fp32 <<<divUp(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits_gpu, p->vocab_size);
    
#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf(" t: %g ", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    cudaMemcpy(s->logits, s->logits_temp, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

int str_lookup(char *str, char **vocab, int vocab_size) {
    // find the first perfect match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {
    
    // a temporary buffer to merge two consecutive tokens
    char* str_buffer = (char*) malloc((max_token_length*2+1) * sizeof(char)); // *2 for concat, +1 for null terminator

    // first encode every individual byte in the input string
    *n_tokens = 0; // the number of tokens
    for (char *c = text; *c != '\0'; c++) {
        sprintf(str_buffer, "%c", *c);
        int id = str_lookup(str_buffer, vocab, vocab_size);
        if (id == -1) { printf("not good\n"); exit(1);}
        tokens[*n_tokens] = id;
        (*n_tokens)++;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // poor man's C argparse
    char *checkpoint = NULL;  // e.g. out/model.bin
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    char *prompt = NULL;      // prompt string

    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [steps] [prompt]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        steps = atoi(argv[2]);
    }
    if (argc >= 4) {
        prompt = argv[3];
    }

    // read in the model.bin file
    Config config = {};
    TransformerWeights weights;
    {
        FILE* file = nullptr;

        file = fopen(checkpoint, "rb");
        if (!file) { printf("Couldn't open file %s\n", checkpoint); return 1; }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }

        // Dump model config
        printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\n\n",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.n_layers, config.seq_len, config.vocab_size);

        config.vocab_size = abs(config.vocab_size);
        // read in the Transformer weights
        malloc_weights(&weights, &config);
        if (checkpoint_init_weights(&weights, &config, file)) { return 1; }
    }
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));
    unsigned int max_token_length;
    {
        FILE *file = fopen("tokenizer.bin", "rb");
        if (!file) { printf("couldn't load tokenizer.bin\n"); return 1; }
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) { printf("failed read\n"); return 1; }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) { printf("failed read\n"); return 1;}
            if (fread(&len, sizeof(int), 1, file) != 1) { printf("failed read\n"); return 1; }
            vocab[i] = (char *)malloc(len + 1);
            if (fread(vocab[i], len, 1, file) != 1) { printf("failed read\n"); return 1; }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);

    // process the prompt, if any
    int *prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    prompt_tokens = (int*)malloc(config.seq_len * sizeof(int));

    char input_message[2048];
    strcpy(input_message, prompt);

    while (1)
    {
        if (input_message != NULL) {
            bpe_encode(input_message, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
        }

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence
        printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons
        while (pos < steps) {

            // forward the transformer to get logits for the next token
            transformer(token, pos, &config, &state, &weights);

            if (pos < num_prompt_tokens) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos];
            }
            else {
                // sample the next token using greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size);
            }

            // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
            char* token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next] + 1 : vocab[next];
            printf("%s", token_str);
            //printf(" [%d - %s] ", next, token_str);
            fflush(stdout);

            if (next == 2) break; // break if EOS token is reached

            // advance forward
            token = next;
            pos++;
            // init our timer here because the first iteration could be slow
            if (start == 0) { start = time_in_ms(); }
        }

        // report achieved tok/s
        long end = time_in_ms();
        double time = (end - start) / 1000.0;
        int timed_tokens = pos - 1;
        printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);

        printf("enter next prompt: ");
        fgets(input_message, sizeof(input_message), stdin);
    }

    // memory cleanup
    free_run_state(&state);
    free_weights(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    if (prompt_tokens != NULL) free(prompt_tokens);
    return 0;
}
