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
#include <time.h>
#include <math.h>
#include <time.h>

#include "common.h"
#include "gpu_kernels.h"
#include "tokenizer.h"
#include "sampler.h"
#include "perplexity.h"

constexpr int group_size = 128; // hardcoded for this implementation
#define DUMP_PER_TOKEN_TIMINGS 0
#define USE_CUDA_GRAPHS 1

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

void malloc_run_state(RunState* s, Config* p, bool allocLogitsArray) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, sizeof(half) * p->n_layers * p->seq_len * kv_dim);    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, sizeof(half) * p->n_layers * p->seq_len * kv_dim);

    cudaMalloc((void**)&s->pos, sizeof(int));
    cudaMallocHost((void**)&s->shared_data, sizeof(SharedData));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->pos || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->logits || !s->key_cache
        || !s->value_cache || !s->shared_data) {
        printf("malloc failed for allocaing run state!\n");
        exit(EXIT_FAILURE);
    }

    if (allocLogitsArray) {
        cudaMalloc((void**)&s->logits_array, sizeof(float) * p->seq_len * p->vocab_size);
        if (!s->logits_array) {
            printf("malloc failed for allocaing logits_array!\n");
            exit(EXIT_FAILURE);
        }
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->pos);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    cudaFreeHost(s->shared_data);
}

size_t getPackedWeightHeight(size_t height)
{
    // Each uint32 element in the packed weight matrix contain 8 elements from the original matrix.
    // Also we load 4 uint's (32 elements) in a single instruction for getting better memory efficiency
    // This requires us to align the "height" dimension to a multiple of 4 uint (or 32 elements)
    return divUp(height, 32) * 4;
}

void allocQWeight(QWeight* pWeight, size_t height, size_t width) {
    size_t packed_wt_height = getPackedWeightHeight(height);
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
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half));
    w->layers = (PerLayerWeight*)malloc(p->n_layers * sizeof(PerLayerWeight));
    w->num_layers = p->n_layers;
    for (int l = 0; l < p->n_layers; l++)
    {
        PerLayerWeight* layer = &(w->layers[l]);
        cudaMalloc((void**)&layer->rms_att_weight,  p->dim * sizeof(half));
        cudaMalloc((void**)&layer->rms_ffn_weight,  p->dim * sizeof(half));
        allocQWeight(&layer->wq_q, p->dim, p->dim);
        allocQWeight(&layer->wq_k, p->dim, kv_dim);
        allocQWeight(&layer->wq_v, p->dim, kv_dim);
        allocQWeight(&layer->wq_o, p->dim, p->dim);
        allocQWeight(&layer->wq_gate, p->dim, p->hidden_dim);
        allocQWeight(&layer->wq_up, p->dim, p->hidden_dim);
        allocQWeight(&layer->wq_down, p->hidden_dim, p->dim);
    }

    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(half));
    cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->layers ||
        !w->rms_final_weight || !w->wcls) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
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
    if (fread(scratch, 1, bytes, fp) != bytes) { printf("error reading weights");  exit(EXIT_FAILURE); }
    cudaMemcpyAsync(op, scratch, bytes, cudaMemcpyHostToDevice);
}

void uploadQWeight(QWeight& weight, FILE* fp, size_t height, size_t width, void* scratch) {
    int meta_height = divUp(height, group_size);
    int packed_wt_height = getPackedWeightHeight(height);
    int packed_zeros_height = divUp(meta_height, 8);

    readWeight(weight.weight, fp, packed_wt_height * width * sizeof(uint32_t), scratch);
    readWeight(weight.zeros,  fp, packed_zeros_height * width * sizeof(uint32_t), scratch);
    readWeight(weight.scales, fp, meta_height * width * sizeof(half), scratch);
}

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f) {
    size_t scratch_size = std::max(p->vocab_size, p->hidden_dim) * p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    scratch_size *= sizeof(half);
    void* scratchCpu = malloc(scratch_size);

    printf("\nLoading Weights... ");

    readWeight(w->token_embedding_table, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    readWeight(w->wcls, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    readWeight(w->rms_final_weight, f, p->dim * sizeof(half), scratchCpu);

    // upload decoder block weight for each layer
    for (int i = 0; i < p->n_layers; i++) {
        uploadQWeight(w->layers[i].wq_q, f, p->dim, p->dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_k, f, p->dim, kv_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_v, f, p->dim, kv_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_o, f, p->dim, p->dim, scratchCpu);

        uploadQWeight(w->layers[i].wq_up  , f, p->dim, p->hidden_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_gate, f, p->dim, p->hidden_dim, scratchCpu);
        uploadQWeight(w->layers[i].wq_down, f, p->hidden_dim, p->dim, scratchCpu);

        readWeight(w->layers[i].rms_att_weight, f, p->dim * sizeof(half), scratchCpu);
        readWeight(w->layers[i].rms_ffn_weight, f, p->dim * sizeof(half), scratchCpu);
    }

    printf("done!\n");
    free(scratchCpu);
    return 0;
}


// ----------------------------------------------------------------------------
// neural net blocks
cudaStream_t stream;

void rmsnorm(half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<< 1, 1024, 0, stream>>> (o, x, weight, size, elementsPerThread);
}

void matmul(half* xout, half* x, half* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    if ((n & 7) || (d & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;
    mat_vec_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha);
}

void matmul(half* xout, half* x, QWeight &w, int inpSize, int opSize, bool accum = false, int loff = -1, int *pPos = nullptr) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    mat_vec_kernel_int4 <<<grid_dim, block_dim, 0, stream >>> (xout, x, w.weight, w.zeros, w.scales, inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, accum, loff, pPos);
}

void qkv_matvec(half* q, half *key_cache, half *value_cache, half* x, QWeight& qw, QWeight& kw, QWeight& vw, int inpSize, int opSize, int loff, int* pPos) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 3);
    qkv_matvec_kernel <<<grid_dim, block_dim, 0, stream >>> (q, key_cache, value_cache, x,
                                                             qw.weight, qw.zeros, qw.scales, 
                                                             kw.weight, kw.zeros, kw.scales, 
                                                             vw.weight, vw.zeros, vw.scales, 
                                                             inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height, loff, pPos);
}

void ffn_matvec_silu(half* xout, half* x, QWeight& gate_w, QWeight& up_w, int inpSize, int opSize) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int scales_height = divUp(inpSize, 128);
    int packed_wt_height = getPackedWeightHeight(inpSize);
    int packed_zeros_height = divUp(scales_height, 8);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    ffn_matvec_silu_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, gate_w.weight, gate_w.zeros, gate_w.scales, 
                                                                  up_w.weight, up_w.zeros, up_w.scales, 
                                                                  inpSize, opSize, packed_zeros_height, scales_height, packed_wt_height);
}

void RoPERotation(half *q, half *k, int num_heads, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
    RoPERotation_kernel <<<num_heads, head_size / 2, 0, stream >>> (q, k, num_kv_heads, head_size, pPos, loff, rope_theta);
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half * value_cache, half *att, int num_heads, int head_size, int kv_mul, int max_seq_len, int *pPos) {
    int dim = head_size * num_heads;
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(max_seq_len, 32), num_heads);      // using max_seq_len instead of real seq_len here has measurable impact on perf (2%) :-/
    mat_vec_kernel_simple <<< grid_dim1, block_dim, 0, stream >>> (att, q, key_cache, head_size, serialElements, head_size, head_size, dim / kv_mul, 1.0 / sqrt(head_size), pPos, kv_mul);

    // 2. Run softmax kernel
    if (max_seq_len <= MAX_SEQ_LEN_SMEM_KERNEL)
        softmax_kernel <<< num_heads, 1024, 0, stream >>> (att, num_heads, pPos);
    else
        softmax_kernel_no_smem <<< num_heads, 1024, 0, stream >>> (att, num_heads, pPos);

    // 3. weighted sum of the values to get the final result
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim, 0, stream >>> (output, att, value_cache, head_size, pPos, head_size, head_size, dim / kv_mul, kv_mul);
}

void run_llama_network(int *pPos, Config* p, RunState* s, TransformerWeights* w, int seq_len_bin) {
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery

    copy_embedding_kernel <<<divUp(dim, 256), 256, 0, stream >>> (x, w->token_embedding_table, dim, s->shared_data->tokens, pPos);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_att_weight, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

        // qkv matmuls for this position (opt: can be done in single kernel as batch of 3 - but only when num_kv_heads == num_heads)
        if (dim == kv_dim) {
            qkv_matvec(s->q, s->key_cache, s->value_cache, s->xb, w->layers[l].wq_q, w->layers[l].wq_k, w->layers[l].wq_v, dim, dim, loff, pPos);
        }
        else {
            matmul(s->q, s->xb, w->layers[l].wq_q, dim, dim);
            matmul(s->key_cache, s->xb, w->layers[l].wq_k, dim, kv_dim, false, loff, pPos);
            matmul(s->value_cache, s->xb, w->layers[l].wq_v, dim, kv_dim, false, loff, pPos);
        }

        // apply RoPE rotation to the q and k vectors for each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(s->q, s->key_cache, p->n_heads, p->n_kv_heads, head_size, pPos, loff, p->rope_theta);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, kv_mul, seq_len_bin, pPos);

        // final matmul to get the output of the attention fused with residual connection back into x
        matmul(s->x, s->xb, w->layers[l].wq_o, dim, dim, true);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_ffn_weight, dim);

        // apply gate proj and up proj and then the silu activation in a single fused kernel
        ffn_matvec_silu(s->hb, s->xb, w->layers[l].wq_gate, w->layers[l].wq_up, dim, hidden_dim);

        // final matmul (down proj) to get the output of the ffn fused with residual connection back into x
        matmul(s->x, s->hb, w->layers[l].wq_down, hidden_dim, dim, true);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

constexpr int MAX_GRAPHS = 8;
cudaGraphExec_t cudaGraphInstance[MAX_GRAPHS];
bool graphCaptured[MAX_GRAPHS];

void run_transformer(bool gen_token, Config* p, RunState* s, TransformerWeights* w, bool copyLogits, Sampler *pSampler) {
#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
#endif

    int seq_len = s->shared_data->pos + 1;
#if USE_CUDA_GRAPHS
    int graphIndex;
    int seq_len_bin = 128;
    for (graphIndex = 0; graphIndex < MAX_GRAPHS - 1; seq_len_bin *= 2, graphIndex++)
        if (seq_len <= seq_len_bin) break;
    if ((seq_len > seq_len_bin) || (graphIndex == MAX_GRAPHS - 1)) seq_len_bin = p->seq_len;    // last bin holds max seq len

    if (!graphCaptured[graphIndex])
    {
        cudaGraph_t graph = {};
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        run_llama_network(s->pos, p, s, w, seq_len_bin);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&cudaGraphInstance[graphIndex], graph, 0);
        cudaGraphDestroy(graph);
        graphCaptured[graphIndex] = true;
    }
    cudaGraphLaunch(cudaGraphInstance[graphIndex], stream);
#else
    run_llama_network(s->pos, p, s, w, seq_len);
#endif

    if (copyLogits) {
        // copy to the right slot in logits_array (and convert to FP32)
        // we compute perplexity on the CPU later.
        float* pOutput = s->logits_array + p->vocab_size * s->shared_data->pos;
        convert_fp16_to_fp32 << < divUp(p->vocab_size, 128), 128, 0, stream >> > (pOutput, s->logits, p->vocab_size);
    }

    sample(pSampler, s, gen_token, stream);

#if DUMP_PER_TOKEN_TIMINGS == 1
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf(" t: %g ", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


void build_transformer(Transformer* t, char* checkpoint_path, bool perplexity) {
    // read in the model.bin file
    FILE* file = nullptr;
    file = fopen(checkpoint_path, "rb");
    if (!file) { printf("Couldn't open file %s\n", checkpoint_path); exit(1); }
    // read in the config header
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { printf("Invalid header size\n");  exit(1); }
    // Dump model config
    printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\nrope_theta: %g\n",
        t->config.dim, t->config.hidden_dim, t->config.n_heads, t->config.n_kv_heads, t->config.n_layers, t->config.seq_len, t->config.vocab_size, t->config.rope_theta);

    // read in the Transformer weights
    malloc_weights(&t->weights, &t->config);
    if (checkpoint_init_weights(&t->weights, &t->config, file)) { exit(1); }

    malloc_run_state(&t->state, &t->config, perplexity);

    fclose(file);
}

void free_transformer(Transformer* t) {
    // free the RunState buffers
    free_run_state(&t->state);
    free_weights(&t->weights);
}

// ----------------------------------------------------------------------------
// generation loop
void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps) {
    char empty_prompt[] = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    printf("\nEncoding Prompt... ");   // Encoding can take a long time, print a message to show progress
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    printf("Done!\n");

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = time_in_ms();    // used to time our code
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence

    // copy the prompt tokens into shared list of tokens (so that GPU can access them).
    // init state
    cudaMemset(transformer->state.pos, 0, sizeof(int));
    transformer->state.shared_data->pos = 0;
    memcpy(&transformer->state.shared_data->tokens, prompt_tokens, sizeof(int) * num_prompt_tokens);

    while (pos < steps) {
        // wait for GPU work for previous iteration to complete
        // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
        cudaStreamSynchronize(stream);
        // Perf note: don't put CPU work here "before" calling transformer as it won't overlap with GPU execution.
        run_transformer(pos >= num_prompt_tokens - 1, &transformer->config, &transformer->state, &transformer->weights, false, sampler); // forward the transformer to get next token

        if (pos > 0) {
            next = transformer->state.shared_data->tokens[pos];  // Note: this is output token from previous iteration
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);             // same as printf("%s", piece), but skips "unsafe" bytes
            if (next == eos_token) break;   // break if EOS token is reached
            // advance forward
            token = next;
        }
        pos++;
    }
    printf("\n");

    // report achieved tok/s
    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    int timed_tokens = pos - 1;
    printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
    char* cli_user_prompt, char* cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int pos = 0;     // position in the sequence

    // init GPU state
    cudaMemset(transformer->state.pos, pos, sizeof(int));
    transformer->state.shared_data->pos = pos;

    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                }
                else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            }
            else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            }
            else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }

            printf("\nRendered prompt: %s\n", rendered_prompt); // Ankan - test!

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");

            // copy encoded tokens to GPU
            memcpy(&transformer->state.shared_data->tokens[pos], prompt_tokens, sizeof(int) * num_prompt_tokens);
        }

        // wait for GPU work for previous iteration to complete
        // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
        cudaStreamSynchronize(stream);
        run_transformer(user_idx >= num_prompt_tokens - 1, &transformer->config, &transformer->state, &transformer->weights, false, sampler); // forward the transformer to get next token

        user_idx++;

        if (user_idx > 0) {
            next = transformer->state.shared_data->tokens[pos];  // Note: this is output token from previous iteration
            if (next == eos_token) { 
                user_turn = 1;  // EOS token ends the Assistant turn
                printf("\n");
            } 
            else if (user_idx > num_prompt_tokens) {
                char* piece = decode(tokenizer, token, next);
                safe_printf(piece);
            }

            // advance forward
            token = next;
        }
        pos++;
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
void error_usage(char *argv[]) {
    fprintf(stderr, "Usage:   %s <checkpoint> [options]\n", argv[0]);
    fprintf(stderr, "Example: %s model.bin -n 256 -i \"Write a poem on GPUs\"\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <int>    max number of steps to run for, default = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -f <string> path to file containing input prompt. Can be used with for multi-line prompts.\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.5\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|perplexity, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -q <string> dataset file for computing perplexity\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {

    // default parameters
    char* checkpoint_path = NULL;  // e.g. out/model.bin
    char default_tokenizer_path[] = "tokenizer.bin";
    char* tokenizer_path = default_tokenizer_path;
    char* dataset_path = NULL;
    int steps = 0;              // number of steps to run for
    char* prompt = nullptr;     // prompt string
    bool perplexity = false;
    float temperature = 0.5f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.6f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    unsigned long long rng_seed = 0; // seed rng with time by default
    char default_mode[] = "generate";
    char* mode = default_mode;  // generate|chat
    char* system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(argv); }

    for (int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(argv); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(argv); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(argv); } // must be -x (one dash, one letter)
        // read in the args
        switch (argv[i][1]) {
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': rng_seed = atoi(argv[i + 1]); break;
            case 'm': mode = argv[i + 1]; break;
            case 'y': system_prompt = argv[i + 1]; break;
            case 'q': {
                dataset_path = argv[i + 1];
                break;
            }
            case 'f': {
                FILE* file = fopen(argv[i + 1], "r");
                if (!file) { printf("Couldn't open file %s\n", argv[i + 1]); exit(1); }
                fseek(file, 0, SEEK_END);
                long fsize = ftell(file);
                fseek(file, 0, SEEK_SET);
                if (prompt) { printf("Warning: -f overrides -i\n"); }
                prompt = (char*)malloc(fsize + 1);
                fread(prompt, fsize, 1, file);
                fclose(file);
                prompt[fsize] = 0;
                break;
            }
            default: error_usage(argv);
        }
    }

    if (strcmp(mode, "perplexity") == 0) perplexity = true;

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (!perplexity && dataset_path)
        printf("Warning: dataset path is ignored in non-perplexity mode\n");

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, perplexity);
    if (steps <= 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // create and init the tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    cudaStreamCreate(&stream);

    if (perplexity)
        parseDataSetAndComputePreplexity(dataset_path, &tokenizer, &transformer.config, &transformer.state, &transformer.weights, &sampler);
    else if (strcmp(mode, "generate") == 0)
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    else if (strcmp(mode, "chat") == 0)
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    else
        error_usage(argv);

    // memory cleanup
    free_transformer(&transformer);
#if USE_CUDA_GRAPHS
    for (int i = 0; i < MAX_GRAPHS; i++)
        if (graphCaptured[i]) cudaGraphExecDestroy(cudaGraphInstance[i]);
#endif

    free_tokenizer(&tokenizer);
    return 0;
}