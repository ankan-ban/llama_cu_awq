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
    int head_size = p->dim / p->n_heads;
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
    softmax_kernel <<< num_heads, 1024, 0, stream >>> (att, num_heads, pPos);

    // 3. weighted sum of the values to get the final result
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim, 0, stream >>> (output, att, value_cache, head_size, pPos, head_size, head_size, dim / kv_mul, kv_mul);
}

void siluElementwiseMul(half *hb, half *hb2, int size) {
   silu_element_wise_mul_kernel <<< divUp(size, 256), 256, 0, stream >>> (hb, hb2, size);
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
        matmul(s->q, s->xb, w->layers[l].wq_q, dim, dim);
        matmul(s->key_cache, s->xb, w->layers[l].wq_k, dim, kv_dim, false, loff, pPos);
        matmul(s->value_cache, s->xb, w->layers[l].wq_v, dim, kv_dim, false, loff, pPos);

        // apply RoPE rotation to the q and k vectors for each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(s->q, s->key_cache, p->n_heads, p->n_kv_heads, head_size, pPos, loff, p->rope_theta);

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, kv_mul, seq_len_bin, pPos);

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
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

constexpr int MAX_GRAPHS = 8;
cudaGraphExec_t cudaGraphInstance[MAX_GRAPHS];
bool graphCaptured[MAX_GRAPHS];

void transformer(bool gen_token, Config* p, RunState* s, TransformerWeights* w, bool copyLogits, Sampler *pSampler) {
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

// ----------------------------------------------------------------------------
void error_usage(char *argv[]) {
    fprintf(stderr, "Usage:   %s <checkpoint> [options]\n", argv[0]);
    fprintf(stderr, "Example: %s model.bin -n 256 -i \"Write a poem on GPUs\"\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <int>    max number of steps to run for, default = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.5\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -q <string> compute perplexity on the given dataset file\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {

    // default parameters
    char* checkpoint_path = NULL;  // e.g. out/model.bin
    char* tokenizer_path = "tokenizer.bin";
    char* dataset_path = NULL;
    int steps = 0;              // number of steps to run for
    char* prompt = nullptr;     // prompt string
    bool perplexity = false;
    float temperature = 0.5f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.6f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    unsigned long long rng_seed = 0; // seed rng with time by default

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
            case 'q': {
                dataset_path = argv[i + 1];
                perplexity = true;
                break;
            }
            default: error_usage(argv);
        }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // read in the model.bin file
    Config config = {};
    TransformerWeights weights;
    FILE* file = nullptr;
    file = fopen(checkpoint_path, "rb");
    if (!file) { printf("Couldn't open file %s\n", checkpoint_path); return 1; }
    // read in the config header
    if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
    // Dump model config
    printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\nrope_theta: %g\n",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.n_layers, config.seq_len, config.vocab_size, config.rope_theta);
    config.vocab_size = abs(config.vocab_size);
    // read in the Transformer weights
    malloc_weights(&weights, &config);
    if (checkpoint_init_weights(&weights, &config, file)) { return 1; }

    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    // create and init the tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config, perplexity);
    cudaStreamCreate(&stream);

    int *prompt_tokens = (int*)malloc(config.seq_len * sizeof(int));
    int num_prompt_tokens = 0;

    char input_message[2048];
    if (prompt) strcpy(input_message, prompt);
    else input_message[0] = 0;

    if (perplexity) {
        parseDataSetAndComputePreplexity(dataset_path, &tokenizer, &config, &state, &weights, &sampler);
    } 
    else
    while (1) {
        encode(&tokenizer, input_message, 1, 0, prompt_tokens, &num_prompt_tokens);
        //printf("\nPrompt tokens: %d - \n", num_prompt_tokens);
        //for (int i = 0; i < num_prompt_tokens; i++) printf("%d ", prompt_tokens[i]);
        //printf("\n");

        // start the main loop
        long start = time_in_ms();  // used to time our code
        int next;                   // will store the next token in the sequence
        int token = bos_token;      // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;                // position in the sequence

        // copy the prompt tokens into shared list of tokens (so that GPU can access them).
        // init state
        cudaMemset(state.pos, 0, sizeof(int));
        state.shared_data->pos = 0;
        memcpy(&state.shared_data->tokens, prompt_tokens, sizeof(int) * num_prompt_tokens);

        printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons
        while (pos < steps) {
            // wait for GPU work for previous iteration to complete
            // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
            cudaStreamSynchronize(stream);
            // Perf note: don't put CPU work here "before" calling transformer as it won't overlap with GPU execution.
            transformer(pos >= num_prompt_tokens - 1, &config, &state, &weights, false, &sampler); // forward the transformer to get next token

            if (pos > 0)
            {
                next = state.shared_data->tokens[pos];  // Note: this is output token from previous iteration
                char* piece = decode(&tokenizer, token, next);
                safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
                if (next == eos_token) break; // break if EOS token is reached

                // advance forward
                token = next;
            }
            pos++;
        }

        // report achieved tok/s
        long end = time_in_ms();
        double time = (end - start) / 1000.0;
        int timed_tokens = pos - 1;
        printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);

        printf("enter next prompt: ");
        fgets(input_message, sizeof(input_message), stdin);
        // strip newline
        size_t len = strlen(input_message);
        if (len > 0 && input_message[len - 1] == '\n') {
            input_message[len - 1] = '\0';
        }
    }

    // memory cleanup
    free_run_state(&state);
    free_weights(&weights);
#if USE_CUDA_GRAPHS
    for (int i = 0; i < MAX_GRAPHS; i++)
        if (graphCaptured[i]) cudaGraphExecDestroy(cudaGraphInstance[i]);
#endif

    free_tokenizer(&tokenizer);
    if (prompt_tokens != NULL) free(prompt_tokens);
    return 0;
}