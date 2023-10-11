#pragma once

#include <stdint.h>
#include <cuda_fp16.h>

constexpr int MAX_SEQ_LEN_SMEM_KERNEL = 8192; // 8k is the max sequence length supported by the kernel that uses shared memory
constexpr int MAX_SEQ_LEN = 128 * 1024;       // Can be arbitirarily large, but we need to allocate memory for the whole sequence

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 32000 for llama2 models.
    int seq_len; // max sequence length
    float rope_theta; // theta for the rope rotational embedding
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

// data shared between CPU and GPU (allocated in host memory)
struct SharedData {
    volatile int pos;         // current token index
    int tokens[MAX_SEQ_LEN];  // seq_len (tokens processed/generated so far) allocated in host memory so that CPU can read this
};

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)
    half* xb; // same, but inside a residual branch (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; // query (dim,)
    half* att; // buffer for scores/attention values (n_heads, seq_len)
    half* logits; // output logits
    // kv cache
    half* key_cache;   // (layer, seq_len, kv_dim)
    half* value_cache; // (layer, seq_len, kv_dim)

    int* pos;  // GPU copy of the current position (just 1 element)
    SharedData* shared_data;

    float* logits_array;  // array of output logits used to compute perplexity (seq_len, vocab_size)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}
