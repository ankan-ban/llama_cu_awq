#include "common.h"
#include "gpu_kernels.h"
#include "tokenizer.h"
#include "sampler.h"
#include "transformer.h"

typedef struct Model_t
{
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
} Model;

extern "C" Model* init_model(char* checkpoint_path, char* tokenizer_path, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    Model* m = (Model*)malloc(sizeof(Model));
    build_transformer(&m->transformer, checkpoint_path, false);
    build_tokenizer(&m->tokenizer, tokenizer_path, vocab_size);
    build_sampler(&m->sampler, vocab_size, temperature, topp, rng_seed);
    cudaStreamCreate(&stream);
    return m;
}

extern "C" void generate(Model* m, char* prompt, int steps) {
    generate(&m->transformer, &m->tokenizer, &m->sampler, prompt, steps);
}

extern "C" void free_model(Model* m) {
    free_transformer(&m->transformer);
    free_tokenizer(&m->tokenizer);
    free(m);
}
