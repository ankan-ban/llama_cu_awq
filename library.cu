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
    memset(m, 0, sizeof(Model));
    build_transformer(&m->transformer, checkpoint_path, false);
    build_tokenizer(&m->tokenizer, tokenizer_path, vocab_size);
    build_sampler(&m->sampler, vocab_size, temperature, topp, rng_seed);
    cudaStreamCreate(&stream);
    return m;
}

typedef void (*Handler)(char*);

extern "C" void generate(Model* m, char* prompt, int steps, Handler handler) {
    Tokenizer* tokenizer = &m->tokenizer;
    Transformer* transformer = &m->transformer;
    Sampler* sampler = &m->sampler;

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence
    int last = num_prompt_tokens - 1;

    cudaMemset(transformer->state.pos, 0, sizeof(int));
    transformer->state.shared_data->pos = 0;
    memcpy(&transformer->state.shared_data->tokens, prompt_tokens, sizeof(int) * num_prompt_tokens);

    while (pos < steps) {
        cudaStreamSynchronize(stream);
        run_transformer(pos >= last, &transformer->config, &transformer->state, &transformer->weights, false, sampler);
        if (pos > last) {
            next = transformer->state.shared_data->tokens[pos];
            if (next == eos_token) break;
            char* piece = decode(tokenizer, token, next);
            handler(piece);
            token = next;
        }
        pos++;
    }

    free(prompt_tokens);
}

extern "C" void free_model(Model* m) {
    free_transformer(&m->transformer);
    free_tokenizer(&m->tokenizer);
    free(m);
}
