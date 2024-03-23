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
#include "transformer.h"

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