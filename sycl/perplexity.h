#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#pragma once

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// tokens is an array of integers representing the input text
// logits is a 2D array of floats representing the predicted probabilities of each token for each possible word in the vocabulary
// vocab_size is a constant integer representing the size of the vocabulary
// num_tokens is an integer representing the number of tokens in the input text
float compute_perplexity(int* tokens, float* logits, int num_tokens, int vocab_size) {
    // initialize a variable to store the sum of log probabilities
    double sum = 0.0;
    // loop through each token in the input text
    for (int i = 0; i < num_tokens; i++) {
        // get the index of the actual word in the vocabulary
        int word_index = tokens[i];

        // the logits that we get from GPU are pre-softmax, need to apply softmax first
        softmax(&logits[i * vocab_size], vocab_size);

        // get the predicted probability of that word from the logits array
        double prob = logits[i * vocab_size + word_index];

        //printf(" %g,", prob);

        // add the log probability to the sum
        sum += log(prob);
    }

    // compute the average log probability
    double avg_log_prob = sum / num_tokens;
    // compute the perplexity as the exponentiation of the negative average log probability
    return float(exp(-avg_log_prob));
}


void run_transformer(bool gen_token, Config* p, RunState* s, TransformerWeights* w, bool copyLogits, Sampler* pSampler);

// ----------------------------------------------------------------------------
float get_dataset_perplexity(char *dataset, Tokenizer *tokenizer,
                             Config *config, RunState *state,
                             TransformerWeights *weights, Sampler *pSampler) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int bytes = strlen(dataset);
    int* datasetTokens = &(state->shared_data->tokens[1]);

    printf("\nTokenizing Dataset...");
    int totalTokens;
    encode(tokenizer, dataset, 0, 0, datasetTokens, &totalTokens);
    printf("done!\n");

    printf("Found %d characters, %d tokens", bytes, totalTokens);

    int numTokens = totalTokens;
    if (numTokens >= config->seq_len) {
        numTokens = config->seq_len - 1;
        printf("\nTruncated to %d tokens", numTokens);
    }

    printf("\nRunning the network to get logits...");
    // run the transformer model to get logits
    q_ct1.memset(state->pos, 0, sizeof(int)).wait();
    state->shared_data->pos = 0;
    state->shared_data->tokens[0] = bos_token;
    for (int pos = 0; pos < numTokens; pos++) {
        run_transformer(false, config, state, weights, true, pSampler);
        dev_ct1.queues_wait_and_throw();
    }
    printf("done!\n");

    printf("Computing perplexity...");

    // copy the logits and compute preplexity
    float* logits_arr = (float*)malloc(numTokens * config->vocab_size * sizeof(float));
    q_ct1.memcpy(logits_arr, state->logits_array, numTokens * config->vocab_size * sizeof(float)).wait();

    float pplx = compute_perplexity(datasetTokens, logits_arr, numTokens, config->vocab_size);

    printf("\nPerplexity computed on %d tokens: %f\n\n", numTokens, pplx);
    free(logits_arr);
    return pplx;
}


void parseDataSetAndComputePreplexity(char* textFileName, Tokenizer* tokenizer, Config* config, RunState* state, TransformerWeights* weights, Sampler *pSampler)
{
    FILE* fp = fopen(textFileName, "rb+");
    printf("\nLoading Dataset...");

    // find the number of bytes in the file
    fseek(fp, 0, SEEK_END);
    int bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *dataset = (char*)malloc(bytes + 1);

    fread(dataset, 1, bytes, fp);
    fclose(fp);
    printf("done!\n");

    dataset[bytes] = 0;     // null terminate in case it wasn't

    int count = 0;
    double pplx_product = 1;

    // search for <|endoftext|> and break down the dataset into multiple sequences
    char* currentSeq = dataset;
    while (currentSeq) {
        char* nextseq;
        if (nextseq = strstr(currentSeq, "<|endoftext|>")) {
            *nextseq = 0;
            nextseq += 13;
            pplx_product *= get_dataset_perplexity(currentSeq, tokenizer, config, state, weights, pSampler);
            count++;
            currentSeq = nextseq;
        }
        else {
            pplx_product *= get_dataset_perplexity(currentSeq, tokenizer, config, state, weights, pSampler);
            count++;
            break;
        }
    }

    free(dataset);
    printf("\nGeomean perplexity on %d sequences: %f\n\n", count, pow(pplx_product, 1.0 / count));
}
