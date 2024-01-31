#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#pragma once

typedef struct dpct_type_113308 {
    int vocab_size = 0;
    int* indices = nullptr;
    void* tempStorage_scan = nullptr;
    void* tempStorage_sort = nullptr;
    size_t temp_storage_bytes_scan = 0;
    size_t temp_storage_bytes_sort = 0;
    float temperature = 0;
    float topp = 0;
    unsigned long long rng_state = 0;
} Sampler;

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;

    // buffer only used with nucleus sampling
    sampler->indices =
        sycl::malloc_device<int>(vocab_size, dpct::get_in_order_queue());
}

void destroy_sampler(Sampler *sampler) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::free(sampler->indices, q_ct1);
    sycl::free(sampler->tempStorage_sort, q_ct1);
    sycl::free(sampler->tempStorage_scan, q_ct1);
}

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

// sample the token given the logits and some hyperparameters
void sample(Sampler *sampler, RunState *s, bool gen_token,
            dpct::queue_ptr stream) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);

    if (sampler->temperature == 0.0f || !gen_token) {
        // greedy argmax sampling: take the token with the highest probability
        /*
        DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 0> shared_val_acc_ct1(cgh);
            sycl::local_accessor<int, 0> global_max_pos_acc_ct1(cgh);

            sycl::half *s_logits_ct0 = s->logits;
            int sampler_vocab_size_ct1 = sampler->vocab_size;
            int *s_shared_data_tokens_ct2 = &(s->shared_data->tokens[0]);
            volatile int *s_shared_data_pos_ct3 = &(s->shared_data->pos);
            int *s_pos_ct4 = s->pos;

            cgh.parallel_for<dpct_kernel_name<class argmax_kernel_f109c8>>(
                sycl::nd_range(sycl::range(1, 1, 1024),
                               sycl::range(1, 1, 1024)),
                [=](sycl::nd_item<3> item_ct1) {
                    argmax_kernel(s_logits_ct0, sampler_vocab_size_ct1,
                                  s_shared_data_tokens_ct2,
                                  s_shared_data_pos_ct3, s_pos_ct4, gen_token,
                                  item_ct1, shared_val_acc_ct1,
                                  global_max_pos_acc_ct1);
                });
        });
    }
    else {
        // apply the temperature to the logits, and then perform softmax
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<float, 0> shared_val_acc_ct1(cgh);

                sycl::half *s_logits_ct0 = s->logits;
                int sampler_vocab_size_ct1 = sampler->vocab_size;
                float sampler_temperature_ct2 = sampler->temperature;
                int *sampler_indices_ct3 = sampler->indices;

                cgh.parallel_for<
                    dpct_kernel_name<class softmax_logits_kernel_e48cfc>>(
                    sycl::nd_range(sycl::range(1, 1, 1024),
                                   sycl::range(1, 1, 1024)),
                    [=](sycl::nd_item<3> item_ct1) {
                        softmax_logits_kernel(
                            s_logits_ct0, sampler_vocab_size_ct1,
                            sampler_temperature_ct2, sampler_indices_ct3,
                            item_ct1, shared_val_acc_ct1);
                    });
            });
        }

        float threshold = 0.0f;
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            threshold = coin;
        }
        else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            if (sampler->temp_storage_bytes_sort == 0) {
                dpct::sort_pairs(oneapi::dpl::execution::device_policy(*stream),
                                 s->logits, s->logits, sampler->indices,
                                 sampler->indices, sampler->vocab_size, true, 0,
                                 sizeof(sycl::half) * 8);
                sampler->tempStorage_sort = (void *)sycl::malloc_device(
                    sampler->temp_storage_bytes_sort, q_ct1);
            }

            dpct::sort_pairs(oneapi::dpl::execution::device_policy(*stream),
                             s->logits, s->logits, sampler->indices,
                             sampler->indices, sampler->vocab_size, true, 0,
                             sizeof(sycl::half) * 8);
            threshold = coin * sampler->topp;
        }

        // Sample from the predicted probability distribution
        if (sampler->temp_storage_bytes_scan == 0) {
            oneapi::dpl::inclusive_scan(
                oneapi::dpl::execution::device_policy(*stream), s->logits,
                s->logits + sampler->vocab_size, s->logits);
            sampler->tempStorage_scan = (void *)sycl::malloc_device(
                sampler->temp_storage_bytes_scan, q_ct1);
        }
        oneapi::dpl::inclusive_scan(
            oneapi::dpl::execution::device_policy(*stream), s->logits,
            s->logits + sampler->vocab_size, s->logits);

        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});
            stream->submit([&](sycl::handler &cgh) {
                sycl::half *s_logits_ct0 = s->logits;
                int *sampler_indices_ct1 = sampler->indices;
                int sampler_vocab_size_ct2 = sampler->vocab_size;
                int *s_shared_data_tokens_ct4 = &(s->shared_data->tokens[0]);
                volatile int *s_shared_data_pos_ct5 = &(s->shared_data->pos);
                int *s_pos_ct6 = s->pos;

                cgh.parallel_for<
                    dpct_kernel_name<class sample_top_p_kernel_46e8ca>>(
                    sycl::nd_range(sycl::range(1, 1, 1024),
                                   sycl::range(1, 1, 1024)),
                    [=](sycl::nd_item<3> item_ct1) {
                        sample_top_p_kernel(s_logits_ct0, sampler_indices_ct1,
                                            sampler_vocab_size_ct2, threshold,
                                            s_shared_data_tokens_ct4,
                                            s_shared_data_pos_ct5, s_pos_ct6,
                                            item_ct1);
                    });
            });
        }
    }
}