## llama2_q4.cu

Simple and fast Pure Cuda inference for 4-bit AWQ quantized models (https://github.com/mit-han-lab/llm-awq).

Based on llama2.c (https://github.com/karpathy/llama2.c/)

1. First generate AWQ int-4 quantized weights following steps in https://github.com/mit-han-lab/llm-awq
 E.g:
```
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-chat-metadata.pt
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-chat-metadata.pt --q_backend real --dump_quant awq_weights/llama2-7b-awq.pt
```
 Note - AWQ scripts doesn't run on Windows. Use Linux or WSL.

2. Convert AWQ weights into individual weight binary files using convert_awq_to_bin.py

3. Convert/repack the weight binary files using the weight_repacker.cpp utility.

4. Run this program pointing to the final weight file.

## Sample output and performance
We get ~200 tokens per second with RTX 4090 for 7b paramater models:

```
llama2_q4_opt.exe C:\LLM\llama2-7b-awq-q4.bin 256 "write an essay about GPUs"

Model params:-
dim: 4096
hidden_dim: 11008
n_heads: 32
n_kv_heads: 32
n_layers: 32
seq_len: 2048
vocab_size: 32000


loaded weights
<s>
write an essay about GPUs

Introduction:

GPU (Graphics Processing Unit) is a specialized electronic circuit designed to accelerate the manipulation of graphical data. It is a key component of a computer's hardware that is used to improve the performance of graphics-intensive applications such as video games, computer-aided design (CAD) software, and scientific simulations. In this essay, we will explore the history of GPUs, their architecture, and their impact on the computer industry.
History of GPUs:
The concept of a GPU can be traced back to the 1960s when computer graphics were still in their infancy. At that time, computer graphics were primarily used for scientific visualization and were not yet a major component of mainstream computing. However, as computer graphics became more popular in the 1980s and 1990s, the need for specialized hardware to handle the increasingly complex graphics tasks became apparent. In the early 1990s, the first GPUs were developed, which were designed to offload the computationally intensive graphics tasks from the CPU (Central Processing Unit) to the GPU.
Architecture
achieved tok/s: 200.787402. Tokens: 255, seconds: 1.27
```
## License

MIT
