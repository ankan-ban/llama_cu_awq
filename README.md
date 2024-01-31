## llama2_q4.cu

Simple and fast Pure Cuda inference for 4-bit [AWQ](https://github.com/mit-han-lab/llm-awq) quantized models

Based on [llama2.c](https://github.com/karpathy/llama2.c)

## sycl / llama2_q4.sycl.cpp
Sycl inference on **Intel dGPU/iGPU** platforms. More details [here](./sycl)

## Build

```
git clone https://github.com/ankan-ban/llama_cu_awq
cd llama_cu_awq
mdkir build
cd build
cmake ..
cmake --build . --config Release
cd ..
```

## Run

The simpler way is to download a pre-converted model from Huggingface, but you can also do all the steps

### Using Huggingface

You can use one of these models:

* 7B: [Llama-2-7b-chat-hf-w4-g128-awq](https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq)
* 13B: [Llama-2-13b-chat-hf-w4-g128-awq](https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq)

Here are the commands for the 7B model:

```
wget https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/pytorch_model.bin
wget https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq/resolve/main/config.json

pip install numpy torch
python3 convert_awq_to_bin.py pytorch_model.bin output
./weight_packer config.json output llama2-7b-awq-q4.bin 1

./llama2_q4 llama2-7b-awq-q4.bin -n 256 -i "write an essay about GPUs"
```

And here are the commands for the 13B model:

```
wget https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/config.json
wget https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/pytorch_model-00001-of-00003.bin
wget https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/pytorch_model-00002-of-00003.bin
wget https://huggingface.co/abhinavkulkarni/meta-llama-Llama-2-13b-chat-hf-w4-g128-awq/resolve/main/pytorch_model-00003-of-00003.bin

pip install numpy torch
python3 convert_awq_to_bin.py pytorch_model-00001-of-00003.bin output
python3 convert_awq_to_bin.py pytorch_model-00002-of-00003.bin output
python3 convert_awq_to_bin.py pytorch_model-00003-of-00003.bin output

./weight_packer config.json output llama2-13b-awq-q4.bin 1

./llama2_q4 llama2-13b-awq-q4.bin -n 256 -i "write an essay about GPUs"
```
Note: the last argument of weight_packer is used to indicate whether the awq weights are using old packing format (that need repacking). If you use latest AWQ repo from github, it will generate weights in new packing format. The weights at https://huggingface.co/abhinavkulkarni/ are still using old format so we are setting the param to 1 above.


### Converting yourself

1. First generate AWQ int-4 quantized weights following steps in [llm-awq](https://github.com/mit-han-lab/llm-awq)
2. Convert AWQ weights into individual weight binary files using convert_awq_to_bin.py
3. Convert/repack the weight binary files using the weight_repacker.cpp utility.
4. Run the inference (llama2_q4.cu) pointing to the final weight file.

> Note: AWQ scripts doesn't run on Windows. Use Linux or WSL.

Example:

```
python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-chat-metadata.pt
python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-chat-metadata.pt --q_backend real --dump_quant awq_weights/llama2-7b-awq.pt

pip install numpy torch
python3 convert_awq_to_bin.py awq_weights/llama2-7b-awq.pt output
./weight_packer config.json output llama2-7b-awq-q4.bin 0
```


## Sample output and performance

We get ~200 tokens per second with RTX 4090 for 7b paramater models:

```
llama2_q4.exe llama2-7b-awq-q4.bin -n 256 -i "write an essay about GPUs"

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
