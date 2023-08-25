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

4. Run the inference (llama2_q4.cu) pointing to the final weight file.

## License

MIT
