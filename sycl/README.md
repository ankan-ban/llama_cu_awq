# llama_sycl_awq

llama_sycl_awq is the SCYL version of [llama_cu_awq](https://github.com/ankan-ban/llama_cu_awq).

# Requirements
- icpx compiler which is included in oneAPI Base Toolkit available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
- model checkpoint needs to prepared by following steps [here](https://github.com/ankan-ban/llama_cu_awq?tab=readme-ov-file#using-huggingface). Same page also has the tokenizer.bin file.

# Build Instructions
First, source icpx compiler. Then,

```
make
```
This creates the executable `runsycl`

# Run Instructions

```
./runsycl <converted check_point> -z <tokenizer_path> -m generate -t 0.0 -n <num_tot_tokens> -i <prompt>
```
