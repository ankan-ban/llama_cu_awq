## llama2_q4.sycl.cpp

Simple and fast Pure CPP inference on **Intel** **dGPU/iGPU** platforms for 4-bit [AWQ](https://github.com/mit-han-lab/llm-awq) quantized models.
Based on [llama2.c](https://github.com/karpathy/llama2.c)

Verified platforms: **PVC, ARC770, iGPU**(Meteor Lake CPU)    
oneAPI version: 2024.0

## Build

```
# for PVC
$ icpx -fsycl -O2 -ffast-math -fsycl-targets=spir64_gen -Xsycl-target-backend "-device pvc -q" llama2_q4.sycl.cpp -o llama2_awq

# for ARC770 
$ icpx -fsycl -O2 -ffast-math -fsycl-targets=spir64_gen -Xsycl-target-backend "-device dg2-g10 -q" llama2_q4.sycl.cpp -o llama2_awq

# for iGPU
$ icpx -fsycl -O2 -ffast-math llama2_q4.sycl.cpp -o llama2_awq
```


## Get model
```
$ cd ..
```
Refer to README.md to use convert_awq_to_bin.py and weight_packer to get llama2-7b-awq-q4.bin

## Run
```
$ cd ./sycl
$ ./llama2_awq llama2-7b-awq-q4.bin -n 256 -i "write an essay about dogs"
Model params:-
dim: 4096
hidden_dim: 11008
n_heads: 32
n_kv_heads: 32
n_layers: 32
seq_len: 4096
vocab_size: 32000
rope_theta: 10000

Loading Weights... done!

Encoding Prompt... Done!
write an essay about dogs and their importance in our lives.
Dogs are considered to be man's best friend for a reason. They have been a part of human life for thousands of years, providing companionship, protection, and unconditional love. From the moment they are born, dogs are trained to be loyal and obedient, and they quickly become an integral part of our families.

One of the most important aspects of dogs is their ability to provide emotional support. They are known to have a calming effect on people, and can help to reduce stress and anxiety. This is especially true for people who suffer from mental health conditions such as depression and PTSD. Studies have shown that interacting with dogs can lower levels of cortisol, a hormone associated with stress, and increase levels of oxytocin, a hormone associated with feelings of happiness and well-being.

In addition to providing emotional support, dogs are also important in our lives because of their ability to protect us. They have a strong instinct to defend their pack, and will often put themselves in harm's way to protect their family. This makes them excellent guard dogs, and they are often used
```
