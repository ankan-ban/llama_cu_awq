cmake_minimum_required(VERSION 3.10)
project(llama2_q4 LANGUAGES CXX CUDA)
enable_language(CUDA)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

add_executable(weight_packer weight_packer.cpp)
add_executable(llama2_q4 llama2_q4.cu)
