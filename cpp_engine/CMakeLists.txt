cmake_minimum_required(VERSION 3.18)
project(custom_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Lets the editor resolve includes and flags exactly as the build does.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENGINE_CUDA "Build the CUDA GEMM convolution backend" OFF)

# language-guarded: nvcc rejects -Wall, and this is directory-scoped
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-O2>")

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(custom_engine
  main.cpp
  tensor.cpp
  model.cpp
  ops.cpp
  image_io.cpp
)

target_include_directories(custom_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(custom_engine PRIVATE nlohmann_json::nlohmann_json)

# stb is vendored third-party code; don't let its warnings drown out ours.
set_source_files_properties(image_io.cpp PROPERTIES COMPILE_FLAGS
  "-Wno-unused-but-set-variable -Wno-missing-field-initializers")

if(NOT ENGINE_CUDA)
  target_compile_definitions(custom_engine PRIVATE ENGINE_HAS_CUDA=0)
endif()

if(ENGINE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # T4. Settable with no device attached, so a GPU-less box can still build.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()

  target_sources(custom_engine PRIVATE
    cuda/device_probe.cu
    cuda/gemm.cu
    cuda/igemm.cu
    cuda/conv_cuda.cu
  )
  target_compile_definitions(custom_engine PRIVATE ENGINE_HAS_CUDA=1)

  # --fmad=false so device products match the host, which has no -mfma.
  # never --use_fast_math: the FP32 epilogue must stay reproducible.
  target_compile_options(custom_engine PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;-lineinfo;-Xcompiler=-Wall,-Wextra>")

  # kernels on their own, no model or engine
  add_executable(gemm_selftest
    cuda/gemm_selftest.cu
    cuda/gemm.cu
    cuda/igemm.cu
    cuda/gemm_reference.cu
  )
  target_include_directories(gemm_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(gemm_selftest PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;-lineinfo>")
endif()
