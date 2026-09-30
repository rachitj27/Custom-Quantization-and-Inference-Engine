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

# Language-guarded: nvcc rejects -Wall outright, and add_compile_options is
# directory-scoped, so an unguarded list reaches every .cu in the tree.
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

if(ENGINE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # T4 on Colab. Settable without a device attached, so the local WSL box
  # compiles and links the same binary it cannot run.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()

  target_sources(custom_engine PRIVATE
    cuda/device_probe.cu
  )
  target_compile_definitions(custom_engine PRIVATE ENGINE_HAS_CUDA=1)

  # --fmad=false keeps a*b+c uncontracted so device products match the host,
  # which builds without -march/-mfma. Never --use_fast_math here: the FP32
  # epilogue has to stay reproducible against the scalar kernel.
  target_compile_options(custom_engine PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;-lineinfo;-Xcompiler=-Wall,-Wextra>")

  set_target_properties(custom_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
