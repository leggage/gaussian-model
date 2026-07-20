#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include  "rasterize_impl.h"
#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include "config.h"
std::function<char*(size_t N)> resizefunctional(torch::Tensor& t)
{
    auto lambda = [&t](size_t N){
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

//rendered_image,image-state,geometry-state,binning-state,
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
rasterize_gaussian(
    torch::Tensor colors,
    torch::Tensor opacity,
    torch::Tensor mean3w,
    torch::Tensor q,      //顺序依次为qr\qi\qj\qk
    torch::Tensor s,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    int64_t height,
    int64_t width,
    float fx,
    float fy) {
  const c10::cuda::CUDAGuard device_guard(colors.device());
  
  int N = static_cast<int>(colors.size(0));    //gaussian num
  auto options = colors.options().dtype(torch::kByte);

  auto image_buffer    = torch::empty({0},options);
  auto geometry_buffer = torch::empty({0},options);
  auto binning_buffer  = torch::empty({0},options);
  auto image           = torch::empty({height, width, 3}, colors.options());

  std::function<char*(size_t N)> imageFunc = resizefunctional(image_buffer);
  std::function<char*(size_t N)> geomFunc  = resizefunctional(geometry_buffer);
  std::function<char*(size_t N)> binnFunc  = resizefunctional(binning_buffer);
  
  dim3 grid((width+BlockSize_x-1)/BlockSize_x,(height+BlockSize_y-1)/BlockSize_y,1);
  dim3 block(BlockSize_x,BlockSize_y,1);

  cudaRasterizer::forward(
    N,
    height,
    width,
    fx,
    fy,    
    viewmatrix.contiguous().data_ptr<float>(),
    projmatrix.contiguous().data_ptr<float>(),    
    q.contiguous().data_ptr<float>(),
    s.contiguous().data_ptr<float>(),
    mean3w.contiguous().data_ptr<float>(),
    opacity.contiguous().data_ptr<float>(),
    colors.contiguous().data_ptr<float>(),
    image.contiguous().data_ptr<float>(),
    grid,
    block,
    imageFunc,
    geomFunc,
    binnFunc    
  );
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(image,image_buffer,geometry_buffer,binning_buffer);
  }
