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
std::tuple<int,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
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
  int num_rendered = 0;
  num_rendered = cudaRasterizer::forward(
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
  return std::make_tuple(num_rendered,image,image_buffer,geometry_buffer,binning_buffer);
  }

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
backward_grad(
  int num_rendered,
  int64_t height,
  int64_t width,
  float fx,
  float fy,
  torch::Tensor viewmatrix,
  torch::Tensor projmatrix,

  torch::Tensor image_buffer,
  torch::Tensor binning_buffer,
  torch::Tensor geometry_buffer,

  torch::Tensor dl_dpixel,
  torch::Tensor colors,
  torch::Tensor opacity,
  torch::Tensor cen3w,
  torch::Tensor q,
  torch::Tensor s
)
{
  int N = static_cast<int>(colors.size(0));    //gaussian num
  auto options = colors.options().dtype(torch::kByte);
  dim3 grid((width+BlockSize_x-1)/BlockSize_x,(height+BlockSize_y-1)/BlockSize_y,1);
  dim3 block(BlockSize_x,BlockSize_y,1);   

  auto grad_buffer    = torch::empty({0},options);
  std::function<char*(size_t N)> gradFunc = resizefunctional(grad_buffer);

  torch::Tensor dl_dcolor    = torch::zeros({N, 3}, colors.options());
  torch::Tensor dl_dopacity  = torch::zeros({N, 1}, colors.options());
  torch::Tensor dl_dq        = torch::zeros({N, 4}, colors.options());
  torch::Tensor dl_ds        = torch::zeros({N, 3}, colors.options());
  torch::Tensor dl_dcen3w    = torch::zeros({N, 3}, colors.options());
  
  cudaRasterizer::backward(
    N,
    num_rendered, 
    grid,
    block,

    height,
    width,
    fx,
    fy,
    viewmatrix.contiguous().data_ptr<float>(),
    projmatrix.contiguous().data_ptr<float>(),

    reinterpret_cast<char*>(image_buffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(binning_buffer.contiguous().data_ptr()),
    reinterpret_cast<char*>(geometry_buffer.contiguous().data_ptr()),  

    dl_dpixel.data_ptr<float>(),
    colors.data_ptr<float>(),
    opacity.data_ptr<float>(), 
    cen3w.data_ptr<float>(),
    q.data_ptr<float>(),
    s.data_ptr<float>(),

    dl_dcolor.data_ptr<float>(),
    dl_dopacity.data_ptr<float>(),
    dl_dq.data_ptr<float>(),
    dl_ds.data_ptr<float>(),
    dl_dcen3w.data_ptr<float>(),
    gradFunc 
);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(dl_dcolor , dl_dopacity , dl_dq , dl_ds , dl_dcen3w);
}