#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
__global__ void rasterization_kernel(
    const int gaussian_num,
    const int image_height,
    const int image_width,
    const float* __restrict__ inv_k,
    const float* __restrict__ weight,
    const float* __restrict__ cen,
    const float* __restrict__ cov,
    const float* __restrict__ colors,
    float* __restrict__ image) {
  const int image_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int image_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (image_x >= image_width || image_y >= image_height) {
    return;
  }

  const float ray_x = image_x * inv_k[0] + image_y * inv_k[1] + inv_k[2];
  const float ray_y = image_x * inv_k[3] + image_y * inv_k[4] + inv_k[5];

  float out_r = 0.0f;
  float out_g = 0.0f;
  float out_b = 0.0f;
  float transmittance = 1.0f;

  for (int i = 0; i < gaussian_num; ++i) {
    const float dx = ray_x - cen[2 * i];
    const float dy = ray_y - cen[2 * i + 1];

    const float mahalanobis =
        dx * dx * cov[i * 4] +
        dx * dy * (cov[i * 4 + 1] + cov[i * 4 + 2]) +
        dy * dy * cov[i * 4 + 3];
    float alpha = weight[i] * expf(-0.5f * mahalanobis);
    alpha = fminf(fmaxf(alpha, 0.0f), 0.99f);

    out_r += colors[i * 3 + 0] * transmittance * alpha;
    out_g += colors[i * 3 + 1] * transmittance * alpha;
    out_b += colors[i * 3 + 2] * transmittance * alpha;

    transmittance *= (1.0f - alpha);
  }

  const int offset = (image_y * image_width + image_x) * 3;
  image[offset + 0] = out_r;
  image[offset + 1] = out_g;
  image[offset + 2] = out_b;
}

torch::Tensor render_forward_cuda(
  torch::Tensor colors,
  torch::Tensor cens,
  torch::Tensor inv_covs,
  torch::Tensor weights,
  torch::Tensor inv_k,
  int64_t height,
  int64_t width) {
const c10::cuda::CUDAGuard device_guard(colors.device());
auto image = torch::empty({height, width, 3}, colors.options());

const dim3 block(16, 16);
const dim3 grid(
    (static_cast<unsigned int>(width) + block.x - 1) / block.x,
    (static_cast<unsigned int>(height) + block.y - 1) / block.y);

// rasterization_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
rasterization_kernel<<<grid, block>>>(
    static_cast<int>(colors.size(0)),
    static_cast<int>(height),
    static_cast<int>(width),
    inv_k.contiguous().data_ptr<float>(),
    weights.contiguous().data_ptr<float>(),
    cens.contiguous().data_ptr<float>(),
    inv_covs.contiguous().data_ptr<float>(),
    colors.contiguous().data_ptr<float>(),
    image.contiguous().data_ptr<float>());

C10_CUDA_KERNEL_LAUNCH_CHECK();
return image;
}