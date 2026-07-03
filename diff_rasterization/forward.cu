#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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


__global__ void rasterization_kernel_v2(
  const int gaussian_num,
  const int image_height,
  const int image_width,
  const float* __restrict__ conics2d,
  const float* __restrict__ means2d,
  const float* __restrict__ weights
  const float* __restrict__ colors,
  const float* __restrict__ opacity,
  float* __restrict__ image) {
const int image_x = blockIdx.x * blockDim.x + threadIdx.x;
const int image_y = blockIdx.y * blockDim.y + threadIdx.y;
if (image_x >= image_width || image_y >= image_height) {
  return;
}

float out_r = 0.0f;
float out_g = 0.0f;
float out_b = 0.0f;
float transmittance = 1.0f;

for (int i = 0; i < gaussian_num; ++i) {
  const float dx = image_x - means2d[2 * i];
  const float dy = image_y - means2d[2 * i + 1];

  const float mahalanobis =
      dx * dx * conics2d[i * 3] +
      dx * dy * 2 *conics2d[i * 3+1] +
      dy * dy * cov[i * 3 + 2];
  float alpha = opacity[i]*weight[i] * expf(-0.5f * mahalanobis);
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


__gloabl__  void preprocess_render(
  const float* __restrict__ cov3d,
  const float* __restrict__ mean3d,
  const float* __restrict__ viewmatrix,
  const float* __restrict__ projmatrix,
  const int height,
  const int width,
  const float fx,
  const float fy,
  float* __restrict__ conics2d,
  float* __restrict__ means2d,
  float* __restrict__ weights
){
  auto idx = blockIdx.x * blockDim.x+ threadIdx.x;
  if(idx>=gaussian){return;}

  //PREPARE MEANS2D
  float3 point_ori = {mean3d[idx*3],mean3d[idx*3+1],mean3d[idx*3+2]};
  float3 p_cam = transformPoint4x3(point_ori,viewmatrix);
  float4 point_cla = transformPoint4x4(point_ori,projmatrix);
  float p_hom = 1/(point_cla.w+0.00000001);
  float3 p_proj ={point_cla.x*p_hom,point_cla.y*p_hom,point_cla.z*p_hom};
  means2d[idx*2] = ndc2pixel(p_proj.x,width);
  means2d[idx*2+1] = ndc2pixel(p_proj.y,height);

  //PREPARE CONICS2D
  glm::mat3 W= glm::mat3(viewmatrix[0],viewmatrix[1],viewmatrix[2],
                        viewmatrix[3],viewmatrix[4],viewmatrix[5],
                        viewmatrix[6],viewmatrix[7],viewmatrix[8]);
  glm::mat3 J=glm::mat3(fx/p_cam.z,0,0,
                        0,fy/p_cam.z,0,
                        -(p_cam.x*fx)/(p_cam.z*p_cam.z),-(p_cam.y*fy)/(p_cam.z*p_cam.z),0);

  glm::mat3 M=J*W;
  glm::mat3 VRK=glm::mat3(cov3d[i*6],cov3d[i*6+1],cov3d[i*6+2],
                            cov3d[i*6+1],cov3d[i*6+3],cov3d[i*6+4],
                            cov3d[i*6+2],cov3d[i*6+4],cov3d[i*6+5]);
  glm::mat3 cov2d_m = M*VRK*glm::transpose(M);

  float3 cov2d_p = {cov2d_m[0][0],cov2d_m[0][1],cov2d_m[1][1]};
  float det_inv = cov2d_p.x*cov2d_p.z-cov2d_p.y*cov2d_p.y;
  conics2d[idx*3]=det_inv*cov2d_p.z;
  conics2d[idx*3+1]=-det_inv*cov2d_p.y;
  conics2d[idx*3+2]=det_inv*cov2d_p.x;

  //prepare weights(integration bias)
  float l = sqrt(p_cam.z*p_cam.z+p_cam.y*p_cam.y+p_cam.x*p_cam.x);
  glm::mat3 Jr3=glm::mat3(1/p_cam.z,0,p_cam.x/l,
                           0,1/p_cam.z,p_cam.y/l,
                           -(p_cam.x)/(p_cam.z*p_cam.z),-(p_cam.y)/(p_cam.z*p_cam.z),p_cam.z/l);
  glm::mat3 Jr2=glm::mat3(1/p_cam.z,0,0,
                          0,1/p_cam.z,0,
                          -(p_cam.x)/(p_cam.z*p_cam.z),-(p_cam.y)/(p_cam.z*p_cam.z),0); 
  glm::mat3 cov3dray = Jr3*W*VRK*glm::transpose(Jr3*W);
  glm::mat3 cov2dray = Jr2*W*VRK*glm::transpose(Jr2*W);
  glm::mat2 cov2d = glm::mat2(cov2dray[0][0],cov2dray[0][1],cov2dray[1][0],cov2dray[1][1]);

  float det1 = glm::determinant(cov3dray);
  float det2 = glm::determinant(cov2d);
  weights[idx] = sqrt(2*pi*det1/det2);

}



torch::Tensor render(
  torch::Tensor colors,
  torch::Tensor opacity,
  torch::Tensor mean3d,
  torch::Tensor cov3d,      //上半区域，6个元素
  torch::Tensor viewmatrix,
  torch::Tensor projmatrix,
  int64_t height,
  int64_t width
  float fx,
  float fy,) {
const c10::cuda::CUDAGuard device_guard(colors.device());

int N = static_cast<int>(colors.size(0));    //gaussian num
float conics2d[N*3]=0;
float means2d[N*2]=0;
float weights[N]=0;
preprocess_render<<<(N+255)/256,256>>>(
  cov3d.contiguous().data_ptr<float>(),
  mean3d.contiguous().data_ptr<float>(),
  viewmatrix.contiguous().data_ptr<float>(),
  projmatrix.contiguous().data_ptr<float>(),
  height,
  width,
  fx,
  fy,
  conics2d,
  means2d,
  weights
);

auto image = torch::empty({height, width, 3}, colors.options());

const dim3 block(16, 16);
const dim3 grid(
    (static_cast<unsigned int>(width) + block.x - 1) / block.x,
    (static_cast<unsigned int>(height) + block.y - 1) / block.y);

// rasterization_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
rasterization_kernel_v2<<<grid, block>>>(
    static_cast<int>(colors.size(0)),
    static_cast<int>(height),
    static_cast<int>(width),
    conics2d,
    means2d,
    weights,
    colors.contiguous().data_ptr<float>(),
    opacity.contiguous().data_ptr<float>(),
    image.contiguous().data_ptr<float>());

C10_CUDA_KERNEL_LAUNCH_CHECK();
return image;
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