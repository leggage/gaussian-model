#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cooperative_groups.h>
#include "config.h"
#include "forward.h"
namespace cg = cooperative_groups;
namespace {
constexpr float kPi = 3.14159265358979323846f;
}




__global__ void __launch_bounds__(BlockSize_x*BlockSize_y) 
render_forward(
  const int image_height,
  const int image_width,
  const float* __restrict__ conics2d,
  const float* __restrict__ means2d,
  const float* __restrict__ weights,
  const float* __restrict__ colors,
  const float* __restrict__ opacity,
  const uint2* __restrict__ ranges,
  const uint64_t* __restrict__ point_list,
  float* __restrict__ Tfinal,
  uint32_t* __restrict__ ncontributor,
  float* __restrict__ image) {
const int image_x = blockIdx.x * blockDim.x + threadIdx.x;
const int image_y = blockIdx.y * blockDim.y + threadIdx.y;
bool pixel_valid =
image_x < image_width && image_y < image_height;
bool Done = !pixel_valid;

auto idx = cg::this_grid().thread_rank();
//data fetching
int gridid = blockIdx.y * gridDim.x + blockIdx.x;
auto threadid = threadIdx.y*blockDim.x+threadIdx.x;
int rangex = ranges[gridid].x;
int rangey = ranges[gridid].y;
int gaussian_num = rangey-rangex;
int todo = rangey-rangex;
int rounds = (gaussian_num+255)/256;
//__shared__ int collected_id[gaussian_num];
__shared__ float collected_conic[BlockSize_x*BlockSize_y*3];
__shared__ float collected_weight[BlockSize_x*BlockSize_y];
__shared__ float collected_opacity[BlockSize_x*BlockSize_y];
__shared__ float collected_cen2p[BlockSize_x*BlockSize_y*2];
__shared__ float collected_color[BlockSize_x*BlockSize_y*3];

float out_r = 0.0f;
float out_g = 0.0f;
float out_b = 0.0f;
float transmittance = 1.0f;
uint32_t contributor=0;
uint32_t last_contributor=0;
for(int i=0;i<rounds;i++,todo-=BlockSize_x*BlockSize_y)
{
  int collected_id = threadid; 
  int gaussian_offset = i*(BlockSize_x*BlockSize_y)+threadid; 
  bool valid = gaussian_offset < gaussian_num;
  if(valid)
  {
    int off = rangex+gaussian_offset;      
    int gaussianid  = point_list[off];
    collected_conic[collected_id*3+0] = conics2d[gaussianid*3+0];
    collected_conic[collected_id*3+1] = conics2d[gaussianid*3+1];
    collected_conic[collected_id*3+2] = conics2d[gaussianid*3+2];
    collected_weight[collected_id]    = weights[gaussianid];
    collected_opacity[collected_id]   = opacity[gaussianid];
    collected_cen2p[collected_id*2+0] = means2d[gaussianid*2+0];
    collected_cen2p[collected_id*2+1] = means2d[gaussianid*2+1];
    collected_color[collected_id*3+0] = colors[gaussianid*3+0];
    collected_color[collected_id*3+1] = colors[gaussianid*3+1];
    collected_color[collected_id*3+2] = colors[gaussianid*3+2];
  }
  __syncthreads();

//alpha-blending for a batch of 256 gaussians
  if(pixel_valid)
  {
    for (int j = 0; !Done && j < min(256,todo); j++) {
      contributor++;
      const float dx = image_x - collected_cen2p[2 * j];
      const float dy = image_y - collected_cen2p[2 * j + 1];
    
      const float mahalanobis =
          dx * dx * collected_conic[j * 3] +
          dx * dy * 2 *collected_conic[j * 3+1] +
          dy * dy * collected_conic[j * 3 + 2];
      float alpha = collected_opacity[j]*collected_weight[j] * expf(-0.5f * mahalanobis);
      alpha = fminf(fmaxf(alpha, 0.0f), 0.99f);
      if(alpha<0.001){
        continue;
      }
      if(transmittance < 0.0001)
      {
        Done = true;
        continue;
      }      
      out_r += collected_color[j * 3 + 0] * transmittance * alpha;
      out_g += collected_color[j * 3 + 1] * transmittance * alpha;
      out_b += collected_color[j * 3 + 2] * transmittance * alpha;
    
      transmittance *= (1.0f - alpha);
      last_contributor = contributor;
    }
    ncontributor[idx] = last_contributor;
    Tfinal[idx] = transmittance;
    const int offset = (image_y * image_width + image_x) * 3;
    image[offset + 0] = out_r;
    image[offset + 1] = out_g;
    image[offset + 2] = out_b;       
  }
  __syncthreads();

}


}




__global__ void preprocess_forward(
  const float* __restrict__ q,
  const float* __restrict__ s,
  const float* __restrict__ mean3d,
  const float* __restrict__ viewmatrix,
  const float* __restrict__ projmatrix,
  const int N,
  const int height,
  const int width,
  const float fx,
  const float fy,
  const dim3 grid,
  float* __restrict__ conics2d,
  float* __restrict__ depths,
  float* __restrict__ radii,
  uint32_t* __restrict__ tiles_touched, 
  float* __restrict__ cov3w,
  float* __restrict__ means2d,
  float* __restrict__ weights  

){
  auto idx = blockIdx.x * blockDim.x+ threadIdx.x;
  if(idx>=N){return;}

  //
  radii[idx]=0;
  tiles_touched[idx]=0;

  //PREPARE MEANS2D and depths
  float3 point_ori = {mean3d[idx*3],mean3d[idx*3+1],mean3d[idx*3+2]};
  float3 p_cam = transformPoint4x3_Mrow(point_ori,viewmatrix);  

  if(!checkinfrustum(point_ori,viewmatrix,p_cam)){return;}

  float4 point_cla = transformPoint4x4_Mrow(point_ori,projmatrix);
  float p_hom = 1/(point_cla.w+0.00000001);
  float3 p_proj ={point_cla.x*p_hom,point_cla.y*p_hom,point_cla.z*p_hom};
  means2d[idx*2] = ndc2pixel(p_proj.x,width);
  means2d[idx*2+1] = ndc2pixel(p_proj.y,height);
  depths[idx] = p_cam.z;
  
  

  //PREPARE CONICS2D/cov3w/radii/tiles_touched
  glm::mat3 W = glm::mat3(viewmatrix[0],viewmatrix[3],viewmatrix[6],
                          viewmatrix[1],viewmatrix[4],viewmatrix[7],
                          viewmatrix[2],viewmatrix[5],viewmatrix[8]);
  glm::mat3 J2p=glm::mat3(fx/p_cam.z,0,0,
                          0,fy/p_cam.z,0,
                        -(p_cam.x*fx)/(p_cam.z*p_cam.z),-(p_cam.y*fy)/(p_cam.z*p_cam.z),0);
  glm::mat3 M=J2p*W;                       
    //obtain R,S to compose VRK(cov3w)
  float4 q4 = {q[idx*4],q[idx*4+1],q[idx*4+2],q[idx*4+3]};
  float R[9] = {};
  computeRotation(q4,R);

  glm::mat3 RS = glm::mat3(s[idx*3+0]*R[0],s[idx*3+0]*R[3],s[idx*3+0]*R[6],
                              s[idx*3+1]*R[1],s[idx*3+1]*R[4],s[idx*3+1]*R[7],
                              s[idx*3+2]*R[2],s[idx*3+2]*R[5],s[idx*3+2]*R[8]);
  

  glm::mat3 VRK=RS*glm::transpose(RS);
  glm::mat3 cov2p = M*VRK*glm::transpose(M);

  float3 cov2d_p = {cov2p[0][0],cov2p[0][1],cov2p[1][1]};

  float det_inv = 1.0f/(cov2d_p.x*cov2d_p.z-cov2d_p.y*cov2d_p.y);
  conics2d[idx*3]=det_inv*cov2d_p.z;
  conics2d[idx*3+1]=-det_inv*cov2d_p.y;
  conics2d[idx*3+2]=det_inv*cov2d_p.x;

  cov3w[idx*6+0]=VRK[0][0];
  cov3w[idx*6+1]=VRK[1][0];
  cov3w[idx*6+2]=VRK[2][0];
  cov3w[idx*6+3]=VRK[1][1];
  cov3w[idx*6+4]=VRK[2][1];
  cov3w[idx*6+5]=VRK[2][2];

    //obtain radii,tiles_touched
  float mid = (cov2d_p.x+cov2d_p.z)/2;
  float lambda1 = mid+sqrt(mid*mid-(cov2d_p.x*cov2d_p.z-cov2d_p.y*cov2d_p.y));
  float lambda2 = mid-sqrt(mid*mid-(cov2d_p.x*cov2d_p.z-cov2d_p.y*cov2d_p.y));
  radii[idx] = max(0.1,3*sqrt(max(lambda1,lambda2)));

  uint2 rctmin;
  uint2 rctmax;
  getrect(means2d+idx*2,radii[idx],rctmin,rctmax,grid);
  if((rctmax.x-rctmin.x)*(rctmax.y-rctmin.y)==0){return;}
  tiles_touched[idx] = (rctmax.x-rctmin.x)*(rctmax.y-rctmin.y);



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
  glm::mat2 cov2r = glm::mat2(cov2dray[0][0],cov2dray[0][1],cov2dray[1][0],cov2dray[1][1]);

  float det1 = glm::determinant(cov3dray);
  float det2 = glm::determinant(cov2r);
  weights[idx] = sqrt(2*kPi*det1/det2);

}






void Forward::preprocess(
  const float* __restrict__ q,
  const float* __restrict__ s,
  const float* __restrict__ mean3d,
  const float* __restrict__ viewmatrix,
  const float* __restrict__ projmatrix,
  const int N,
  const int height,
  const int width,
  const float fx,
  const float fy,
  const dim3 grid,
  float* __restrict__ conics2d,
  float* __restrict__ depths,
  float* __restrict__ radii,
  uint32_t* __restrict__ tiles_touched, 
  float* __restrict__ cov3w,
  float* __restrict__ means2d,
  float* __restrict__ weights  
)
{
  preprocess_forward<<<(N+255)/256,256>>>(
    q,
    s,
    mean3d,
    viewmatrix,
    projmatrix,
    N,
    height,
    width,
    fx,
    fy,
    grid,
    conics2d,
    depths,
    radii,
    tiles_touched,
    cov3w,
    means2d,
    weights
  );
}

void Forward::render(
  const int image_height,
  const int image_width,
  const float* __restrict__ conics2d,
  const float* __restrict__ means2d,
  const float* __restrict__ weights,
  const float* __restrict__ colors,
  const float* __restrict__ opacity,
  const uint2* __restrict__ ranges,
  const uint64_t* __restrict__ point_list,
  const dim3 grid,
  const dim3 block,  
  float* __restrict__ Tfinal,
  uint32_t* __restrict__ ncontributor,
  float* __restrict__ image

)
{
  render_forward<<<grid,block>>>(
    image_height,
    image_width,
    conics2d,
    means2d,
    weights,
    colors,
    opacity,
    ranges,
    point_list,
    Tfinal,
    ncontributor,
    image
  );
}




// torch::Tensor render_forward_v2(
//   torch::Tensor colors,
//   torch::Tensor opacity,
//   torch::Tensor mean3d,
//   torch::Tensor q,      //顺序依次为qr\qi\qj\qk
//   torch::Tensor s,
//   torch::Tensor viewmatrix,
//   torch::Tensor projmatrix,
//   int64_t height,
//   int64_t width,
//   float fx,
//   float fy) {
// const c10::cuda::CUDAGuard device_guard(colors.device());

// int N = static_cast<int>(colors.size(0));    //gaussian num

// auto options   = colors.options();
// auto conics2d  = torch::zeros({N,3},options);
// auto means2d   = torch::zeros({N,2},options); 
// auto weights   = torch::zeros({N,1},options); 
// auto cov3w     = torch::zeros({N,6},options); 


// preprocess_params_forward<<<(N+255)/256,256>>>(
//   const float* __restrict__ q,
//   const float* __restrict__ s,
//   const float* __restrict__ mean3d,
//   const float* __restrict__ viewmatrix,
//   const float* __restrict__ projmatrix,
//   const int N,
//   const int height,
//   const int width,
//   const float fx,
//   const float fy,
//   const dim3 grid,
//   float* __restrict__ conics2d,
//   float* __restrict__ depths,
//   float* __restrict__ radii,
//   uint32_t* __restrict__ tiles_touched, 
//   float* __restrict__ cov3w,
//   float* __restrict__ means2d,
//   float* __restrict__ weights  
// );

// auto image = torch::empty({height, width, 3}, colors.options());

// const dim3 block(16, 16);
// const dim3 grid(
//     (static_cast<unsigned int>(width) + block.x - 1) / block.x,
//     (static_cast<unsigned int>(height) + block.y - 1) / block.y);

// // rasterization_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
// rasterization_kernel_v2<<<grid, block>>>(
//     static_cast<int>(N),
//     static_cast<int>(height),
//     static_cast<int>(width),
//     conics2d.contiguous().data_ptr<float>(),
//     means2d.contiguous().data_ptr<float>(),
//     weights.contiguous().data_ptr<float>(),
//     colors.contiguous().data_ptr<float>(),
//     opacity.contiguous().data_ptr<float>(),
//     image.contiguous().data_ptr<float>());

// C10_CUDA_KERNEL_LAUNCH_CHECK();
// return image;
// }


// //this need bias computed in python
// torch::Tensor render_forward_cuda(
//   torch::Tensor colors,
//   torch::Tensor cens,
//   torch::Tensor inv_covs,
//   torch::Tensor weights,
//   torch::Tensor inv_k,
//   int64_t height,
//   int64_t width) {
// const c10::cuda::CUDAGuard device_guard(colors.device());
// auto image = torch::empty({height, width, 3}, colors.options());

// const dim3 block(16, 16);
// const dim3 grid(
//     (static_cast<unsigned int>(width) + block.x - 1) / block.x,
//     (static_cast<unsigned int>(height) + block.y - 1) / block.y);

// // rasterization_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
// rasterization_kernel<<<grid, block>>>(
//     static_cast<int>(colors.size(0)),
//     static_cast<int>(height),
//     static_cast<int>(width),
//     inv_k.contiguous().data_ptr<float>(),
//     weights.contiguous().data_ptr<float>(),
//     cens.contiguous().data_ptr<float>(),
//     inv_covs.contiguous().data_ptr<float>(),
//     colors.contiguous().data_ptr<float>(),
//     image.contiguous().data_ptr<float>());

// C10_CUDA_KERNEL_LAUNCH_CHECK();
// return image;
// }
