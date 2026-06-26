#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ void grad_kernel(
    float* dl_dpixel,
    float* dl_dcolor,
    int ch,
    int gaussian_num,
    int image_height,
    int image_width,
    float *inv_k,
    float *weight,
    float*cen,
    float*cov)
{
    int image_x = blockIdx.x*blockDim.x+threadIdx.x;
    int image_y = blockIdx.y*blockDim.y+threadIdx.y;
    if (image_x >= image_width || image_y >= image_height) return;
    int image_index = image_y * image_width + image_x;

    float ray_x = image_x*inv_k[0]+image_y*inv_k[1]+inv_k[2];
    float ray_y = image_x*inv_k[3]+image_y*inv_k[4]+inv_k[5];

    float T =1;
    for(int i=0;i<gaussian_num;++i){
        float dx = ray_x - cen[2*i];
        float dy = ray_y - cen[2*i+1];

        float mahlo = dx*dx*cov[i*4]+dx*dy*(cov[i*4+1]+cov[i*4+2])+dy*dy*cov[i*4+3];
        float alpha = weight[i]*expf(-0.5*mahlo);

        alpha = fminf(fmaxf(alpha, 0.0f), 0.99f);
        float dpixel_dcolor = T*alpha;

        for(int j=0;j<ch;j++)
        {
            float dl_dp = dl_dpixel[image_index * 3 + j];
            atomicAdd(&dl_dcolor[i * 3 + j], dl_dp * dpixel_dcolor);

        }
        T = T*(1-alpha);
        if (T < 1e-4f) break;
    }
}

torch::Tensor backward_cuda(
    torch::Tensor dl_dpixel,
    torch::Tensor inv_k,
    torch::Tensor weight,
    torch::Tensor cens,
    torch::Tensor covs,
    int64_t chans,
    int64_t gaussian_nums,
    int64_t width,
    int64_t height
){
    auto dl_dcolors = torch::empty({gaussian_nums,chans});
    const dim3 block(16,16);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1) / block.y);  
    grad_kernel<<<grid, block>>>(
        dl_dpixel.contiguous().data_ptr<float>(),
        dl_dcolors.contiguous().data_ptr<float>(),
        chans,
        gaussian_nums,
        height,
        width,
        inv_k.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        cens.contiguous().data_ptr<float>(),
        covs.contiguous().data_ptr<float>());
    return dl_dcolors;
}

