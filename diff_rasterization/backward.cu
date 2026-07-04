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

__global__ void backward(
  const int gaussian_num,
  const int image_height,
  const int image_width,
  const float* __restrict__ dl_dpixel; //height*width*chan
  const float* __restrict__ conics2d,
  const float* __restrict__ means2d,
  const float* __restrict__ weights
  const float* __restrict__ colors,
  const float* __restrict__ opacity,
  const float* __restrict__ Tfinal,
  float* __restrict__ dl_dcolor;
  float* __restrict__ dl_dopacity;
  float* __restrict__ dl_dconics2d,
  float* __restrict__ dl_dcovr3d,
  float* __restrict__ dl_dcovr2d,
  float* __restrict__ dl_dcenp2d,

  float* __restrict__ image)
{
    const int image_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int image_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (image_x >= image_width || image_y >= image_height) {
    return;
    }
    const int Idx = image_y*image_width+image_x;
//根据公式dchanch_dalphai = Ti(ci[ch]-cbehind[ch])
    float T = Tfinal;
    float cbehind[3] = 0;
    float dchan_dalpha[3*gaussian_num]=0;

    //用于累加梯度的中间变量
    float dalpha_dconics[3]=0; 
    float dalpha_dcov3r[6]=0;
    float dalpha_dcov2r[3]=0;
    for(int i=gaussian_num-1;i>=0;i--)
    {
        //求当前gaussian的透射率 Ti      
        float dx = image_x-means2d[i*2];
        float dy = image_y-means2d[i*2+1];
        float maloh = -0.5*(dx*dx*conics2d[i*3]+
                            2*dx*dy*conics2d[i*3+1]+
                            dy*dy*conics2d[i*3+2]);
        float G =expf(maloh);
        float alpha = opacity[i]*weights[i]*G;
        T = T/(1-alpha);
        //dl_dpixel[ch]代表dl_dchanch
        //求cbehind,由于color是rgb三通道，所以最后叠加梯度时要三通道；计算dchan_dalpha
        for(int ch=0;ch<3;ch++)
        {
            float color = colors[i*3+ch];
            //dchan_dalpha存储方式是通道优先，先通道一中所有gaussian再通道2...
            dchan_dalpha[ch*gaussian_num+i] = T(color-cbehind[ch]);
            cbehind[ch]=cbehind[ch]*(1-alpha)+color*alpha;
        }

        float dalpha_dopacity = weights[i]*G;

        //计算dalpha_dconic,只存上三角区域
        dalpha_dconic[0] = -0.5*alpha*dx*dx;
        dalpha_dconic[1] = -0.5*alpha*dx*dy;
        dalpha_dconic[2] = -0.5*alpha*dy*dy;

        //计算dalpha_dcov3r,dalpha_dcov2r,需要用到矩阵求逆
        glm::mat3 dcov3r = 0.5*alpha*glm::inverse(cov3r);
        glm::mat2 dcov2r = -0.5*alpha*glm::inverse(cov2r);
        dalpha_dcov3r = {dcov3r[0][0],dcov3r[0][1],dcov3r[0][2],dcov3r[1][1],dcov3r[1][2],dcov3r[2][2]};
        dalpha_dcov2r = {dcov2r[0][0],dcov2r[0][1],dcov2r[1][1]};



        float dchan_dcolor = T*alpha;
        //dl_...
        //这里对dl_dpixel的索引方式是建立在rgb存储通道优先的假设上的
        for(int c=0;c<3;c++)
        {
           atomicAdd(&dl_dcolor[i*3+c],dl_dpixel[Idx*3+c]*dchan_dcolor); 
           atomicAdd(&dl_dopacity[i],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dopacity);
           for(int j=0;j<3;j++)
           {
            atomicAdd(&dl_dconics2d[i*3+j],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dconic[j]);
            atomicAdd(&dl_dcovr2d[i*3+j],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dcov2r[j]);
           }
           for(int j=0;j<6;j++)
           {
            atomicAdd(&dl_dcovr3d[i*6+j],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dcov3r[j]);
           }           

        }
        
        


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

