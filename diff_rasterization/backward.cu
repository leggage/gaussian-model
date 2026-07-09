#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#

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

__focreinline__ __device__ void computeRfromq(float4& q, float* R)
{
    float qr = q.x;
    float qi = q.y;
    float qj = q.z;
    float qk = q.w;
    R[0] = 2*(0.5-(qj*qj+qk*qk));
    R[1] = 2*(qi*qj-qr*qk);
    R[2] = 2*(qi*qk+qr*qj);
    R[3] = 2*(qi*qj+qr*qk);
    R[4] = 2*(0.5-(qi*qi+qk*qk));
    R[5] = 2*(qj*qk-qr*qi);
    R[6] = 2*(qi*qk-qr*qj);
    R[7] = 2*(qj*qk+qr*qi);
    R[4] = 2*(0.5-(qi*qi+qj*qj));
}



__global__ void preprocess(
  const int gaussian_num,
  const int fx,
  const int fy,
  const float* __restrict__ viewmatrix,
  const float* __restrict__ projmatrix,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
  const float* __restrict__ cen3w,
  const float* __restrict__ cov3w,
  const float* __restrict__ conics2d,

  const float* __restrict__ q,
  const float* __restrict__ s,

  const float* __restrict__ dl_dconics2d,
  const float* __restrict__ dl_dcovr3d,
  const float* __restrict__ dl_dcovr2d,
  const float* __restrict__ dl_dcenp2d,

  float* __restrict__ dl_dq,
  float* __restrict__ dl_ds,
  float* __restrict__ dL_dcen3w)
  {
    auto idx = blockIdx.x * blockDim.x+ threadIdx.x;
    if (idx>=gaussian_num){return;}
// compute dl_dcov3w  from  dl_dconics2d dl_dcovr3d dl_dcovr2d part
    glm::mat2 conic = glm::mat2(conics2d[idx*3],conics2d[idx*3+1],conics2d[idx*3+1],conics2d[idx*3+2]);
    glm::mat2 dl_dconic = glm::mat2(dl_dconics2[idx*3],dl_dconics2d[idx*3+1],dl_dconics2d[idx*3+1],dl_dconics2d[idx*3+2]);
    glm::mat2 dl_dcov2p = -conic*dl_dconic*conic;

    float3 p3w = {cen3w[idx*3],cen3w[idx*3+1],cen3w[idx*3+2]};
    float3 p3c = transformPoint4*3(p3w,viewmatrix);
    //dl_dcov3w_conic PART
    glm::mat3 J2p = glm::mat3(fx/p3c.z,0,0,
                              0,fy/p3c.z,0,
                              -p3c.x*fx/(p3c.z*p3c.z),-p3c.y*fy/(p3c.z*p3c.z),0);
    glm::mat3 W = glm::mat3(viewmatrix[0],viewmatrix[3],viewmatrix[6],
                            viewmatrix[1],viewmatrix[4],viewmatrix[7],
                            viewmatrix[2],viewmatrix[5],viewmatrix[8])
    glm::mat3 M = J2p*W;

    glm::mat3 dldcov2p = glm::mat3(dl_dcov2p[0][0],dl_dcov2p[1][0],0,
                                   dl_dcov2p[0][1],dl_dcov2p[1][1],0,
                                   0,0,0);
    glm::mat3 dl_dcov3w_conic = glm::transpose(M)*dldcov2p*M;

    //dl_dcov3w_cov3r PART
    float3 len = sqrt(p3c.x*p3c.x+p3c.y*p3c.y+p3c.z*p3c.z);
    glm::mat3 J3r = glm::mat3(1/p3c.z,0,p3c.x/len,
                              0,1/p3c.z,p3c.y/len,
                              -p3c.x/(p3c.z*p3c.z),-p3c.y/(p3c.z*p3c.z),p3c.z/len);
    M = J3r*W;
    glm::mat3 dldcov3r = glm::mat3(dl_dcovr3d[idx*6],dl_dcovr3d[idx*6+1],dl_dcovr3d[idx*6+2],
                                   dl_dcovr3d[idx*6+1],dl_dcovr3d[idx*6+3],dl_dcovr3d[idx*6+4],
                                   dl_dcovr3d[idx*6+2],dl_dcovr3d[idx*6+4],dl_dcovr3d[idx*6+5]);   
    glm::mat3 dl_dcov3w_cov3r = glm::transpose(M)*dldcov3r*M;

    //dl_dcov3w_cov2 PART
    glm::mat3 J2r = glm::mat3(1/p3c.z,0,0,
                              0,1/p3c.z,0,
                              -p3c.x/(p3c.z*p3c.z),-p3c.y/(p3c.z*p3c.z),0);
    M = J2r*W;
    glm::mat3 dldcov2r = glm::mat3(dl_dcovr3d[idx*6],dl_dcovr3d[idx*6+1],0,
                                   dl_dcovr3d[idx*6+1],dl_dcovr3d[idx*6+3],0,
                                   0,0,0);
    glm::mat3 dl_dcov3w_cov2r = glm::transpose(M)*dldcov2r*M;
    //TOTAL dl_dcov3w
    glm::mat3 dl_dcov3w  = dl_dcov3w_conic+dl_dcov3w_cov3r+dl_dcov3w_cov2r;

//obtain dl_dq and dl_ds from  dl_dcov3w
    //obtain dl_dM,M=RS
    float4 q4 = {q[idx*4],q[idx*4+1],q[idx*4+2],q[idx*4+3]};
    float R[9]=0;
    computeRfromq(q4,R);
         // glm::mat3 RT = glm::mat3(R[0],R[3],R[6],R[1],R[4],R[7],R[2],R[5],R[8]);
    glm::mat3 M = glm::mat3(s[idx*3+0]*R[0],s[idx*3+0]*R[3],s[idx*3+0]*R[6],
                            s[idx*3+1]*R[1],s[idx*3+1]*R[4],s[idx*3+1]*R[7],
                            s[idx*3+2]*R[2],s[idx*3+2]*R[5],s[idx*3+2]*R[8]);
    glm::mat3 dl_dM = 2*dl_dcov3w*M;
    
    //obtain dl_ds = dl_dM * dM_ds
    float dl_dsx = dl_dM[0][0]*R[0]+dl_dM[0][1]*R[3]+dl_dM[0][2]*R[6];
    float dl_dsy = dl_dM[1][0]*R[1]+dl_dM[1][1]*R[4]+dl_dM[1][2]*R[7];
    float dl_dsz = dl_dM[2][0]*R[2]+dl_dM[2][1]*R[5]+dl_dM[2][2]*R[8];

    //obtain dl_dq = dl_dM * dM_dq
    float dldm[9] = {dl_dM[0][0],dl_dM[1][0],dl_dM[2][0],
                     dl_dM[0][1],dl_dM[1][1],dl_dM[2][1],
                     dl_dM[0][2],dl_dM[1][2],dl_dM[2][2]};
    float qr = q[idx*4];
    float qi = q[idx*4+1];
    float qj = q[idx*4+2];
    float qk = q[idx*4+3];

    float sx = s[idx*3+0];
    float sy = s[idx*3+1];
    float sz = s[idx*3+2];


    float dl_dqr = 2*(-sy*qk*dldm[1]+sz*qj*dldm[2]+sx*qk*dldm[3]-sz*qi*dldm[5]-sx*qj*dldm[6]+sy*qi*dldm[7]);
    float dl_dqi = 2*(sy*qj*dldm[1]+sz*qk*dldm[2]+sx*qj*dldm[3]-2*sy*qi*dldm[4]-sy*qr*dldm[5]+sx*qk*dldm[6]+sy*qr*dldm[7]-2*sz*qi*dldm[8]);
    float dl_dqj = 2*(-2*sx*qj*dldm[0]+sy*qi*dldm[1]+sz*qr*dldm[2]+sx*qi*dldm[3]+sz*qk*dldm[5]-sx*qr*dldm[6]+sy*qk*dldm[7]-2*sz*qj*dldm[8]);
    float dl_dqk = 2*(-2*sx*qk*dldm[0]-sy*qr*dldm[1]+sz*qi*dldm[2]+sx*qr*dldm[3]-2*sy*qk*dldm[4]+sz*qj*dldm[5]+sx*qi*dldm[6]+sy*qj*dldm[7]);


//compute dl_dcen3w
  //compute dl_dcen3c = dLdJ3R*dJ3r_dcen3c + dLdJ2R*dJ2r_dcen3c + dLdJ2P*dJ2p_dcen3  +dLdcen2p*dcen2p_dcen3c,
    //compute dl_dJ3r,dl_dJ2r,dl_dJ2p 
    glm::mat3 cov3W = glm::mat3(cov3w[idx*6],cov3w[idx*6+1],cov3w[idx*6+2],
                                cov3w[idx*6+1],cov3w[idx*6+3],cov3w[idx*6+4],
                                cov3w[idx*6+2],cov3w[idx*6+4],cov3w[idx*6+5]);
    M = W*cov3W*glm::transpose(W);
    glm::mat3 dl_dJ3r = 2*dldcov3r*J3r*M;
    glm::vec9 dLdJ3R = glm::vec9(dl_dJ3r[0][0],dl_dJ3r[1][0],dl_dJ3r[2][0],
                                 dl_dJ3r[0][1],dl_dJ3r[1][1],dl_dJ3r[2][1],
                                 dl_dJ3r[0][2],dl_dJ3r[1][2],dl_dJ3r[2][2]);
    glm::mat3 dl_dJ2r = 2*dldcov2r*J2r*M;
    glm::vec9 dLdJ2R = glm::vec9(dl_dJ2r[0][0],dl_dJ2r[1][0],dl_dJ2r[2][0],
                                 dl_dJ2r[0][1],dl_dJ2r[1][1],dl_dJ2r[2][1],
                                 dl_dJ2r[0][2],dl_dJ2r[1][2],dl_dJ2r[2][2]);    
    glm::mat3 dl_dJ2p = 2*dldcov2p*J2p*M;
    glm::vec9 dLdJ2P = glm::vec9(dl_dJ2p[0][0],dl_dJ2p[1][0],dl_dJ2p[2][0],
                                 dl_dJ2p[0][1],dl_dJ2p[1][1],dl_dJ2p[2][1],
                                 dl_dJ2p[0][2],dl_dJ2p[1][2],dl_dJ2p[2][2]); 

    //compute dJ3r_dcen3c,dJ2r_dcen3c,dJ2p_dcen3c
    float xc = p3c.x;
    float yc = p3c.y;
    float zc = p3c.z;
    glm::mat3x9 dJ3r_dcen3c = glm::mat3x9(0,0,-1/(zc*zc),0,0,0,(len-xc*xc/len)/(len*len),(-xc*yc/len)/(len*len),(-xc*zc/len)/(len*len),
                                          0,0,0,0,0,-1/(zc*zc),(-xc*yc/len)/(len*len),(len-yc*yc/len)/(len*len),(-yc*zc/len)/(len*len),
                                          -1/(zc*zc),0,2xc/(zc*zc*zc),0,-1/(zc*zc),2yc/(zc*zc*zc),(-xc*zc/len)/(len*len),(-yc*zc/len)/(len*len),(len-zc*zc/len)/(len*len));

    glm::mat3x9 dJ2r_dcen3c = glm::mat3x9(0,0,-1/(zc*zc),0,0,0,0,0,0,
                                          0,0,0,0,0,-1/(zc*zc),0,0,0,
                                          -1/(zc*zc),0,2xc/(zc*zc*zc),0,-1/(zc*zc),2yc/(zc*zc*zc),0,0,0);

    glm::mat3x9 dJ2p_dcen3c = glm::mat3x9(0,0,-1*fx/(zc*zc),0,0,0,0,0,0,
                                          0,0,0,0,0,-1*fy/(zc*zc),0,0,0,
                                          -1*fx/(zc*zc),0,2xc*fx/(zc*zc*zc),0,-1*fy/(zc*zc),2yc*fy/(zc*zc*zc),0,0,0); 
                                
    //obtain matrix version of dl_dcen2p,dcen2p_dcen3c
    glm::vec2 dl_dcen2p = glm::vec2(dl_dcenp2d[idx*2],dl_dcenp2d[idx*2+1]);
    glm::mat3x2 dcen2p_dcen3c = glm::mat3x2(fx/zc,0,
                                            0,fy/zc,
                                            -fx*xc/(zc*zc),-fy*yc/(zc*zc));                                 
    //compute dl_dcen3c
    glm::vec3 dl_dcen3c = dLdJ3R*dJ3r_dcen3c + dLdJ2R*dJ2r_dcen3c + dLdJ2P*dJ2p_dcen3c + dl_dcen2p*dcen2p_dcen3c;

  //obtain dl_dcen3w        
    glm::vec3 dl_dcen3w = glm::transpose(W)*dl_dcen3c;

// transfer data 
    dl_dq[idx*4] = dl_dqr;
    dl_dq[idx*4+1] = dl_dqi;
    dl_dq[idx*4+2] = dl_dqj;
    dl_dq[idx*4+3] = dl_dqk;

    dl_ds[idx*3] = dl_dsx;
    dl_ds[idx*3+1] = dl_dsy;
    dl_ds[idx*3+2] = dl_dsz;

    dL_dcen3w[idx*3] = dl_dcen3w[0];
    dL_dcen3w[idx*3+1] = dl_dcen3w[1];
    dL_dcen3w[idx*3+2] = dl_dcen3w[2];
                                          
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

//链式法则dl_da = for(pixel in all)for(ch in all),atomicAdd(,dl_dpixel[ch]*dchan[ch]_dalpha*dalpha_da)
    //最后链式法则求dl_...要用到的中间变量
    float dalpha_dconics[3]=0; 
    float dalpha_dcov3r[6]=0;
    float dalpha_dcov2r[3]=0;

    float T = Tfinal[Idx];
    float cbehind[3] = 0;
    float dchan_dalpha[3]=0;    
    for(int i=gaussian_num-1;i>=0;i--)
    {
              
        float dx = image_x-means2d[i*2];
        float dy = image_y-means2d[i*2+1];
        float maloh = -0.5*(dx*dx*conics2d[i*3]+
                            2*dx*dy*conics2d[i*3+1]+
                            dy*dy*conics2d[i*3+2]);
        float G =expf(maloh);
        float alpha = opacity[i]*weights[i]*G;  //当前gaussian的不透明度 alphai
        T = T/(1-alpha);                        //当前gaussian的透射率 Ti

    //计算dchan_dalpha=Ti(ci[ch]-cbehind[ch])
        for(int ch=0;ch<3;ch++)
        {
            float color = colors[i*3+ch];
            //dchan_dalpha存储方式是通道优先，先通道一中所有gaussian再通道2...
            dchan_dalpha[ch] = T(color-cbehind[ch]);
            cbehind[ch]=cbehind[ch]*(1-alpha)+color*alpha;
        }


        //计算dalpha_dopacit
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

        //dalpha_dcen2p=alpha*conic*u;
        float dalpha_dcen2p[2] = {alpha*(dx*conics2d[i*3]+dy*conics2d[i*3+1]),alpha*(dx*conics2d[i*3+1]+dy*conics2d[i*3+2])};
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
           for(int j=0;j<2;j++)
           {
            atomicAdd(&dl_dcenp2d[i*2+j],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dcen2p[j]);
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

