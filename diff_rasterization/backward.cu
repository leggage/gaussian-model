#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cooperative_groups.h>
#include "config.h"
#include "backward.h"
namespace cg = cooperative_groups;
namespace {
    constexpr float kPi = 3.14159265358979323846f;
    
    template <int Rows, int Cols>
    struct StaticMatrix {
        float data[Rows * Cols];
    
        __host__ __device__ float& operator()(int row, int col) {
            return data[col * Rows + row];
        }
    
        __host__ __device__ const float& operator()(int row, int col) const {
            return data[col * Rows + row];
        }
    };
    
    template <int Rows, int Cols>
    __device__ void vectorJacobianProduct(
        const float (&vector)[Rows],
        const StaticMatrix<Rows, Cols>& jacobian,
        float (&result)[Cols]) {
        #pragma unroll
        for (int col = 0; col < Cols; ++col) {
            float sum = 0.0f;
            #pragma unroll
            for (int row = 0; row < Rows; ++row) {
                sum += vector[row] * jacobian(row, col);
            }
            result[col] = sum;
        }
    }
    }

__global__ void preprocess_backward(
  const int gaussian_num,
  const float fx,
  const float fy,
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
    glm::mat2 dl_dconic = glm::mat2(dl_dconics2d[idx*3],dl_dconics2d[idx*3+1],dl_dconics2d[idx*3+1],dl_dconics2d[idx*3+2]);
    glm::mat2 dl_dcov2p = -conic*dl_dconic*conic;

    float3 p3w = {cen3w[idx*3],cen3w[idx*3+1],cen3w[idx*3+2]};
    float3 p3c = transformPoint4x3_Mrow(p3w,viewmatrix);
    //dl_dcov3w_conic PART
    glm::mat3 J2p = glm::mat3(fx/p3c.z,0,0,
                              0,fy/p3c.z,0,
                              -p3c.x*fx/(p3c.z*p3c.z),-p3c.y*fy/(p3c.z*p3c.z),0);
    glm::mat3 W = glm::mat3(viewmatrix[0],viewmatrix[3],viewmatrix[6],
                            viewmatrix[1],viewmatrix[4],viewmatrix[7],
                            viewmatrix[2],viewmatrix[5],viewmatrix[8]);
    glm::mat3 M = J2p*W;

    glm::mat3 dldcov2p = glm::mat3(dl_dcov2p[0][0],dl_dcov2p[1][0],0,
                                   dl_dcov2p[0][1],dl_dcov2p[1][1],0,
                                   0,0,0);
    glm::mat3 dl_dcov3w_conic = glm::transpose(M)*dldcov2p*M;

    //dl_dcov3w_cov3r PART
    float len = sqrt(p3c.x*p3c.x+p3c.y*p3c.y+p3c.z*p3c.z);
    glm::mat3 J3r = glm::mat3(1/p3c.z,0,p3c.x/len,
                              0,1/p3c.z,p3c.y/len,
                              -p3c.x/(p3c.z*p3c.z),-p3c.y/(p3c.z*p3c.z),p3c.z/len);
    M = J3r*W;
    glm::mat3 dldcov3r = glm::mat3(dl_dcovr3d[idx*6],dl_dcovr3d[idx*6+1],dl_dcovr3d[idx*6+2],
                                   dl_dcovr3d[idx*6+1],dl_dcovr3d[idx*6+3],dl_dcovr3d[idx*6+4],
                                   dl_dcovr3d[idx*6+2],dl_dcovr3d[idx*6+4],dl_dcovr3d[idx*6+5]);
    glm::mat3 dl_dcov3w_cov3r = glm::transpose(M)*dldcov3r*M;

    //dl_dcov3w_cov2r PART
    glm::mat3 J2r = glm::mat3(1/p3c.z,0,0,
                              0,1/p3c.z,0,
                              -p3c.x/(p3c.z*p3c.z),-p3c.y/(p3c.z*p3c.z),0);
    M = J2r*W;
    glm::mat3 dldcov2r = glm::mat3(dl_dcovr2d[idx*3],dl_dcovr2d[idx*3+1],0,
                                   dl_dcovr2d[idx*3+1],dl_dcovr2d[idx*3+2],0,
                                   0,0,0);
    glm::mat3 dl_dcov3w_cov2r = glm::transpose(M)*dldcov2r*M;
    //TOTAL dl_dcov3w
    glm::mat3 dl_dcov3w  = dl_dcov3w_conic+dl_dcov3w_cov3r+dl_dcov3w_cov2r;

//obtain dl_dq and dl_ds from  dl_dcov3w
    //obtain dl_dM,M=RS
    float4 q4 = {q[idx*4],q[idx*4+1],q[idx*4+2],q[idx*4+3]};
    float R[9] = {};
    computeRotation(q4,R);
         // glm::mat3 RT = glm::mat3(R[0],R[3],R[6],R[1],R[4],R[7],R[2],R[5],R[8]);
    glm::mat3 RS = glm::mat3(s[idx*3+0]*R[0],s[idx*3+0]*R[3],s[idx*3+0]*R[6],
                             s[idx*3+1]*R[1],s[idx*3+1]*R[4],s[idx*3+1]*R[7],
                             s[idx*3+2]*R[2],s[idx*3+2]*R[5],s[idx*3+2]*R[8]);
    glm::mat3 dl_dM = 2.0f*dl_dcov3w*RS;
    
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
    glm::mat3 dl_dJ3r = 2.0f*dldcov3r*J3r*M;
    float dLdJ3R[9] = {dl_dJ3r[0][0],dl_dJ3r[1][0],dl_dJ3r[2][0],
                       dl_dJ3r[0][1],dl_dJ3r[1][1],dl_dJ3r[2][1],
                       dl_dJ3r[0][2],dl_dJ3r[1][2],dl_dJ3r[2][2]};
    glm::mat3 dl_dJ2r = 2.0f*dldcov2r*J2r*M;
    float dLdJ2R[9] = {dl_dJ2r[0][0],dl_dJ2r[1][0],dl_dJ2r[2][0],
                       dl_dJ2r[0][1],dl_dJ2r[1][1],dl_dJ2r[2][1],
                       dl_dJ2r[0][2],dl_dJ2r[1][2],dl_dJ2r[2][2]};
    glm::mat3 dl_dJ2p = 2.0f*dldcov2p*J2p*M;
    float dLdJ2P[9] = {dl_dJ2p[0][0],dl_dJ2p[1][0],dl_dJ2p[2][0],
                       dl_dJ2p[0][1],dl_dJ2p[1][1],dl_dJ2p[2][1],
                       dl_dJ2p[0][2],dl_dJ2p[1][2],dl_dJ2p[2][2]};

    //compute dJ3r_dcen3c,dJ2r_dcen3c,dJ2p_dcen3c
    float xc = p3c.x;
    float yc = p3c.y;
    float zc = p3c.z;
    StaticMatrix<9, 3> dJ3r_dcen3c = {{
        0,0,-1/(zc*zc),0,0,0,(len-xc*xc/len)/(len*len),(-xc*yc/len)/(len*len),(-xc*zc/len)/(len*len),
        0,0,0,0,0,-1/(zc*zc),(-xc*yc/len)/(len*len),(len-yc*yc/len)/(len*len),(-yc*zc/len)/(len*len),
        -1/(zc*zc),0,2*xc/(zc*zc*zc),0,-1/(zc*zc),2*yc/(zc*zc*zc),(-xc*zc/len)/(len*len),(-yc*zc/len)/(len*len),(len-zc*zc/len)/(len*len)
    }};

    StaticMatrix<9, 3> dJ2r_dcen3c = {{
        0,0,-1/(zc*zc),0,0,0,0,0,0,
        0,0,0,0,0,-1/(zc*zc),0,0,0,
        -1/(zc*zc),0,2*xc/(zc*zc*zc),0,-1/(zc*zc),2*yc/(zc*zc*zc),0,0,0
    }};

    StaticMatrix<9, 3> dJ2p_dcen3c = {{
        0,0,-fx/(zc*zc),0,0,0,0,0,0,
        0,0,0,0,0,-fy/(zc*zc),0,0,0,
        -fx/(zc*zc),0,2*xc*fx/(zc*zc*zc),0,-fy/(zc*zc),2*yc*fy/(zc*zc*zc),0,0,0
    }};
                                
    //obtain matrix version of dl_dcen2p,dcen2p_dcen3c
    float dl_dcen2p[2] = {dl_dcenp2d[idx*2],dl_dcenp2d[idx*2+1]};
    StaticMatrix<2, 3> dcen2p_dcen3c = {{fx/zc,0,
                                         0,fy/zc,
                                         -fx*xc/(zc*zc),-fy*yc/(zc*zc)}};
    //compute dl_dcen3c
    float dl_dcen3c_from_J3r[3];
    float dl_dcen3c_from_J2r[3];
    float dl_dcen3c_from_J2p[3];
    float dl_dcen3c_from_proj[3];
    vectorJacobianProduct(dLdJ3R, dJ3r_dcen3c, dl_dcen3c_from_J3r);
    vectorJacobianProduct(dLdJ2R, dJ2r_dcen3c, dl_dcen3c_from_J2r);
    vectorJacobianProduct(dLdJ2P, dJ2p_dcen3c, dl_dcen3c_from_J2p);
    vectorJacobianProduct(dl_dcen2p, dcen2p_dcen3c, dl_dcen3c_from_proj);
    glm::vec3 dl_dcen3c(
        dl_dcen3c_from_J3r[0] + dl_dcen3c_from_J2r[0] + dl_dcen3c_from_J2p[0] + dl_dcen3c_from_proj[0],
        dl_dcen3c_from_J3r[1] + dl_dcen3c_from_J2r[1] + dl_dcen3c_from_J2p[1] + dl_dcen3c_from_proj[1],
        dl_dcen3c_from_J3r[2] + dl_dcen3c_from_J2r[2] + dl_dcen3c_from_J2p[2] + dl_dcen3c_from_proj[2]);

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



__global__ void __launch_bounds__(BlockSize_x * BlockSize_y) 
render_backward(
  const int image_height,
  const int image_width,
  const float* __restrict__ dl_dpixel, //height*width*chan
  const float* __restrict__ conics2d,
  const float* __restrict__ means2d,
  const float* __restrict__ weights,
  const float* __restrict__ colors,
  const float* __restrict__ opacity,
  const float* __restrict__ Tfinal,
  const uint32_t* __restrict__ ncontributor,
  const float* __restrict__ cov3r,
  const float* __restrict__ cov2r,
  const uint2* __restrict__ ranges,
  const uint64_t* __restrict__ point_list,

  float* __restrict__ dl_dcolor,
  float* __restrict__ dl_dopacity,
  float* __restrict__ dl_dconics2d,
  float* __restrict__ dl_dcovr3d, 
  float* __restrict__ dl_dcovr2d,
  float* __restrict__ dl_dcenp2d)
{
    const int image_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int image_y = blockIdx.y * blockDim.y + threadIdx.y;
    bool pixel_valid = image_x < image_width && image_y < image_height;
    bool done = !pixel_valid;

    int gridid = blockIdx.y * gridDim.x + blockIdx.x;
    int rangex = ranges[gridid].x;
    int rangey = ranges[gridid].y;
    int gaussian_num = ranges[gridid].y-ranges[gridid].x;
    int todo = gaussian_num;

    int BlockSize = BlockSize_x*BlockSize_y;

    int idx = threadIdx.y*blockDim.x+threadIdx.x;
    int Idx = cg::this_grid().thread_rank();
    int collected_id = idx;

    __shared__ uint32_t collected_gaussianid[BlockSize_x*BlockSize_y];
    __shared__ float collected_conic[BlockSize_x*BlockSize_y*3];
    __shared__ float collected_weight[BlockSize_x*BlockSize_y];
    __shared__ float collected_opacity[BlockSize_x*BlockSize_y];
    __shared__ float collected_cen2p[BlockSize_x*BlockSize_y*2];
    __shared__ float collected_color[BlockSize_x*BlockSize_y*3];
    __shared__ float collected_cov2r[BlockSize_x*BlockSize_y*3];
    __shared__ float collected_cov3r[BlockSize_x*BlockSize_y*6];
    uint32_t contributor = rangey-rangex;
    int rounds = (gaussian_num+BlockSize-1)/BlockSize;

    //最后链式法则求dl_...要用到的中间变量  
    float dalpha_dconics[3] = {0};
    float dalpha_dcov3r[6] = {0};
    float dalpha_dcov2r[3] = {0};

    float T = Tfinal[Idx];
    float cbehind[3] = {0};
    float dchan_dalpha[3] = {0};     
    for(int i=0;i<rounds;i++,todo-=BlockSize)
    {
        int offset = i*BlockSize+idx;
        //data fetching        
        if(offset<gaussian_num)
        {
            int progress = rangey -1 - offset;              //back to front
            int gaussianid = point_list[progress];
            collected_gaussianid[collected_id]= gaussianid;
            collected_weight[collected_id]    = weights[gaussianid];
            collected_opacity[collected_id]   = opacity[gaussianid];
            collected_cen2p[collected_id*2+0] = means2d[gaussianid*2+0];
            collected_cen2p[collected_id*2+1] = means2d[gaussianid*2+1];
 
            for(int t=0;t<6;t++)
            {
                if(t<3)
                {
                    collected_conic[collected_id*3+t] = conics2d[gaussianid*3+t];
                    collected_color[collected_id*3+t] = colors[gaussianid*3+t];
                    collected_cov2r[collected_id*3+t] = cov2r[gaussianid*3+t];                    
                }
                collected_cov3r[collected_id*6+t]  =  cov3r[gaussianid*6+t];
            }
        }
        __syncthreads();
        //data processing
        if(pixel_valid)
        {
            //链式法则dl_da = for(pixel in all)for(ch in all),atomicAdd(,dl_dpixel[ch]*dchan[ch]_dalpha*dalpha_da)          
            for(int j=0;j<min(todo,256) && !done;j++)
            {
                uint32_t g = collected_gaussianid[j];
                contributor--;
                if(contributor >= (ncontributor[Idx]-1)){continue;}
                float dx = image_x-collected_cen2p[j*2];
                float dy = image_y-collected_cen2p[j*2+1];
                float maloh = -0.5*(dx*dx*collected_conic[j*3]+
                                    2*dx*dy*collected_conic[j*3+1]+
                                    dy*dy*collected_conic[j*3+2]);
                float G = expf(maloh);
                float alpha = collected_opacity[j]*collected_weight[j]*G;  //当前gaussian的不透明度 alphai
                alpha = fminf(fmaxf(alpha, 0.0f), 0.99f);
                if(alpha<0.001){
                  continue;
                }
                T = T/(1-alpha);                                            //当前gaussian的透射率 Ti    

                //计算dchan_dalpha=Ti(ci[ch]-cbehind[ch])
                for(int ch=0;ch<3;ch++)
                {
                    float color = collected_color[j*3+ch];
                    //dchan_dalpha存储方式是通道优先，先通道一中所有gaussian再通道2...
                    dchan_dalpha[ch] = T*(color-cbehind[ch]);
                    cbehind[ch]=cbehind[ch]*(1-alpha)+color*alpha;
                }

                //计算dalpha_dopacit
                float dalpha_dopacity = collected_weight[j]*G;

                //计算dalpha_dconic,只存上三角区域
                dalpha_dconics[0] = -0.5*alpha*dx*dx;
                dalpha_dconics[1] = -0.5*alpha*dx*dy;
                dalpha_dconics[2] = -0.5*alpha*dy*dy;               
                //计算dalpha_dcov3r,dalpha_dcov2r,需要用到矩阵求逆
                glm::mat3 cov3r_matrix = glm::mat3(collected_cov3r[j*6],collected_cov3r[j*6+1],collected_cov3r[j*6+2],
                                                   collected_cov3r[j*6+1],collected_cov3r[j*6+3],collected_cov3r[j*6+4],
                                                   collected_cov3r[j*6+2],collected_cov3r[j*6+4],collected_cov3r[j*6+5]);
                glm::mat2 cov2r_matrix = glm::mat2(collected_cov2r[j*3],collected_cov2r[j*3+1],
                                                   collected_cov2r[j*3+1],collected_cov2r[j*3+2]);
                glm::mat3 dcov3r = 0.5f*alpha*glm::inverse(cov3r_matrix);
                glm::mat2 dcov2r = -0.5f*alpha*glm::inverse(cov2r_matrix);
                dalpha_dcov3r[0] = dcov3r[0][0];
                dalpha_dcov3r[1] = dcov3r[0][1];
                dalpha_dcov3r[2] = dcov3r[0][2];
                dalpha_dcov3r[3] = dcov3r[1][1];
                dalpha_dcov3r[4] = dcov3r[1][2];
                dalpha_dcov3r[5] = dcov3r[2][2];
                dalpha_dcov2r[0] = dcov2r[0][0];
                dalpha_dcov2r[1] = dcov2r[0][1];
                dalpha_dcov2r[2] = dcov2r[1][1];
                //dalpha_dcen2p=alpha*conic*u;
                float dalpha_dcen2p[2] = {alpha*(dx*collected_conic[j*3]+dy*collected_conic[j*3+1]),alpha*(dx*collected_conic[j*3+1]+dy*collected_conic[j*3+2])};
                float dchan_dcolor = T*alpha;
                
            //dl_...
                //这里对dl_dpixel的索引方式是建立在rgb存储通道优先的假设上的,如p0r,p0g,p0b,p1r,p1g,p1b...
                for(int c=0;c<3;c++)
                {
                    atomicAdd(&dl_dcolor[g*3+c],dl_dpixel[Idx*3+c]*dchan_dcolor); 
                    atomicAdd(&dl_dopacity[g],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dopacity);
                    for(int k=0;k<3;k++)
                    {
                        atomicAdd(&dl_dconics2d[g*3+k],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dconics[k]);
                        atomicAdd(&dl_dcovr2d[g*3+k],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dcov2r[k]);
                    }
                    for(int k=0;k<6;k++)
                    {
                        atomicAdd(&dl_dcovr3d[g*6+k],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dcov3r[k]);
                    }
                    for(int k=0;k<2;k++)
                    {
                        atomicAdd(&dl_dcenp2d[g*2+k],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dcen2p[k]);
                    }           

                }                
            }
        }
        __syncthreads();

    }


}


void Backward::render(
    const int image_height,
    const int image_width,
    const float* __restrict__ dl_dpixel, //height*width*chan
    const float* __restrict__ conics2d,
    const float* __restrict__ means2d,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
    const float* __restrict__ Tfinal,
    const uint32_t* __restrict__ ncontributor,
    const float* __restrict__ cov3r,
    const float* __restrict__ cov2r,
    const uint2* __restrict__ ranges,
    const uint64_t* __restrict__ point_list,
    const dim3 grid,
    const dim3 block,   
    float* __restrict__ dl_dcolor,
    float* __restrict__ dl_dopacity,
    float* __restrict__ dl_dconics2d,
    float* __restrict__ dl_dcovr3d, 
    float* __restrict__ dl_dcovr2d,
    float* __restrict__ dl_dcenp2d 
)
{
    render_backward<<<grid,block>>>(
        image_height,
        image_width,
        dl_dpixel, //height*width*chan
        conics2d,
        means2d,
        weights,
        colors,
        opacity,
        Tfinal,
        ncontributor,
        cov3r,
        cov2r,
        ranges,
        point_list,
        dl_dcolor,
        dl_dopacity,
        dl_dconics2d,
        dl_dcovr3d, 
        dl_dcovr2d,
        dl_dcenp2d         
    );
}

void Backward::preprocess(
  const int gaussian_num,
  const float fx,
  const float fy,
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
  float* __restrict__ dL_dcen3w
)
{
    preprocess_backward<<<(gaussian_num+255)/256,256>>>(
        gaussian_num,
        fx,
        fy,
        viewmatrix,
        projmatrix,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        cen3w,
        cov3w,
        conics2d,
      
        q,
        s,
      
        dl_dconics2d,
        dl_dcovr3d,
        dl_dcovr2d,
        dl_dcenp2d,
      
        dl_dq,
        dl_ds,
        dL_dcen3w);
}





// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// backward_v2(
//     torch::Tensor dl_dpixel,
//     torch::Tensor Tfinal,
//     torch::Tensor colors,
//     torch::Tensor opacity,
//     torch::Tensor mean3d,
//     torch::Tensor q,      //顺序依次为qr\qi\qj\qk
//     torch::Tensor s,
//     torch::Tensor viewmatrix,
//     torch::Tensor projmatrix,
//     int64_t height,
//     int64_t width,
//     float fx,
//     float fy)
// {
//     int N = static_cast<int>(colors.size(0));    //gaussian num
//     int chans = static_cast<int>(colors.size(1)); 
//     auto options   = colors.options();
//     auto conics2d  = torch::zeros({N,3},options);
//     auto means2d   = torch::zeros({N,2},options); 
//     auto weights   = torch::zeros({N,1},options); 
//     auto cov3w     = torch::zeros({N,6},options); 
//     auto cov3r     = torch::zeros({N,6},options);
//     auto cov2r     = torch::zeros({N,3},options);
    
    
//     preprocess_params_backward<<<(N+255)/256,256>>>(
//       q.contiguous().data_ptr<float>(),
//       s.contiguous().data_ptr<float>(),
//       mean3d.contiguous().data_ptr<float>(),
//       viewmatrix.contiguous().data_ptr<float>(),
//       projmatrix.contiguous().data_ptr<float>(),
//       N,
//       height,
//       width,
//       fx,
//       fy,
//       conics2d.contiguous().data_ptr<float>(),
//       cov3w.contiguous().data_ptr<float>(),
//       means2d.contiguous().data_ptr<float>(),
//       weights.contiguous().data_ptr<float>(),
//       cov3r.contiguous().data_ptr<float>(),
//       cov2r.contiguous().data_ptr<float>()
//     );

//     auto dl_dcolors   = torch::zeros({N, chans}, options);
//     auto dl_dopacity  = torch::zeros({N, 1}, options);
//     auto dl_dconics2d = torch::zeros({N, 3}, options);
//     auto dl_dcovr3d   = torch::zeros({N, 6}, options);
//     auto dl_dcovr2d   = torch::zeros({N, 3}, options);
//     auto dl_dcenp2d   = torch::zeros({N, 2}, options);  // 若是 2D 投影梯度
//     const dim3 block(16,16);
//     const dim3 grid(
//         (static_cast<unsigned int>(width) + block.x - 1) / block.x,
//         (static_cast<unsigned int>(height) + block.y - 1) / block.y); 

//     backward<<<grid, block>>>(
//         N,
//         height,
//         width,
//         dl_dpixel.contiguous().data_ptr<float>(), //height*width*chan
//         conics2d.contiguous().data_ptr<float>(),
//         means2d.contiguous().data_ptr<float>(),
//         weights.contiguous().data_ptr<float>(),
//         colors.contiguous().data_ptr<float>(),
//         opacity.contiguous().data_ptr<float>(),
//         Tfinal.contiguous().data_ptr<float>(),
//         cov3r.contiguous().data_ptr<float>(),
//         cov2r.contiguous().data_ptr<float>(),
        
//         dl_dcolors.contiguous().data_ptr<float>(),
//         dl_dopacity.contiguous().data_ptr<float>(),
//         dl_dconics2d.contiguous().data_ptr<backward_cuda 3}, options);       // (N, 3)

//     preprocess_backward<<<(N+255)/256,256>>>(
//         N,
//         fx,
//         fy,
//         viewmatrix.contiguous().data_ptr<float>(),
//         projmatrix.contiguous().data_ptr<float>(),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
//         mean3d.contiguous().data_ptr<float>(),
//         cov3w.contiguous().data_ptr<float>(),
//         conics2d.contiguous().data_ptr<float>(),
        
//         q.contiguous().data_ptr<float>(),
//         s.contiguous().data_ptr<float>(),
        
//         dl_dconics2d.contiguous().data_ptr<float>(),
//         dl_dcovr3d.contiguous().data_ptr<float>(),
//         dl_dcovr2d.contiguous().data_ptr<float>(),
//         dl_dcenp2d.contiguous().data_ptr<float>(),
        
//         dl_dq.contiguous().data_ptr<float>(),
//         dl_ds.contiguous().data_ptr<float>(),
//         dl_dcen3w.contiguous().data_ptr<float>());
//     return std::make_tuple(dl_dcolors, dl_dopacity, dl_dq, dl_ds, dl_dcen3w);
// }



// torch::Tensor backward_cuda(
//     torch::Tensor dl_dpixel,
//     torch::Tensor inv_k,
//     torch::Tensor weight,
//     torch::Tensor cens,
//     torch::Tensor covs,
//     int64_t chans,
//     int64_t gaussian_nums,
//     int64_t width,
//     int64_t height
// ){
//     auto dl_dcolors = torch::empty({gaussian_nums,chans});
//     const dim3 block(16,16);
//     const dim3 grid(
//         (static_cast<unsigned int>(width) + block.x - 1) / block.x,
//         (static_cast<unsigned int>(height) + block.y - 1) / block.y);  
//     grad_kernel<<<grid, block>>>(
//         dl_dpixel.contiguous().data_ptr<float>(),
//         dl_dcolors.contiguous().data_ptr<float>(),
//         chans,
//         gaussian_nums,
//         height,
//         width,
//         inv_k.contiguous().data_ptr<float>(),
//         weight.contiguous().data_ptr<float>(),
//         cens.contiguous().data_ptr<float>(),
//         covs.contiguous().data_ptr<float>());
//     return dl_dcolors;
// }
