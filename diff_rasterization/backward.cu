#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

    //dl_dcov3w_cov2 PART
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

__global__ void preprocess_params_backward(
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
  
    float* __restrict__ conics2d,
    float* __restrict__ cov3w,
    float* __restrict__ means2d,
    float* __restrict__ weights,
    float* __restrict__ cov3r,
    float* __restrict__ cov2r
  ){
    auto idx = blockIdx.x * blockDim.x+ threadIdx.x;
    if(idx>=N){return;}
  
    //PREPARE MEANS2D
    float3 point_ori = {mean3d[idx*3],mean3d[idx*3+1],mean3d[idx*3+2]};
    float3 p_cam = transformPoint4x3_Mrow(point_ori,viewmatrix);  
  
    if(!checkinfrustum(point_ori,viewmatrix,p_cam)){return;}
  
    float4 point_cla = transformPoint4x4_Mrow(point_ori,projmatrix);
    float p_hom = 1/(point_cla.w+0.00000001);
    float3 p_proj ={point_cla.x*p_hom,point_cla.y*p_hom,point_cla.z*p_hom};
    means2d[idx*2] = ndc2pixel(p_proj.x,width);
    means2d[idx*2+1] = ndc2pixel(p_proj.y,height);
  
    //PREPARE CONICS2D/cov3w
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
    glm::mat2 cov2r_matrix = glm::mat2(cov2dray[0][0],cov2dray[0][1],cov2dray[1][0],cov2dray[1][1]);

    cov3r[idx*6+0] = cov3dray[0][0];
    cov3r[idx*6+1] = cov3dray[1][0];
    cov3r[idx*6+2] = cov3dray[2][0];
    cov3r[idx*6+3] = cov3dray[1][1];
    cov3r[idx*6+4] = cov3dray[2][1];
    cov3r[idx*6+5] = cov3dray[2][2];
    cov2r[idx*3+0] = cov2dray[0][0];
    cov2r[idx*3+1] = cov2dray[1][0];
    cov2r[idx*3+2] = cov2dray[1][1];
  
    float det1 = glm::determinant(cov3dray);
    float det2 = glm::determinant(cov2r_matrix);
    weights[idx] = sqrt(2*kPi*det1/det2);
  
  }
  


__global__ void backward(
  const int gaussian_num,
  const int image_height,
  const int image_width,
  const float* __restrict__ dl_dpixel, //height*width*chan
  const float* __restrict__ conics2d,
  const float* __restrict__ means2d,
  const float* __restrict__ weights,
  const float* __restrict__ colors,
  const float* __restrict__ opacity,
  const float* __restrict__ Tfinal,
  const float* __restrict__ cov3r,
  const float* __restrict__ cov2r,

  float* __restrict__ dl_dcolor,
  float* __restrict__ dl_dopacity,
  float* __restrict__ dl_dconics2d,
  float* __restrict__ dl_dcovr3d, 
  float* __restrict__ dl_dcovr2d,
  float* __restrict__ dl_dcenp2d)
{
    const int image_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int image_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (image_x >= image_width || image_y >= image_height) {
    return;
    }
    const int Idx = image_y*image_width+image_x;

//链式法则dl_da = for(pixel in all)for(ch in all),atomicAdd(,dl_dpixel[ch]*dchan[ch]_dalpha*dalpha_da)
    //最后链式法则求dl_...要用到的中间变量
    float dalpha_dconics[3] = {};
    float dalpha_dcov3r[6] = {};
    float dalpha_dcov2r[3] = {};

    float T = Tfinal[Idx];
    float cbehind[3] = {};
    float dchan_dalpha[3] = {};
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
            dchan_dalpha[ch] = T*(color-cbehind[ch]);
            cbehind[ch]=cbehind[ch]*(1-alpha)+color*alpha;
        }


        //计算dalpha_dopacit
        float dalpha_dopacity = weights[i]*G;

        //计算dalpha_dconic,只存上三角区域
        dalpha_dconics[0] = -0.5*alpha*dx*dx;
        dalpha_dconics[1] = -0.5*alpha*dx*dy;
        dalpha_dconics[2] = -0.5*alpha*dy*dy;

        //计算dalpha_dcov3r,dalpha_dcov2r,需要用到矩阵求逆
        glm::mat3 cov3r_matrix = glm::mat3(cov3r[i*6],cov3r[i*6+1],cov3r[i*6+2],
                                           cov3r[i*6+1],cov3r[i*6+3],cov3r[i*6+4],
                                           cov3r[i*6+2],cov3r[i*6+4],cov3r[i*6+5]);
        glm::mat2 cov2r_matrix = glm::mat2(cov2r[i*3],cov2r[i*3+1],
                                           cov2r[i*3+1],cov2r[i*3+2]);
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
            atomicAdd(&dl_dconics2d[i*3+j],dl_dpixel[Idx*3+c]*dchan_dalpha[c]*dalpha_dconics[j]);
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



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_v2(
    torch::Tensor dl_dpixel,
    torch::Tensor Tfinal,
    torch::Tensor colors,
    torch::Tensor opacity,
    torch::Tensor mean3d,
    torch::Tensor q,      //顺序依次为qr\qi\qj\qk
    torch::Tensor s,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    int64_t height,
    int64_t width,
    float fx,
    float fy)
{
    int N = static_cast<int>(colors.size(0));    //gaussian num
    int chans = static_cast<int>(colors.size(1)); 
    auto options   = colors.options();
    auto conics2d  = torch::zeros({N,3},options);
    auto means2d   = torch::zeros({N,2},options); 
    auto weights   = torch::zeros({N,1},options); 
    auto cov3w     = torch::zeros({N,6},options); 
    auto cov3r     = torch::zeros({N,6},options);
    auto cov2r     = torch::zeros({N,3},options);
    
    
    preprocess_params_backward<<<(N+255)/256,256>>>(
      q.contiguous().data_ptr<float>(),
      s.contiguous().data_ptr<float>(),
      mean3d.contiguous().data_ptr<float>(),
      viewmatrix.contiguous().data_ptr<float>(),
      projmatrix.contiguous().data_ptr<float>(),
      N,
      height,
      width,
      fx,
      fy,
      conics2d.contiguous().data_ptr<float>(),
      cov3w.contiguous().data_ptr<float>(),
      means2d.contiguous().data_ptr<float>(),
      weights.contiguous().data_ptr<float>(),
      cov3r.contiguous().data_ptr<float>(),
      cov2r.contiguous().data_ptr<float>()
    );

    auto dl_dcolors   = torch::zeros({N, chans}, options);
    auto dl_dopacity  = torch::zeros({N, 1}, options);
    auto dl_dconics2d = torch::zeros({N, 3}, options);
    auto dl_dcovr3d   = torch::zeros({N, 6}, options);
    auto dl_dcovr2d   = torch::zeros({N, 3}, options);
    auto dl_dcenp2d   = torch::zeros({N, 2}, options);  // 若是 2D 投影梯度
    const dim3 block(16,16);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1) / block.y); 

    backward<<<grid, block>>>(
        N,
        height,
        width,
        dl_dpixel.contiguous().data_ptr<float>(), //height*width*chan
        conics2d.contiguous().data_ptr<float>(),
        means2d.contiguous().data_ptr<float>(),
        weights.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        Tfinal.contiguous().data_ptr<float>(),
        cov3r.contiguous().data_ptr<float>(),
        cov2r.contiguous().data_ptr<float>(),
        
        dl_dcolors.contiguous().data_ptr<float>(),
        dl_dopacity.contiguous().data_ptr<float>(),
        dl_dconics2d.contiguous().data_ptr<float>(),
        dl_dcovr3d.contiguous().data_ptr<float>(), 
        dl_dcovr2d.contiguous().data_ptr<float>(),
        dl_dcenp2d.contiguous().data_ptr<float>());

    auto dl_dq       = torch::zeros({N, 4}, options);       // (N, 4) qr,qi,qj,qk
    auto dl_ds       = torch::zeros({N, 3}, options);       // (N, 3)
    auto dl_dcen3w   = torch::zeros({N, 3}, options);       // (N, 3)

    preprocess_backward<<<(N+255)/256,256>>>(
        N,
        fx,
        fy,
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        mean3d.contiguous().data_ptr<float>(),
        cov3w.contiguous().data_ptr<float>(),
        conics2d.contiguous().data_ptr<float>(),
        
        q.contiguous().data_ptr<float>(),
        s.contiguous().data_ptr<float>(),
        
        dl_dconics2d.contiguous().data_ptr<float>(),
        dl_dcovr3d.contiguous().data_ptr<float>(),
        dl_dcovr2d.contiguous().data_ptr<float>(),
        dl_dcenp2d.contiguous().data_ptr<float>(),
        
        dl_dq.contiguous().data_ptr<float>(),
        dl_ds.contiguous().data_ptr<float>(),
        dl_dcen3w.contiguous().data_ptr<float>());
    return std::make_tuple(dl_dcolors, dl_dopacity, dl_dq, dl_ds, dl_dcen3w);
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
