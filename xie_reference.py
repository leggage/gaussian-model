import cupy as cp
import numpy as np

_kernel_code = r"""
extern "C" __global__
void rasterization(int *block_size,image_height,image_width,float *inv_k,float *weight,float*cen,float*cov,float*color,float* output){
    int block_x = image_width/block_size[1];
    int block_y = image_height/block_size[0];

    int image_x = blockIdx.x%block_x*block_size[1] +threadIdx.x%block_size[1];
    int image_y = blockIdx.x/block_x*block_size[0] + threadIdx.x/block_size[0];


    int ray_x = image_x*inv_k[0]+image_y*inv_k[1]+inv_k[2];
    int ray_y = image_x*inv_k[3]+image_y*inv_k[4]+inv_k[5];

    int T =1
    for(i=0;i<gaussian_num;i++){
        dx = ray_x - cen[2*i];
        dy = ray_y - cen[2*i+1];

        mahlo = dx*dx*cov[i*4]+dx*dy*(cov[i*4+1]+cov[i*4+2])+dy*dy*cov[i*4+3];
        alpha = weight[i]*expr(-0.5*malho);


        output[(image_y*image_width+image_x)*3]+ = color[i*3]*T*alpha;
        output[(image_y*image_width+image_x)*3+1]+ = color[i*3+1]*T*alpha;
        output[(image_y*image_width+image_x)*3+2]+ = color[i*3+2]*T*alpha;

        T = T*(1-alpha);      
    }

}


"""



render_kernel = cp.RawKernerl(_kernel_code,"render code")


if __name__ == "__main__":
