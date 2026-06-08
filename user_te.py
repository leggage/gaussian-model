from matplotlib.font_manager import weight_dict
from pandas import wide_to_long
import cupy as cp
import numpy as np
from sympy import true
from torch._refs import to
from torch.optim import optimizer
from gaussian_model import GaussianModel
import torch
import  matplotlib.pyplot as plt
from math import exp, pi, sqrt

_KERNEL_CODE = r'''
extern "C" __global__
void render_alpha_blending(
    const int gaussian_num,
    const int height,
    const int width,    
    const float* __restrict__ k_inv,
    const float* __restrict__ weights,    
    const float* __restrict__ centers,
    const float* __restrict__ inv_covs,
    const float* __restrict__ colors,
    float* __restrict__ image)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float px = (float)x;
    const float py = (float)y;

    const float ray_x = k_inv[0] * px + k_inv[1] * py + k_inv[2];
    const float ray_y = k_inv[3] * px + k_inv[4] * py + k_inv[5];
    const float ray_z = k_inv[6] * px + k_inv[7] * py + k_inv[8];
    const float inv_ray_z = 1.0f / ray_z;
    const float pos_x = ray_x * inv_ray_z;
    const float pos_y = ray_y * inv_ray_z;

    float out_r = 0.0f;
    float out_g = 0.0f;
    float out_b = 0.0f;
    float transmittance = 1.0f;

    for (int i = 0; i < gaussian_num; ++i) {
        const float dx = pos_x - centers[i * 2 + 0];
        const float dy = pos_y - centers[i * 2 + 1];

        const float a = inv_covs[i * 4 + 0];
        const float b = inv_covs[i * 4 + 1];
        const float c = inv_covs[i * 4 + 2];
        const float d = inv_covs[i * 4 + 3];
        const float mahalanobis = dx * (a * dx + b * dy) + dy * (c * dx + d * dy);
        const float alpha = weights[i] * expf(-0.5f * mahalanobis);

        out_r += colors[i * 3 + 0] * transmittance * alpha;
        out_g += colors[i * 3 + 1] * transmittance * alpha;
        out_b += colors[i * 3 + 2] * transmittance * alpha;
        transmittance *= (1.0f - alpha);
    }

    const int offset = (y * width + x) * 3;
    image[offset + 0] = out_r;
    image[offset + 1] = out_g;
    image[offset + 2] = out_b;
}
'''


_kernel_code = r"""
extern "C" __global__
void rasterization(int gaussian_num,int image_height,int image_width,float *inv_k,float *weight,float*cen,float*cov,float*colors,float* image){


    float image_x = blockIdx.x*blockDim.x+threadIdx.x;
    float image_y = blockIdx.y*blockDim.y+threadIdx.y;


    float ray_x = image_x*inv_k[0]+image_y*inv_k[1]+inv_k[2];
    float ray_y = image_x*inv_k[3]+image_y*inv_k[4]+inv_k[5];

    float out_r = 0.0f;
    float out_g = 0.0f;
    float out_b = 0.0f;
    float T =1;
    for(int i=0;i<gaussian_num;++i){
        float dx = ray_x - cen[2*i];
        float dy = ray_y - cen[2*i+1];

        float mahlo = dx*dx*cov[i*4]+dx*dy*(cov[i*4+1]+cov[i*4+2])+dy*dy*cov[i*4+3];
        float alpha = weight[i]*expf(-0.5*mahlo);


        out_r += colors[i * 3 + 0] * T * alpha;
        out_g += colors[i * 3 + 1] * T * alpha;
        out_b += colors[i * 3 + 2] * T* alpha;

        T = T*(1-alpha);      
    }
    const int offset = (image_y * image_width + image_x) * 3;
    image[offset + 0] = out_r;
    image[offset + 1] = out_g;
    image[offset + 2] = out_b;
}


"""

render_gpt = cp.RawKernel(_KERNEL_CODE,"render_alpha_blending")
render_kernel = cp.RawKernel(_kernel_code,"rasterization")


def render_torch(height,width,inv_k,cen,inv_covs,weights,colors):  ####cen(N,2),inv_covs(N,4),weights(N,),colors(N,3)
    ys,xs = torch.meshgrid(torch.arange(height,device='cuda',dtype=torch.float),
                            torch.arange(width,device='cuda',dtype=torch.float),indexing="ij")
    x_ray = xs*inv_k[0,0]+ys*inv_k[0,1]+inv_k[0,2]
    y_ray = xs*inv_k[1,0]+ys*inv_k[1,1]+inv_k[1,2]

    dx = x_ray.unsqueeze(-1) - cen[:,0]
    dy = y_ray.unsqueeze(-1) - cen[:,1]
    # print(dx.shape)
    mahalanobis = dx*dx*inv_covs[:,0]+dx*dy*(inv_covs[:,1]+inv_covs[:,2])+dy*dy*inv_covs[:,3]
    alpha = weights*torch.exp(-0.5*(mahalanobis))
 
    T = torch.cumprod(torch.cat([torch.ones((*alpha.shape[:-1],1),device="cuda",dtype=torch.float),1.0-alpha[:,:,:-1]],dim=-1),dim=-1)
    rendered = alpha*T@colors
    return rendered



def _normalize_for_display(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)
def _as_numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


if __name__ == "__main__":

    cenp = torch.tensor([1, 2, 3], dtype=torch.float32)
    cenp1 = torch.tensor([4, 1, 5], dtype=torch.float32)

    cov_s = torch.tensor([[1, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=torch.float32)
    cov_s1 = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 4]], dtype=torch.float32)

    color1 = torch.tensor([0.1, 0.4, 0.7], dtype=torch.float32)
    color2 = torch.tensor([0.7, 0.3, 0.2], dtype=torch.float32)
    density = 2

    w = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    c = torch.tensor([-1, -2, 2], dtype=torch.float32)
    intrinsic = torch.tensor([[50, 0, 256], [0, 50, 256], [0, 0, 1]], dtype=torch.float32)

    gs = GaussianModel(cenp, cov_s, color1, density)
    gs1 = GaussianModel(cenp1, cov_s1, color2, density)
    gs._o2c(w, c)
    gs._c2r()
    gs1._o2c(w, c)
    gs1._c2r()
    gaussian_list = [gs, gs1]

    block_size = cp.asarray([16,16],dtype = int)

    image = cp.empty((512,512,3),dtype =cp.float32)
    height = image.shape[0]
    width = image.shape[1]
    


    cens = []
    inv_covs = []
    weights = []
    colors = []

    for gaussian in gaussian_list:
        cens.append(np.asarray(gaussian._cen_ray[:2]).reshape(-1))

        cov2d = gaussian._cov_ray[:2,:2]
        weight = 1/(sqrt((2*pi)**3*torch.det(cov2d))*torch.det(torch.inverse(gaussian._jco))*torch.det(torch.inverse(gaussian._w_cam)))
        weights.append(weight)

        inv_cov = np.asarray(torch.inverse(cov2d)).reshape(-1)
        inv_covs.append(inv_cov)

        colors.append(np.asarray(gaussian._col).reshape(-1))


    cens_cuda = torch.tensor(np.array(cens),device="cuda",dtype=torch.float32).reshape(-1,2)
    inv_covs_cuda = torch.tensor(np.array(inv_covs),device="cuda",dtype=torch.float32).reshape(-1,4)
    weights_cuda = torch.tensor(np.array(weights),device="cuda",dtype=torch.float32)
    colors_cuda = torch.tensor(np.array(colors),device="cuda",dtype=torch.float32).reshape(-1,3)
    inv_k_cuda = torch.inverse(intrinsic).cuda()
    # print("inv_k:",inv_k_cuda)
    # print("weights:",weights_cuda)
    # print("cens:",cens_cuda)
    # print("inv_covs:",inv_covs_cuda)
    # print("colors:",colors_cuda)

    cens = cp.asarray(cens, dtype=cp.float32)
    inv_covs = cp.asarray(inv_covs, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)
    colors = cp.asarray(colors, dtype=cp.float32)
    inv_k = cp.asarray(np.linalg.inv(_as_numpy(intrinsic)), dtype=cp.float32).reshape(-1)

    block = (int(block_size[0]), int(block_size[1]))
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
    # print("grid:",grid)
    # print("block",block)
    print("inv_k:",inv_k)
    print("weights:",weights)
    print("cens:",cens)
    print("inv_covs:",inv_covs)
    print("colors:",colors)
    render_kernel(grid,block,(len(gaussian_list),height,width,inv_k,weights,cens,inv_covs,colors,image))
    gt_image = image.get()
    gt_image = torch.tensor(gt_image,device="cuda")


    


    # random_image = cp.empty([512,512,512],dtype=cp.float32)
    # render_kernel(grid,block,(len(gaussian_list),height,width,inv_k,weights,cens,inv_covs,colors,random_image))
    # random_image =random_image.get()
    # random_image = _normalize_for_display(random_image)
 
    # plt.subplot(1,3,1)
    # plt.title("random_init")
    # plt.imshow(random_image)
    pr_cens = torch.randn([len(gaussian_list),2],device="cuda",requires_grad=True)
    pr_colors = torch.randn([len(gaussian_list),3],device="cuda",requires_grad=True)
    optimizer = torch.optim.Adam([
        {"params": [pr_cens],  "lr": 0.001},
        {"params": [pr_colors], "lr": 0.01},
    ])




    for step in range(30000):
        optimizer.zero_grad()
        pr_image = render_torch(height,width,inv_k_cuda,pr_cens,inv_covs_cuda,weights_cuda,pr_colors)
        # print("pr_image max:", pr_image.max().item())
        loss = torch.mean((pr_image-gt_image)**2)
        loss.backward()

        optimizer.step()
        if step%100 ==0:
            print("iter:",step,"loss:",loss.item())
            print("训练后 pr_colors:",pr_colors.detach().cpu().numpy())
            print("训练后 pr_cens:",pr_cens.detach().cpu().numpy())



    pr_image = pr_image.detach().cpu().numpy()
    gt_image = gt_image.detach().cpu().numpy()
    pr_image = _normalize_for_display(pr_image)
    gt_image = _normalize_for_display(gt_image)

    plt.subplot(1,2,1)
    plt.title("gt")
    plt.imshow(gt_image )

    plt.subplot(1,2,2)
    plt.title("preimage")
    plt.imshow(pr_image)
    plt.show()

