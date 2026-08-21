import math
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from sympy.plotting.plot import centers_of_faces
import torch

from gaussian_model import GaussianModel, rasterization


_KERNEL_CODE = r'''
extern "C" __global__
void render_alpha_blending(
    const float* __restrict__ k_inv,
    const float* __restrict__ centers,
    const float* __restrict__ inv_covs,
    const float* __restrict__ weights,
    const float* __restrict__ colors,
    const int gaussian_num,
    const int height,
    const int width,
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


_RENDER_KERNEL = cp.RawKernel(_KERNEL_CODE, "render_alpha_blending")


def _as_numpy(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _det(value):
    return float(np.linalg.det(_as_numpy(value, dtype=np.float64)))


def _prepare_gaussians(gaussian_list):
    centers = []
    inv_covs = []
    weights = []
    colors = []

    for idx, gaussian in enumerate(gaussian_list):
        if gaussian._cen_ray is None or gaussian._cov_ray is None:
            raise ValueError(
                f"gaussian_list[{idx}] has not been projected to ray space; "
                "call gaussian._o2c(...) and gaussian._c2r() before rendering."
            )

        center = _as_numpy(gaussian._cen_ray[:2])
        cov_2d = _as_numpy(gaussian._cov_ray[:2, :2], dtype=np.float64)
        det_cov = float(np.linalg.det(cov_2d))
        if det_cov <= 0.0:
            raise ValueError(f"gaussian_list[{idx}] has a non-positive 2D covariance determinant.")

        jco_scale = _det(np.linalg.inv(_as_numpy(gaussian._jco, dtype=np.float64)))
        w_scale = _det(np.linalg.inv(_as_numpy(gaussian._w_cam, dtype=np.float64)))
        weight = 1.0 / (math.sqrt((2.0 * math.pi) ** 3 * det_cov) * jco_scale * w_scale)

        centers.append(center)
        inv_covs.append(np.linalg.inv(cov_2d).reshape(4))
        weights.append(weight)
        colors.append(_as_numpy(gaussian._col))

    return (
        cp.asarray(np.asarray(centers, dtype=np.float32)),
        cp.asarray(np.asarray(inv_covs, dtype=np.float32)),
        cp.asarray(np.asarray(weights, dtype=np.float32)),
        cp.asarray(np.asarray(colors, dtype=np.float32)),
    )


def render(K, gaussian_list, image=None, image_shape=None, block_size=(16, 16)):
    if image is None and image_shape is None:
        raise ValueError("Either image or image_shape must be provided.")

    shape = tuple(image.shape) if image is not None else tuple(image_shape)
    if len(shape) < 2:
        raise ValueError("Image shape must contain at least height and width.")

    height, width = int(shape[0]), int(shape[1])
    k_inv = cp.asarray(np.linalg.inv(_as_numpy(K)), dtype=cp.float32).reshape(-1)
    centers, inv_covs, weights, colors = _prepare_gaussians(gaussian_list)
    output = cp.empty((height, width, 3), dtype=cp.float32)

    block = (int(block_size[0]), int(block_size[1]))
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
    print("grid:",grid)
    print("block",block)
    print("inv_k:",k_inv)
    print("weights:",weights)
    print("cens:",centers)
    print("inv_covs:",inv_covs)
    print("colors:",colors)
    _RENDER_KERNEL(
        grid,
        block,
        (
            k_inv,
            centers,
            inv_covs,
            weights,
            colors,
            len(gaussian_list),
            height,
            width,
            output,
        ),
    )
    return output


def rasterization_cuda(image, gaussian_list, K, block_size=(16, 16)):
    return render(K, gaussian_list, image=image, block_size=block_size)


def _normalize_for_display(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


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
    # print("grid:",grid)
    # print("block",block)
    # print("inv_k:",inv_k)
    # print("weights:",weights)
    # print("cens:",cens)
    # print("inv_covs:",inv_covs)
    # print("colors:",colors)
    intrinsic = torch.tensor([[50, 0, 256], [0, 50, 256], [0, 0, 1]], dtype=torch.float32)

    gs = GaussianModel(cenp, cov_s, color1, density)
    gs1 = GaussianModel(cenp1, cov_s1, color2, density)
    gs._o2c(w, c)
    gs._c2r()
    gs1._o2c(w, c)
    gs1._c2r()
    gaussian_list = [gs, gs1]

    rgbimg = torch.rand(512, 512, 3)

    # start = time.perf_counter()
    # rendered_image = rasterization(rgbimg.clone(), gaussian_list, intrinsic)
    # rasterization_time = time.perf_counter() - start

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    rendered_image_cuda = rasterization_cuda(rgbimg, gaussian_list, intrinsic, block_size=(16, 16))
    cp.cuda.Stream.null.synchronize()
    rasterization_cuda_time = time.perf_counter() - start

    # print(f"rasterization time: {rasterization_time:.6f} s")
    print(f"rasterization_cuda time: {rasterization_cuda_time:.6f} s")

    rendered_image_cuda_cpu = rendered_image_cuda.get()

    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(rgbimg.detach().cpu().numpy())

    # plt.subplot(1, 3, 2)
    # plt.title("rasterization")
    # plt.imshow(_normalize_for_display(rendered_image))

    plt.subplot(1, 3, 3)
    plt.title("rasterization_cuda")
    plt.imshow(_normalize_for_display(rendered_image_cuda_cpu))

    plt.tight_layout()
    plt.show()
