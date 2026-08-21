import torch

# from render_extension import render_forward
from diff_rasterization import render_forward,render_backward,rasterizer
from matplotlib import pyplot as plt
def render_torch_reference(colors, cens, inv_covs, weights, inv_k, height, width):
    ys, xs = torch.meshgrid(
        torch.arange(height, device="cuda", dtype=torch.float32),
        torch.arange(width, device="cuda", dtype=torch.float32),
        indexing="ij",
    )
    x_ray = xs * inv_k[0, 0] + ys * inv_k[0, 1] + inv_k[0, 2]
    y_ray = xs * inv_k[1, 0] + ys * inv_k[1, 1] + inv_k[1, 2]

    dx = x_ray.unsqueeze(-1) - cens[:, 0]
    dy = y_ray.unsqueeze(-1) - cens[:, 1]
    mahalanobis = (
        dx * dx * inv_covs[:, 0]
        + dx * dy * (inv_covs[:, 1] + inv_covs[:, 2])
        + dy * dy * inv_covs[:, 3]
    )
    alpha = weights * torch.exp(-0.5 * mahalanobis)
    alpha = torch.clamp(alpha, 0.0, 0.99)
    transmittance = torch.cumprod(
        torch.cat(
            [
                torch.ones((*alpha.shape[:-1], 1), device="cuda", dtype=torch.float32),
                1.0 - alpha[:, :, :-1],
            ],
            dim=-1,
        ),
        dim=-1,
    )
    return alpha * transmittance @ colors


def main():
    torch.manual_seed(3)
    height = 64
    width = 80
    gaussian_num = 4

    colors = torch.rand(gaussian_num, 3, device="cuda", dtype=torch.float32)
    pr_colors = torch.rand(gaussian_num, 3, device="cuda", dtype=torch.float32,requires_grad=True)
    cens = torch.tensor(
        [[0.1, 0.1], [0.0, -0.1], [-0.2, 0.2], [0.3, -0.25]],
        device="cuda",
        dtype=torch.float32,
    )
    inv_covs = torch.tensor(
        [[90.0, 0.0, 0.0, 90.0], [60.0, 5.0, 5.0, 75.0], [80.0, 0.0, 0.0, 45.0], [70.0, -4.0, -4.0, 70.0]],
        device="cuda",
        dtype=torch.float32,
    )
    weights = torch.tensor([0.8, 0.6, 0.7, 0.5], device="cuda", dtype=torch.float32)
    intrinsic = torch.tensor(
        [[50.0, 0.0, width / 2.0], [0.0, 50.0, height / 2.0], [0.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    inv_k = torch.inverse(intrinsic)

    image = render_forward(colors, cens, inv_covs, weights, inv_k, height, width)
    reference = render_torch_reference(colors, cens, inv_covs, weights, inv_k, height, width)

    torch.cuda.synchronize()

    # max_diff = (image - reference).abs().max().item()
    # print("image shape:", tuple(image.shape))
    # print("max abs diff:", max_diff)
    # assert image.shape == (height, width, 3)
    # assert torch.isfinite(image).all()
    # assert max_diff < 1e-5


    optimizer = torch.optim.Adam([pr_colors],lr=0.01)
    for i in range(3000):
        optimizer.zero_grad()
        pr_image = rasterizer.apply(pr_colors, cens, inv_covs, weights, inv_k, height, width)
        loss = torch.mean((pr_image-image)**2)
        loss.backward()
        optimizer.step()

        if i%100 ==0:
            print("step:",i,"loss:",loss,"pr_colors:",pr_colors)

    print("colors",colors)

    image =image.detach().cpu().numpy()
    pr_image =pr_image.detach().cpu().numpy()
    plt.subplot(2,1,1)
    plt.imshow(image)
    plt.subplot(2,1,2)
    plt.imshow(pr_image)
    plt.show()



if __name__ == "__main__":
    main()
