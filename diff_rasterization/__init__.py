
import torch


from . import cuda_rasterization


class rasterizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,colors, cens, inv_covs, weights, inv_k, height, width):

        ctx.save_for_backward(cens, inv_covs, weights, inv_k)
        ctx.height = height
        ctx.width = width
        return cuda_rasterization.rasterize_gaussians(
            colors,
            cens,
            inv_covs,
            weights,
            inv_k,
            int(height),
            int(width),
        )

    @staticmethod
    def backward(ctx, grad_outputs):
        cens, inv_covs, weights, inv_k= ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        N = cens.shape[0]
        dl_dcolors = cuda_rasterization.grad_compute(
            grad_outputs,
            inv_k,
            weights,
            cens,
            inv_covs,
            3,
            N,
            width,
            height
        ).cuda()
        return dl_dcolors,None,None,None,None,None,None

def render_forward(colors, cens, inv_covs, weights, inv_k, height, width):
    if not torch.cuda.is_available():
        raise RuntimeError("render_extension requires CUDA")
    return cuda_rasterization.rasterize_gaussians(
        colors,
        cens,
        inv_covs,
        weights,
        inv_k,
        int(height),
        int(width),
    )

def render_backward(
    dl_dpixel,
    inv_k,
    weight,
    cens,
    covs,
    chans,
    gaussian_nums,
    width,
    height
    ):
    return cuda_rasterization.grad_compute(
        dl_dpixel,
        inv_k,
        weight,
        cens,
        covs,
        chans,
        gaussian_nums,
        width,
        height
    )
