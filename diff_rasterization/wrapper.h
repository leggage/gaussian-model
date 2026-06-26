#include <torch/extension.h>

torch::Tensor render_forward_cuda(
    torch::Tensor colors,
    torch::Tensor cens,
    torch::Tensor inv_covs,
    torch::Tensor weights,
    torch::Tensor inv_k,
    int64_t height,
    int64_t width);


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
);