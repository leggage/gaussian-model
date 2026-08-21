#include <torch/extension.h>


std::tuple<int,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
rasterize_gaussian(
    torch::Tensor colors,
    torch::Tensor opacity,
    torch::Tensor mean3w,
    torch::Tensor q,      //顺序依次为qr\qi\qj\qk
    torch::Tensor s,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    int64_t height,
    int64_t width,
    float fx,
    float fy);


    
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
backward_grad(
    int num_rendered,
    int64_t image_height,
    int64_t image_width,
    float fx,
    float fy,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,

    torch::Tensor image_buffer,
    torch::Tensor binning_buffer,
    torch::Tensor geometry_buffer,

    torch::Tensor dl_dpixel,
    torch::Tensor colors,
    torch::Tensor opacity,
    torch::Tensor cen3w,
    torch::Tensor q,
    torch::Tensor s
);