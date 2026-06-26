#include <torch/extension.h>

torch::Tensor render_forward_cuda(
    torch::Tensor colors,
    torch::Tensor cens,
    torch::Tensor inv_covs,
    torch::Tensor weights,
    torch::Tensor inv_k,
    int64_t height,
    int64_t width);

namespace {

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t dim,
    c10::IntArrayRef tail_shape = {}) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.dim() == dim, name, " must have ", dim, " dimensions");
  for (int64_t i = 0; i < static_cast<int64_t>(tail_shape.size()); ++i) {
    TORCH_CHECK(
        tensor.size(dim - tail_shape.size() + i) == tail_shape[i],
        name,
        " has invalid shape");
  }
}

}  // namespace

torch::Tensor render_forward(
    torch::Tensor colors,
    torch::Tensor cens,
    torch::Tensor inv_covs,
    torch::Tensor weights,
    torch::Tensor inv_k,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  check_tensor(colors, "colors", 2, {3});
  check_tensor(cens, "cens", 2, {2});
  check_tensor(inv_covs, "inv_covs", 2, {4});
  check_tensor(weights, "weights", 1);
  check_tensor(inv_k, "inv_k", 2, {3, 3});

  const auto gaussian_num = colors.size(0);
  TORCH_CHECK(cens.size(0) == gaussian_num, "cens count must match colors");
  TORCH_CHECK(inv_covs.size(0) == gaussian_num, "inv_covs count must match colors");
  TORCH_CHECK(weights.size(0) == gaussian_num, "weights count must match colors");
  TORCH_CHECK(colors.device() == cens.device(), "all tensors must be on the same CUDA device");
  TORCH_CHECK(colors.device() == inv_covs.device(), "all tensors must be on the same CUDA device");
  TORCH_CHECK(colors.device() == weights.device(), "all tensors must be on the same CUDA device");
  TORCH_CHECK(colors.device() == inv_k.device(), "all tensors must be on the same CUDA device");

  return render_forward_cuda(
      colors.contiguous(),
      cens.contiguous(),
      inv_covs.contiguous(),
      weights.contiguous(),
      inv_k.contiguous(),
      height,
      width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_forward", &render_forward, "Gaussian render forward (CUDA)");
}
