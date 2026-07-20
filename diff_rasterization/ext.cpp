
#include <torch/extension.h>

#include "wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("rasterize_gaussians", &render_forward_cuda);
  m.def("rasterize_gaussians", &render_forward_v2);
  m.def("grad_compute",&backward_cuda);
  m.def("grad_compute_v2", &backward_v2);

}