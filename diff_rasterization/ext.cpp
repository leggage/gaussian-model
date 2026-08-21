
#include <torch/extension.h>

#include "rasterize_point.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("render", &rasterize_gaussian);
  m.def("backward",&backward_grad);
}