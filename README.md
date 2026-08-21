# Gaussian Model

This is a small experimental Python project for modeling and visualizing 3D Gaussian primitives. It contains a prototype implementation of Gaussian coordinate transforms, projection to ray/image space, simple alpha blending, rasterization, and Open3D-based visualization.

The current code is closer to a research/demo script than a packaged library.

## Project Structure

```text
.
├── gaussian_model.py   # Core Gaussian model, projection, blending, rasterization demo
├── util_vis.py         # Open3D visualization helpers for Gaussians, cameras, grids
├── sd.py               # Temporary scratch script
├── kfk.py              # Temporary scratch script
├── test.json           # Open3D camera trajectory data, ignored by git
└── .gitignore
```

## Main Components

### `gaussian_model.py`

Defines `GaussianModel`, which stores one 3D Gaussian primitive:

- center position in object/world coordinates
- covariance matrix
- color
- density
- transformed camera-space state
- projected ray-space state

Important methods:

- `_o2c(w, c)`: transforms the Gaussian from object/world coordinates to camera coordinates.
- `_c2r()`: projects the camera-space Gaussian into ray/image parameter space.
- `gs_dist_cam(x)`: evaluates the Gaussian density in camera coordinates.
- `gs_dist_ray(x)`: evaluates the Gaussian density in ray coordinates.
- `footprint(x)`: evaluates the 2D projected footprint used for image blending.

The file also includes:

- `alpha_blending(gaussian_list, x)`: combines sorted Gaussian contributions at a pixel/ray coordinate.
- `rasterization(image, gaussian_list, K)`: loops over image pixels and renders Gaussian colors using camera intrinsics.
- a `__main__` demo that creates two Gaussian primitives and visualizes them with Open3D.

### `util_vis.py`

Provides visualization helpers:

- `visualize_gaussian(cen, cov, scale=1, resolution=24, color=None)`: converts a Gaussian covariance into an ellipsoid mesh.
- `draw_camera(w, c, scale=0.5, width=512, height=512, intrinsic=None)`: draws a camera frustum.
- `draw_camera_texture(...)`: draws a camera frustum and an optional textured image plane.
- `create_grid(...)`: creates a simple Open3D grid on a plane.

## Dependencies

The scripts currently use:

- Python 3
- PyTorch
- NumPy
- Open3D
- Matplotlib
- pandas

Install the main dependencies with:

```bash
pip install torch numpy open3d matplotlib pandas
```

Note: `util_vis.py` imports `cv`, but this module is not used in the current code. If running the file fails on that import, remove the import or replace it with the intended OpenCV import.

## Running the Demo

From this directory:

```bash
python gaussian_model.py
```

The demo creates two Gaussian primitives, transforms them into a camera frame, projects them, and opens an Open3D window showing:

- the Gaussian ellipsoids
- a coordinate frame
- a camera frustum

The rasterized image display code is currently commented out in `gaussian_model.py`.

## Notes and Limitations

- This project does not currently include a `requirements.txt`, `pyproject.toml`, or test suite.
- `sd.py` and `kfk.py` appear to be scratch files.
- `test.json` contains Open3D pinhole camera trajectory data and is ignored by git.
- The Gaussian list used by `alpha_blending` is expected to be sorted by integration/depth order.
- Some imports in `gaussian_model.py` appear unused and can be cleaned up later.
- The implementation is CPU-loop based for rasterization, so it is not optimized for large images or many Gaussians.

## Possible Next Steps

- Add `requirements.txt`.
- Clean unused imports.
- Rename internal methods like `_o2c` and `_c2r` to clearer public names.
- Add tests for coordinate transforms and Gaussian density evaluation.
- Vectorize rasterization with PyTorch for better performance.
- Add a small example image output path instead of only interactive Open3D visualization.
