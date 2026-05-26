"""World-space visualization for 3D Gaussians, cameras, and image-plane footprints."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import torch

from gaussian_model import GaussianModel

CameraExtrinsic = Tuple[torch.Tensor, torch.Tensor]


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def distinct_colors(n: int) -> List[np.ndarray]:
    """Return n visually distinct RGB colors in [0, 1]."""
    if n == 0:
        return []
    cmap = __import__("matplotlib.cm", fromlist=["tab10"]).tab10
    colors = []
    for i in range(n):
        rgba = cmap(i % 10)
        colors.append(np.array(rgba[:3], dtype=np.float64))
    return colors


def cam_point_to_world(p_cam: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Map a camera-space point to world space: p_cam = w @ p_world + c."""
    w_f = _as_float_tensor(w)
    c_f = _as_float_tensor(c)
    p_f = _as_float_tensor(p_cam)
    return torch.linalg.inv(w_f) @ (p_f - c_f)


def camera_center_in_world(w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Optical center in world coordinates."""
    return cam_point_to_world(torch.zeros(3), w, c)


def image_plane_center_in_world(
    w: torch.Tensor, c: torch.Tensor, z_offset: float = 1.0
) -> torch.Tensor:
    """Image-plane center: optical center + z_offset along camera +Z."""
    return cam_point_to_world(torch.tensor([0.0, 0.0, z_offset]), w, c)


def prepare_gaussians_for_camera(
    gaussian_list: Sequence[GaussianModel], w: torch.Tensor, c: torch.Tensor
) -> None:
    for gs in gaussian_list:
        gs._o2c(w, c)
        gs._c2r()


def compute_image_plane_bounds(
    gaussian_list: Sequence[GaussianModel],
    margin: float = 3.0,
    min_half_extent: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Footprint plane bounds (x_min, x_max, y_min, y_max) on z = 1 plane."""
    x_vals: List[float] = []
    y_vals: List[float] = []
    for gs in gaussian_list:
        mu = _as_float_tensor(gs._cen_ray[:2])
        cov2 = _as_float_tensor(gs._cov_ray[:2, :2])
        std = torch.sqrt(torch.clamp(torch.diagonal(cov2), min=1e-6))
        x_vals.extend([mu[0] - margin * std[0], mu[0] + margin * std[0]])
        y_vals.extend([mu[1] - margin * std[1], mu[1] + margin * std[1]])

    if not x_vals:
        return -min_half_extent, min_half_extent, -min_half_extent, min_half_extent

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half_x = max(0.5 * (x_max - x_min), min_half_extent)
    half_y = max(0.5 * (y_max - y_min), min_half_extent)
    return cx - half_x, cx + half_x, cy - half_y, cy + half_y


def create_gaussian_ellipsoid(
    center: torch.Tensor,
    cov: torch.Tensor,
    color: np.ndarray,
    iso_level: float = 1.0,
    resolution: int = 24,
) -> o3d.geometry.TriangleMesh:
    """
    3D Gaussian isosurface ellipsoid: (x-mu)^T Sigma^{-1} (x-mu) = iso_level.
    """
    center_f = _as_float_tensor(center)
    cov_f = _as_float_tensor(cov)
    evals, evecs = torch.linalg.eigh(cov_f)
    evals = torch.clamp(evals, min=1e-8)
    radii = torch.sqrt(evals * iso_level).numpy()

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    transform = np.eye(4)
    transform[:3, :3] = evecs.numpy() @ np.diag(radii)
    transform[:3, 3] = center_f.numpy()
    mesh.transform(transform)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color.tolist())
    return mesh


def render_footprint_texture(
    gaussian_list: Sequence[GaussianModel],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    colors: Sequence[np.ndarray],
    resolution: int = 64,
) -> np.ndarray:
    """Render per-Gaussian colored footprints onto a single RGB image."""
    img = np.zeros((resolution, resolution, 3), dtype=np.float64)
    xs = torch.linspace(x_min, x_max, resolution)
    ys = torch.linspace(y_max, y_min, resolution)

    for gi, gs in enumerate(gaussian_list):
        layer = np.zeros((resolution, resolution), dtype=np.float64)
        for row in range(resolution):
            for col in range(resolution):
                uv = torch.tensor([xs[col], ys[row]], dtype=torch.float32)
                val = gs.footprint(uv)
                layer[row, col] = float(val.item() if torch.is_tensor(val) else val)
        if layer.max() > 0:
            layer = layer / layer.max()
        img += layer[..., None] * colors[gi]

    if img.max() > 0:
        img = img / img.max()
    return img


def create_textured_image_plane(
    w: torch.Tensor,
    c: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    texture_rgb: np.ndarray,
    z_offset: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """Create an image plane mesh in world space with an RGB texture."""
    cam_corners = [
        torch.tensor([x_min, y_min, z_offset], dtype=torch.float32),
        torch.tensor([x_max, y_min, z_offset], dtype=torch.float32),
        torch.tensor([x_max, y_max, z_offset], dtype=torch.float32),
        torch.tensor([x_min, y_max, z_offset], dtype=torch.float32),
    ]
    world_corners = np.stack([cam_point_to_world(p, w, c).numpy() for p in cam_corners])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(world_corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    mesh.textures = [o3d.geometry.Image((np.clip(texture_rgb, 0.0, 1.0) * 255).astype(np.uint8))]
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    mesh.compute_vertex_normals()
    return mesh


def create_camera_axis_lines(
    w: torch.Tensor,
    c: torch.Tensor,
    z_offset: float = 1.0,
    axis_scale: float = 0.5,
) -> o3d.geometry.LineSet:
    """Line from optical center to image-plane center, plus camera axes."""
    origin = camera_center_in_world(w, c).numpy()
    plane_center = image_plane_center_in_world(w, c, z_offset).numpy()
    w_inv = torch.linalg.inv(_as_float_tensor(w)).numpy()

    points = [origin, plane_center]
    lines = [[0, 1]]
    colors = [[1.0, 1.0, 0.0]]

    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        direction = w_inv[:, axis_idx] * axis_scale
        points.append(origin + direction)
        lines.append([0, len(points) - 1])
        colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return line_set


def create_image_plane_wireframe(
    w: torch.Tensor,
    c: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_offset: float = 1.0,
    color: Optional[np.ndarray] = None,
) -> o3d.geometry.LineSet:
    """Wireframe border for the image plane."""
    color = color if color is not None else np.array([1.0, 1.0, 1.0])
    cam_corners = [
        torch.tensor([x_min, y_min, z_offset], dtype=torch.float32),
        torch.tensor([x_max, y_min, z_offset], dtype=torch.float32),
        torch.tensor([x_max, y_max, z_offset], dtype=torch.float32),
        torch.tensor([x_min, y_max, z_offset], dtype=torch.float32),
    ]
    pts = np.stack([cam_point_to_world(p, w, c).numpy() for p in cam_corners])
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color.tolist()] * len(lines))
    return line_set


def visualize_gaussians_world(
    gaussian_list: Sequence[GaussianModel],
    cameras: Sequence[CameraExtrinsic],
    resolution: int = 64,
    iso_level: float = 1.0,
    z_offset: float = 1.0,
    show: bool = True,
    window_name: str = "3D Gaussian World Visualization",
) -> List[o3d.geometry.Geometry]:
    """
    Visualize in world space:
      1. 3D Gaussian isosurfaces (object/world frame)
      2. Camera optical centers and image planes (z = z_offset)
      3. Per-Gaussian footprints composited on each image plane (resolution x resolution)
    """
    if len(gaussian_list) == 0:
        raise ValueError("gaussian_list must not be empty")
    if len(cameras) == 0:
        raise ValueError("cameras must not be empty")

    colors = distinct_colors(len(gaussian_list))
    geometries: List[o3d.geometry.Geometry] = []

    for i, gs in enumerate(gaussian_list):
        ellipsoid = create_gaussian_ellipsoid(
            gs._cen_obj, gs._cov_obj, colors[i], iso_level=iso_level
        )
        geometries.append(ellipsoid)

    cam_colors = distinct_colors(len(cameras))

    for cam_idx, (w, c) in enumerate(cameras):
        prepare_gaussians_for_camera(gaussian_list, w, c)
        x_min, x_max, y_min, y_max = compute_image_plane_bounds(gaussian_list)

        cam_center = camera_center_in_world(w, c).numpy()
        cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        cam_sphere.translate(cam_center)
        cam_sphere.paint_uniform_color(cam_colors[cam_idx].tolist())
        cam_sphere.compute_vertex_normals()
        geometries.append(cam_sphere)

        footprint_tex = render_footprint_texture(
            gaussian_list, x_min, x_max, y_min, y_max, colors, resolution=resolution
        )
        image_plane = create_textured_image_plane(
            w, c, x_min, x_max, y_min, y_max, footprint_tex, z_offset=z_offset
        )
        geometries.append(image_plane)

        geometries.append(
            create_image_plane_wireframe(
                w, c, x_min, x_max, y_min, y_max, z_offset=z_offset, color=cam_colors[cam_idx]
            )
        )
        geometries.append(create_camera_axis_lines(w, c, z_offset=z_offset))

    if show:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1280,
            height=720,
        )

    return geometries


def save_footprint_images(
    gaussian_list: Sequence[GaussianModel],
    cameras: Sequence[CameraExtrinsic],
    output_dir: str,
    resolution: int = 64,
    prefix: str = "footprint",
) -> None:
    """Save per-camera footprint composites as PNG files (headless fallback)."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    colors = distinct_colors(len(gaussian_list))

    for cam_idx, (w, c) in enumerate(cameras):
        prepare_gaussians_for_camera(gaussian_list, w, c)
        x_min, x_max, y_min, y_max = compute_image_plane_bounds(gaussian_list)
        tex = render_footprint_texture(
            gaussian_list, x_min, x_max, y_min, y_max, colors, resolution=resolution
        )
        img = (np.clip(tex, 0.0, 1.0) * 255).astype(np.uint8)
        path = os.path.join(output_dir, f"{prefix}_cam{cam_idx}.png")
        o3d.io.write_image(path, o3d.geometry.Image(img))
