from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

from util_vis import draw_camera, draw_camera_texture


DATASET_DIR = Path("keyboard")
SPARSE_DIR = DATASET_DIR / "sparse" / "0"
IMAGE_DIR = DATASET_DIR / "images"

CAMERA_SCALE = 3.5
IMAGE_Z = 2.5
CAMERA_COLOR = (0.1, 0.35, 1.0)


rec = pycolmap.Reconstruction(str(SPARSE_DIR))
geometries = []

# 1. COLMAP sparse points.
points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
colors = np.array([p.color for p in rec.points3D.values()], dtype=np.float64) / 255.0
print("point",rec.points3D.values().xyz)
# print("point",points)
print("colors","colors")

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
geometries.append(point_cloud)

# 2. COLMAP cameras and registered images.
for image in rec.images.values():
    if not image.has_pose:
        continue

    camera = rec.cameras[image.camera_id]
    pose = image.cam_from_world()

    R = np.asarray(pose.rotation.matrix(), dtype=np.float32)
    t = np.asarray(pose.translation, dtype=np.float32).reshape(3)
    print("R",R,'T',t)
    K = np.asarray(camera.calibration_matrix(), dtype=np.float32)

    image_path = IMAGE_DIR / image.name
    if image_path.exists():
        texture = np.ascontiguousarray(np.asarray(o3d.io.read_image(str(image_path))))
        camera_geometries = draw_camera_texture(
            R,
            t,
            scale=CAMERA_SCALE,
            width=int(camera.width),
            height=int(camera.height),
            image_z=IMAGE_Z,
            image=texture,
            intrinsic=K,
        )
        camera_geometries[0].colors = o3d.utility.Vector3dVector(
            [CAMERA_COLOR] * len(camera_geometries[0].lines)
        )
        geometries.extend(camera_geometries)
    else:
        camera_model = draw_camera(
            R,
            t,
            scale=CAMERA_SCALE,
            width=int(camera.width),
            height=int(camera.height),
            intrinsic=K,
        )
        camera_model.colors = o3d.utility.Vector3dVector(
            [CAMERA_COLOR] * len(camera_model.lines)
        )
        geometries.append(camera_model)

# 3. Display.
geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
o3d.visualization.draw_geometries(geometries, window_name="keyboard COLMAP")
