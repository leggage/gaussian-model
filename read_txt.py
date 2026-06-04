import pycolmap
import numpy as np

rec = pycolmap.Reconstruction("keyboard/sparse/0")

for img_id, img in rec.images.items():
    if not img.has_pose:
        continue
    cam = rec.cameras[img.camera_id]
    pose = img.cam_from_world()

    # 内参
    K = cam.calibration_matrix()
    # 外参
    R = pose.rotation.matrix()
    t = pose.translation.reshape(3, 1)


    print(img_id, K, R.shape, t.shape)


# camere_path = r"keyboard/sparse/0/cameras.bin"

# with open(camere_path,"rb") as f:
#     for line in f:
#         # line = line.strip()
#         # if line.startswith("#") or not line:
#         #     continue
#         line = line.decode("ascii")
#         parts = line.split()

#         print(parts)