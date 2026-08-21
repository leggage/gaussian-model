from gaussian_model import Gaussianmodel,Camera
import numpy as np
import open3d as o3d
import torch
import random
class Scene:
    def __init__(self,rec,imagedir) -> None:
        gaussian = Gaussianmodel()
        gaussian.CreateFromPcd(rec.points3D)
        self.gaussian = gaussian

        camera_list = []

        for im in rec.images.values():
            if not im.has_pose:
                continue
            image_path = imagedir / im.name
            gtproj = np.asarray(o3d.io.read_image(str(image_path)))

            zf = 1
            zn = 0
            camera = rec.cameras[im.camera_id]

            fx=camera.params[0]
            fy=camera.params[0]
            width = camera.width
            height = camera.height
            W = np.eye(4,dtype=np.float32)
            W[:3,:] = im.cam_from_world().matrix()
            W = torch.from_numpy(W)
            fullproj = torch.tensor([[camera.params[0]/(width/2),0,0,0],[0,camera.params[0]/(height/2),0,0],[0,0,zf/(zf-zn),-zn*zf/(zf-zn)],[0,0,1,0]],dtype=torch.float32)
            camera_list.append(Camera(W,fullproj@W,fx,fy,gtproj,height,width))

        self.camera_list = camera_list
        pass

    def get_camera(self):

        return self.camera_list[random.randrange(len(self.camera_list))]