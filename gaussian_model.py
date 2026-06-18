
from math import exp, pi, sqrt
from operator import inv
from turtle import end_fill
from pandas.io.formats.style import plt
from numpy import dtype
import torch
from util_vis import draw_camera, visualize_gaussian as gvs
import open3d as o3d
###for a gaussian in a frame ,

class GaussianModel:
    def __init__(self,cenp,cov,color,opacity) -> None:
        self._cen_obj = cenp
        self._cov_obj = cov

        self._cen_cam = None
        self._cov_cam = None
        self._w_cam = None

        self._cen_ray = None
        self._cov_ray = None
        self._jco = None

        self._col = color
        self.opacity= opacity


        pass

    def _o2c(self,w,c,pos_object=None):
        self._w_cam = w
        self._cen_cam = w@self._cen_obj+c
        self._cov_cam = w@self._cov_obj@w.T
        if pos_object:
            pos_cam = w@pos_object+c
            return pos_cam
    
    def _c2r(self,pos_cam=None):
        self._cen_ray = torch.tensor([self._cen_cam[0]/self._cen_cam[2],self._cen_cam[1]/self._cen_cam[2],torch.norm(self._cen_cam.float())])
        self._jco = torch.tensor([[1/self._cen_cam[2],0,-self._cen_cam[0]/self._cen_cam[2]**2],[0,1/self._cen_cam[2],-self._cen_cam[1]/self._cen_cam[2]**2],[self._cen_cam[0]/torch.norm(self._cen_cam.float()),self._cen_cam[1]/torch.norm(self._cen_cam.float()),self._cen_cam[2]/torch.norm(self._cen_cam.float())]])
        self._cov_ray = self._jco@self._cov_cam.float()@self._jco.T
        if pos_cam:
            pos_ray = torch.tensor([pos_cam[0]/pos_cam[2],pos_cam[1]/pos_cam[2],torch.norm(pos_cam.float())])
            return pos_ray

    def gs_dist_cam(self,x):
        diff = x-self._cen_cam
        return 1/(sqrt((2*pi)**3*torch.det(self._cov_cam))*torch.det(torch.inverse(self._w_cam)))*exp(-0.5*(diff).T@torch.inverse(self._cov_cam)@diff)

    def gs_dist_ray(self,x):
        diff = x-self._cen_ray
        return 1/(sqrt((2*pi)**3*torch.det(self._cov_ray))*torch.det(torch.inverse(self._jco))*torch.det(torch.inverse(self._w_cam)))*exp(-0.5*(diff).T@torch.inverse(self._cov_ray)@diff)

    def footprint(self,x):
        diff = x-self._cen_ray[:2]
        cov_2d = self._cov_ray[:2,:2]
        return 1/(sqrt((2*pi)**3*torch.det(cov_2d))*torch.det(torch.inverse(self._jco))*torch.det(torch.inverse(self._w_cam)))*exp(-0.5*(diff).T@torch.inverse(cov_2d)@diff)


def alpha_blending(gaussian_list,x):   ###gaussian-kernel排序按照射线积分路径顺序，先被积分的在列表前面
    pixel = 0
    for idx,item in enumerate(gaussian_list):
        j = 0
        T = 1
        while j<idx:
            T = T*(1-gaussian_list[j].footprint(x))
            j = j+1
        pixel = pixel + item._col*T*item.footprint(x)
    return pixel
        

def rasterization(image,gaussian_list,K):
    H,W = image.shape[:2]
    image_processed = image
    for i in range(H):
        for j in range(W):
            coord_pixel = torch.tensor([j,i,1],dtype =torch.float32)   ####gaussian坐标顺序为x,y,z;i实际上对应的是y方向索引，w是x方向。
            coord_ray_zeq1 = torch.inverse(K)@coord_pixel
            color = alpha_blending(gaussian_list,coord_ray_zeq1[:2])
            image_processed[i,j,:] = color
    return image_processed


if __name__ == "__main__":
    cenp = torch.tensor([1,2,3],dtype = torch.float32)
    cenp1 = torch.tensor([4,1,5],dtype = torch.float32)

    cov_s = torch.tensor([[1,0,0],[0,3,0],[0,0,4]],dtype = torch.float32)
    cov_s1= torch.tensor([[2,0,0],[0,2,0],[0,0,4]],dtype = torch.float32)

    color1 = torch.tensor([0.1,0.4,0.7],dtype = torch.float32)
    color2 = torch.tensor([0.7,0.3,0.2],dtype = torch.float32)
    opacity = 0.9

    w = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype = torch.float32)
    c = torch.tensor([-1,-2,2],dtype = torch.float32)
    w1 = torch.tensor([[0,-1,0],[1,0,0],[0,0,1]],dtype = torch.float32)
    c1 = torch.tensor([0,0,5],dtype = torch.float32)
    intrinsic = torch.tensor([[256,0,256],[0,256,256],[0,0,1]],dtype = torch.float32)

    gs = GaussianModel(cenp,cov_s,color1,opacity)
    gs1 = GaussianModel(cenp1,cov_s1,color2,opacity)   
    gs._o2c(w,c)
    gs._c2r()

    gs1._o2c(w,c)
    gs1._c2r()
    gaussian_list = [gs,gs1]    

    
    rgbimg = torch.rand(512,512,3)


    plt.subplot(1,2,1)
    plt.title("original")
    plt.imshow(rgbimg)
    rendered_image = rasterization(rgbimg,gaussian_list,intrinsic)
    rendered_image = (rendered_image - rendered_image.min())/(rendered_image.max()-rendered_image.min()+1e-8)
    plt.subplot(1,2,2)
    plt.title("rendered")
    plt.imshow(rendered_image)

    plt.show()












    camera = draw_camera(w,c,intrinsic=intrinsic)
    camera1 = draw_camera(w1,c1,intrinsic=intrinsic)
    geo_gaussians = []
    for gaussian in gaussian_list:
        geo_gaussian = gvs(gaussian._cen_obj,gaussian._cov_obj,color=gaussian._col)
        geo_gaussians.append(geo_gaussian)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries(geo_gaussians+[frame]+[camera])

    # pixel = torch.tensor([1,2])

    # gs._o2c(w,c)
    # gs._c2r()

    # gs1._o2c(w,c)
    # gs1._c2r()
    # gaussian_lis = [gs,gs1]
    # res = alpha_blending(gaussian_lis,pixel)
    # print(res)

    # foot = gs.footprint(pixel)
    # print(foot)