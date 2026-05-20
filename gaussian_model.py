
from math import exp, pi, sqrt
from operator import inv
from turtle import end_fill
import torch

###for a gaussian in a frame ,

class GaussianModel:
    def __init__(self,cenp,cov,color,density) -> None:
        self._cen_obj = cenp
        self._cov_obj = cov

        self._cen_cam = None
        self._cov_cam = None
        self._w_cam = None

        self._cen_ray = None
        self._cov_ray = None
        self._jco = None

        self._col = color
        self.density = density


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
        return 1/(sqrt((2*pi)**3*torch.det(self._cov_ray))*torch.det(torch.inverse(self._jco)))*exp(-0.5*(diff).T@torch.inverse(self._cov_ray)@diff)

    def footprint(self,x):
        diff = x-self._cen_ray[:2]
        cov_2d = self._cov_ray[:2,:2]
        return 1/(sqrt((2*pi)**3*torch.det(self._cov_ray))*torch.det(torch.inverse(self._jco)))*exp(-0.5*(diff).T@torch.inverse(cov_2d)@diff)




if __name__ == "__main__":
    cenp = torch.tensor([1,2,3])
    cov_s = torch.tensor([[1,0,0],[0,3,0],[0,0,4]])
    color =1
    density =2
    w = torch.tensor([[2,0,0],[0,4,0],[0,0,3]])
    c = torch.tensor([0,0,0])

    pixel = torch.tensor([1,2])

    gs = GaussianModel(cenp,cov_s,color,density)
    gs._o2c(w,c)
    gs._c2r()
    foot = gs.footprint(pixel)
    print(foot)