from matplotlib.font_manager import weight_dict
from pandas import wide_to_long
import cupy as cp
import numpy as np
from sympy import true
from torch._refs import to
from torch.optim import optimizer
from torch.utils.dlpack import from_dlpack, to_dlpack
import torch
import  matplotlib.pyplot as plt
from math import exp, pi, sqrt
from diff_rasterization import rasterizer
from rec_scene import Scene
from pathlib import Path
import pycolmap


modelPath = r'keyboard/sparse/0'


imagedir = Path("keyboard/images")

def main():
    rec = pycolmap.Reconstruction(modelPath)
    scene = Scene(rec,imagedir)

    

    gaussian = scene.gaussian
    colors  = gaussian.color
    opacity = gaussian.opacity
    mean3w  = gaussian.cen3w
    q       = gaussian.q
    gaussian.s.data.mul_(0.1)   # 原地改值，仍是 leaf
    s       = gaussian.s   
    opt = torch.optim.Adam([colors,opacity,mean3w,q,s],lr=0.001)
    for i in range(1000):
        opt.zero_grad()
        camera = scene.get_camera()
        image = rasterizer.apply(colors,opacity,mean3w,q,s,camera)
        gtproj = torch.from_numpy(camera.gtproj).cuda()
        loss = torch.mean((image-gtproj)**2)
        loss.backward()
        opt.step()
        print("id",i,"loss",loss,"\n")


        # image = rasterizer.apply(colors,opacity,mean3w,q,s,camera).cpu().detach().numpy()
        # print("image_render idx:",i)
    plt.subplot(1,2,1)
    plt.title("rendered")
    plt.imshow(image.cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.title('gt')
    plt.imshow(camera.gtproj)
    plt.show()    


if __name__ == "__main__":
    main()