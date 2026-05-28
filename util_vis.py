from tkinter import W
import open3d as o3d
import numpy as np
import torch


def visualize_gaussian(cen,cov,scale=1,resolution=24,color=None):
    evals,evecs = torch.linalg.eigh(cov)
    raddi = evecs@np.diag(torch.sqrt(scale*evals))
    T = np.eye(4)
    T[:3,:3] = raddi
    T[:3,3] = cen
    # print(T)
    
    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0,resolution)
    if len(color)!=0:
        sphere.paint_uniform_color(color)
    else:
        color = torch.rand(3)
        sphere.paint_uniform_color(color)

    # print(sphere)
    gaussian = sphere.transform(T)
    gaussian.compute_vertex_normals()
    # o3d.visualization.draw_geometries([sphere])
    return gaussian

def draw_camera(w,c,scale=0.5,width=512,height=512,intrinsic=None):
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = w
    extrinsic[:3,3] = c
    if intrinsic is None:
        intrinsic = torch.tensor([[width,0,width/2],[0,height,height/2],[0,0,1]])
    camera = o3d.geometry.LineSet.create_camera_visualization(width,height,intrinsic,extrinsic,scale)
    return camera

def create_grid(size=5, step=1, z=0.0, color=(0.5, 0.5, 0.5)):
    """在 z=z0 平面上画 size×size 的方格网格"""
    points = []
    lines = []
    idx = 0
    xs = np.arange(-size, size + step, step)
    for x in xs:
        points.append([x, -size, z])
        points.append([x, size, z])
        lines.append([idx, idx + 1])
        idx += 2
    for y in xs:
        points.append([-size, y, z])
        points.append([size, y, z])
        lines.append([idx, idx + 1])
        idx += 2
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


if __name__ =="__main__":

    w = torch.tensor([[2,0,0],[0,4,0],[0,0,3]])
    c = torch.tensor([4,0,7])
    intrinsic = torch.tensor([[50,0,256],[0,50,256],[0,0,1]])
    camera = draw_camera(w,c,intrinsic=intrinsic)


    grid = create_grid(size=20, step=0.1, z=0.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([frame, grid,camera])

    # print("Testing mesh in Open3D...")
    # armadillo_mesh = o3d.data.ArmadilloMesh()
    # mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)



    # knot_mesh = o3d.data.KnotMesh()
    # mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    # colors = np.random.rand(len(mesh.vertices),3)
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([mesh])

    # print(mesh)
    # print('Vertices:',np.asarray(mesh.vertices).shape)
    # print(np.asarray(mesh.vertices))
    # print('vertices color',np.asarray(mesh.vertices.color))
    # print('Triangles:',np.asarray(mesh.triangles).shape)
    # print(np.asarray(mesh.triangles))

    # sphere = o3d.geometry.TriangleMesh.create_sphere()
    # vis = o3d.visualization.Visualizer()
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

# vis.add_geometry(sphere)
# vis.run()
# o3d.visualization.draw([sphere,frame])