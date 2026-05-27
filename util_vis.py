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

if __name__ =="__main__":
    print("Testing mesh in Open3D...")
    # armadillo_mesh = o3d.data.ArmadilloMesh()
    # mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)



    knot_mesh = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    colors = np.random.rand(len(mesh.vertices),3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh])

    print(mesh)
    print('Vertices:',np.asarray(mesh.vertices).shape)
    print(np.asarray(mesh.vertices))
    # print('vertices color',np.asarray(mesh.vertices.color))
    print('Triangles:',np.asarray(mesh.triangles).shape)
    print(np.asarray(mesh.triangles))

    # sphere = o3d.geometry.TriangleMesh.create_sphere()
    # vis = o3d.visualization.Visualizer()
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

# vis.add_geometry(sphere)
# vis.run()
# o3d.visualization.draw([sphere,frame])