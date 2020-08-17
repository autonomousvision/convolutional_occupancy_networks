from os import listdir, makedirs, getcwd
from os.path import join, exists, isdir, exists
import json
import trimesh
import numpy as np 
from copy import deepcopy
import shutil
import zipfile
from tqdm import tqdm
from src.utils.io import export_pointcloud

def create_dir(dir_in):
    if not exists(dir_in):
        makedirs(dir_in)

base_path = 'data/Matterport3D/v1/scans'
scene_name = 'JmbYfDe2QKZ'
out_path = 'data/Matterport3D_processed'
scene_path = join(base_path, scene_name, 'region_segmentations')
regions = [join(scene_path, 'region'+str(m)+'.ply')) 
           for m in range(100) if exists(join(scene_path, 'region'+str(m)+'.ply'))]
outfile = join(out_path, scene_name)
create_dir(outfile)

n_pointcloud_points = 500000
dtype = np.float16
cut_mesh =True
save_part_mesh = False

mat_permute = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
for idx, r_path in tqdm(enumerate(regions)):
    mesh = trimesh.load(r_path)
    z_max = max(mesh.vertices[:, 2])
    z_range = max(mesh.vertices[:, 2]) - min(mesh.vertices[:, 2])
    x_min = min(mesh.vertices[:, 0])
    y_min = min(mesh.vertices[:, 1])
    # For better visualization, cut the ceilings and parts of walls
    if cut_mesh:
        mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([0, 0, -1]), np.array([0, 0, z_max - 0.5*z_range]))
        # mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([0, 1, 0]), np.array([0, y_min + 0.5, 0]))
        mesh = trimesh.intersections.slice_mesh_plane(mesh, np.array([1, 0, 0]), np.array([x_min + 0.2, 0, 0]))
        mesh = deepcopy(mesh)
        mesh.apply_transform(mat_permute)
        if save_part_mesh == True:
            out_file = join(outfile, 'mesh_fused%d.ply'%idx)
            mesh.export(out_file)
    
    if idx == 0:
        faces = mesh.faces
        vertices = mesh.vertices
    else:
        faces = np.concatenate([faces, mesh.faces + vertices.shape[0]])
        vertices = np.concatenate([vertices, mesh.vertices])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
out_file = join(outfile, 'mesh_fused.ply')
mesh.export(out_file)

# Sample surface points
pcl, face_idx = mesh.sample(n_pointcloud_points, return_index=True)
normals = mesh.face_normals[face_idx]

# save surface points
out_file = join(outfile, 'pointcloud.npz')
np.savez(out_file, points=pcl.astype(dtype), normals=normals.astype(dtype))
export_pointcloud(pcl, join(outfile, 'pointcloud.ply'))

# create test.lst file
with open(join(out_path, 'test.lst'), "w") as file:
    file.write(scene_name)