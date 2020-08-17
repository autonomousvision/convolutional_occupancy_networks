import os
import time
import numpy as np
import argparse
import trimesh
from SensorData import SensorData
import tqdm
from os.path import join
from os import listdir
import numpy as np
import multiprocessing

path_in = '/dir/to/scannet_v2'
path_out = '/dir/to/scannet_out'

if not os.path.exists(path_out):
	os.makedirs(path_out)

path_out = join(path_out, 'scenes')
if not os.path.exists(path_out):
	os.makedirs(path_out)


def align_axis(file_name, mesh):
    rotation_matrix = np.array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
    ])
    lines = open(file_name).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    axis_align_matrix = rotation_matrix @ axis_align_matrix
    mesh.apply_transform(axis_align_matrix)
    return mesh, axis_align_matrix

def sample_points(mesh, n_points=100000, p_type=np.float16):
	pcl, idx = mesh.sample(n_points, return_index=True)
	normals = mesh.face_normals[idx]
	out_dict = {
		'points': pcl.astype(p_type),
		'normals': normals.astype(p_type),

	}
	return out_dict

def scale_to_unit_cube(mesh, y_level=-0.5):
        bbox = mesh.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = 1. / (bbox[1] - bbox[0]).max()
        vertices_t = (mesh.vertices - loc.reshape(-1, 3)) * scale
        y_min = min(vertices_t[:, 1])

        # create_transform_matrix
        S_loc = np.eye(4)
        S_loc[:-1, -1] = -loc
        # create scale mat
        S_scale = np.eye(4) * scale
        S_scale[-1, -1] = 1
        # create last translate matrix
        S_loc2 = np.eye(4)
        S_loc2[1, -1] = -y_min + y_level

        S = S_loc2 @ S_scale @ S_loc
        mesh.apply_transform(S)
        
        return mesh, S


def process(scene_name):
        out_path_cur = os.path.join(path_out, scene_name)
        if not os.path.exists(out_path_cur):
            os.makedirs(out_path_cur)

        # load mesh
        mesh = trimesh.load(os.path.join(path_in, scene_name, scene_name+'_vh_clean.ply'), process=False)
        txt_file = os.path.join(path_in, scene_name, '%s.txt' % scene_name)
        mesh, align_mat = align_axis(txt_file, mesh)
        mesh, scale_mat = scale_to_unit_cube(mesh)
        scale_matrix = np.linalg.inv(scale_mat @ align_mat)

        file_cur = os.path.join(path_in, scene_name, scene_name+'.sens')
        sd = SensorData(file_cur)
        sd.export_depth_images(os.path.join(path_out, scene_name, 'depth'), frame_skip=1)
        sd.process_camera_dict(join(path_out, scene_name), scale_matrix)
        pcl = sample_points(mesh)
        out_file = join(path_out, scene_name, 'pointcloud.npz')
        np.savez(out_file, **pcl)

file_list = listdir(path_in)
file_list.sort()
pbar = tqdm.tqdm()
pool = multiprocessing.Pool(processes=8)
for f in file_list:
    pool.apply_async(process, args=(f,), callback=lambda _: pbar.update())
pool.close()
pool.join()
