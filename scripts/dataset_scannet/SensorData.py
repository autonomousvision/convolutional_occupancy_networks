# Original code from ScanNet data exporter: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# Adapted for Python 3: https://github.com/daveredrum/ScanNet/edit/master/SensReader/python/SensorData.py
import os, struct
import numpy as np
import zlib
import imageio
import cv2
from tqdm import tqdm
import torch
import logging
logging.basicConfig(filename='scannet_generation.log',level=logging.DEBUG)


COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise ValueError("invalid type")


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise ValueError("invalid type")


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)


  def load(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      print('Number of frames: %d' % num_frames)      
      self.frames = []
      print('loading', filename)
      for i in tqdm(range(num_frames)):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)

  def extract_depth_images(self, image_size=None, frame_skip=1):
    depth_list, cam_pose_list = [], []
    cam_intr = self.intrinsic_depth[:3, :3]
    print('extracting', len(self.frames)//frame_skip, ' depth maps')
    for f in tqdm(range(0, len(self.frames), frame_skip)):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)

      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      depth = depth.astype(float)/1000.
      depth_list.append(depth)

      cam_pose_list.append(self.frames[f].camera_to_world)
    return depth_list, cam_pose_list, cam_intr
  
  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
    for f in tqdm(range(0, len(self.frames), frame_skip)):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      # imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
      imageio.imwrite(os.path.join(output_path, '%06d.png' % f), depth)

  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting camera intrinsics to', output_path)
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))


  def get_scale_mat(self, mesh, padding=0.0):
    bbox = mesh.bounds 
    loc = (bbox[0] + bbox[1]) / 2
    scale = (1 - 2 * padding) / (bbox[1] - bbox[0]).max()
    S = np.eye(4) * scale
    S[:-1, -1] = -scale*loc
    S[-1, -1] = 1
    S_inv = np.linalg.inv(S)
    return S, S_inv
  

  def process_camera_dict(self, output_path, scale_matrix, resolution=(480, 640)):
    h, w = resolution

    #_, scale_matrix = self.get_scale_mat(mesh)

    out_dict = {}
    instrinsic_mat = self.intrinsic_depth
    # scale pixels to [-1, 1]
    scale_mat = np.array([
      [2. / (w-1), 0, -1, 0],
      [0, 2./(h-1), -1, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ])
    camera_mat = scale_mat @ instrinsic_mat
    mask_camera = []
    for f in range(0, len(self.frames), 1):
      out_dict['camera_mat_%d' % f] = camera_mat.astype(np.float32)

      world_mat_inv = self.frames[f].camera_to_world
      if np.any(np.isnan(world_mat_inv)) or np.any(np.isinf(world_mat_inv)):
          logging.warning('inf world mat for %s: %d' % (output_path, f))
          print('invalid world matrix!')
          mask_camera.append(f)

      try:
          world_mat = np.linalg.inv(world_mat_inv)
      except e:
          world_mat = np.linalg.pinv(world_mat_inv)

      out_dict['world_mat_%d' % f] = world_mat.astype(np.float32)
      out_dict['scale_mat_%d' % f] = scale_matrix.astype(np.float32)
      out_dict['camera_mask'] = mask_camera
    out_file = os.path.join(output_path, 'cameras.npz')
    np.savez(out_file, **out_dict)
