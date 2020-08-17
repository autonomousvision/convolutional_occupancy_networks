import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src.utils import libmcubes
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
import time
import math

counter = 0


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info
        
    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        t0 = time.time()
        
        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c = self.encode_crop(inputs, device)
        else: # input the entire volume
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            t0 = time.time()
            with torch.no_grad():
                c = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0
        
        mesh = self.generate_from_latent(c, stats_dict=stats_dict, **kwargs)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh
    
    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def generate_mesh_sliding(self, data, return_stats=True):
        ''' Generates the output mesh in sliding-window manner.
            Adapt for real-world scale.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # acquire the boundary for every crops
        self.get_crop_bound(inputs)

        nx = self.resolution0
        n_crop = self.vol_bound['n_crop']
        n_crop_axis = self.vol_bound['axis_n_crop']

        # occupancy in each direction
        r = nx * 2**self.upsampling_steps
        occ_values = np.array([]).reshape(r,r,0)
        occ_values_y = np.array([]).reshape(r,0,r*n_crop_axis[2])
        occ_values_x = np.array([]).reshape(0,r*n_crop_axis[1],r*n_crop_axis[2])
        for i in trange(n_crop):
            # encode the current crop
            vol_bound = {}
            vol_bound['query_vol'] = self.vol_bound['query_vol'][i]
            vol_bound['input_vol'] = self.vol_bound['input_vol'][i]
            c = self.encode_crop(inputs, device, vol_bound=vol_bound)

            bb_min = self.vol_bound['query_vol'][i][0]
            bb_max = bb_min + self.vol_bound['query_crop_size']

            if self.upsampling_steps == 0:
                t = (bb_max - bb_min)/nx # inteval
                pp = np.mgrid[bb_min[0]:bb_max[0]:t[0], bb_min[1]:bb_max[1]:t[1], bb_min[2]:bb_max[2]:t[2]].reshape(3, -1).T
                pp = torch.from_numpy(pp).to(device)
                values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                values = values.reshape(nx, nx, nx)
            else:
                mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    pp = points / mesh_extractor.resolution
                    pp = pp * (bb_max - bb_min) + bb_min
                    pp = torch.from_numpy(pp).to(self.device)

                    values = self.eval_points(pp, c, vol_bound=vol_bound, **kwargs).detach().cpu().numpy()
                    values = values.astype(np.float64)
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()
                
                values = mesh_extractor.to_dense()
                # MISE consider one more voxel around boundary, remove
                values = values[:-1, :-1, :-1]

            # concatenate occ_value along every axis
            # along z axis
            occ_values = np.concatenate((occ_values, values), axis=2)
            # along y axis
            if (i+1) % n_crop_axis[2] == 0: 
                occ_values_y = np.concatenate((occ_values_y, occ_values), axis=1)
                occ_values = np.array([]).reshape(r, r, 0)
            # along x axis
            if (i+1) % (n_crop_axis[2]*n_crop_axis[1]) == 0:
                occ_values_x = np.concatenate((occ_values_x, occ_values_y), axis=0)
                occ_values_y = np.array([]).reshape(r, 0,r*n_crop_axis[2])
            
        value_grid = occ_values_x    
        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def get_crop_bound(self, inputs):
        ''' Divide a scene into crops, get boundary for each crop

        Args:
            inputs (dict): input point cloud
        '''
        query_crop_size = self.vol_bound['query_crop_size']
        input_crop_size = self.vol_bound['input_crop_size']
        lb_query_list, ub_query_list = [], []
        lb_input_list, ub_input_list = [], []
        
        lb = inputs.min(axis=1).values[0].cpu().numpy() - 0.01
        ub = inputs.max(axis=1).values[0].cpu().numpy() + 0.01
        lb_query = np.mgrid[lb[0]:ub[0]:query_crop_size,\
                    lb[1]:ub[1]:query_crop_size,\
                    lb[2]:ub[2]:query_crop_size].reshape(3, -1).T
        ub_query = lb_query + query_crop_size
        center = (lb_query + ub_query) / 2
        lb_input = center - input_crop_size/2
        ub_input = center + input_crop_size/2
        # number of crops alongside x,y, z axis
        self.vol_bound['axis_n_crop'] = np.ceil((ub - lb)/query_crop_size).astype(int)
        # total number of crops
        num_crop = np.prod(self.vol_bound['axis_n_crop'])
        self.vol_bound['n_crop'] = num_crop
        self.vol_bound['input_vol'] = np.stack([lb_input, ub_input], axis=1)
        self.vol_bound['query_vol'] = np.stack([lb_query, ub_query], axis=1)
        
    def encode_crop(self, inputs, device, vol_bound=None):
        ''' Encode a crop to feature volumes

        Args:
            inputs (dict): input point cloud
            device (device): pytorch device
            vol_bound (dict): volume boundary
        '''
        if vol_bound == None:
            vol_bound = self.vol_bound

        index = {}
        for fea in self.vol_bound['fea_type']:
            # crop the input point cloud
            mask_x = (inputs[:, :, 0] >= vol_bound['input_vol'][0][0]) &\
                    (inputs[:, :, 0] < vol_bound['input_vol'][1][0])
            mask_y = (inputs[:, :, 1] >= vol_bound['input_vol'][0][1]) &\
                    (inputs[:, :, 1] < vol_bound['input_vol'][1][1])
            mask_z = (inputs[:, :, 2] >= vol_bound['input_vol'][0][2]) &\
                    (inputs[:, :, 2] < vol_bound['input_vol'][1][2])
            mask = mask_x & mask_y & mask_z
            
            p_input = inputs[mask]
            if p_input.shape[0] == 0: # no points in the current crop
                p_input = inputs.squeeze()
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
                if fea == 'grid':
                    ind[~mask] = self.vol_bound['reso']**3
                else:
                    ind[~mask] = self.vol_bound['reso']**2
            else:
                ind = coord2index(p_input.clone(), vol_bound['input_vol'], reso=self.vol_bound['reso'], plane=fea)
            index[fea] = ind.unsqueeze(0)
            input_cur = add_key(p_input.unsqueeze(0), index, 'points', 'index', device=device)
        
        with torch.no_grad():
            c = self.model.encode_inputs(input_cur)
        return c
    
    def predict_crop_occ(self, pi, c, vol_bound=None, **kwargs):
        ''' Predict occupancy values for a crop

        Args:
            pi (dict): query points
            c (tensor): encoded feature volumes
            vol_bound (dict): volume boundary
        '''
        occ_hat = pi.new_empty((pi.shape[0]))
    
        if pi.shape[0] == 0:
            return occ_hat
        pi_in = pi.unsqueeze(0)
        pi_in = {'p': pi_in}
        p_n = {}
        for key in self.vol_bound['fea_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(pi.clone(), vol_bound['input_vol'], plane=key).unsqueeze(0).to(self.device)
        pi_in['p_n'] = p_n
        
        # predict occupancy of the current crop
        with torch.no_grad():
            occ_cur = self.model.decode(pi_in, c, **kwargs).logits
        occ_hat = occ_cur.squeeze(0)
        
        return occ_hat

    def eval_points(self, p, c=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            if self.input_type == 'pointcloud_crop':
                if self.vol_bound is not None: # sliding-window manner
                    occ_hat = self.predict_crop_occ(pi, c, vol_bound=vol_bound, **kwargs)
                    occ_hats.append(occ_hat)
                else: # entire scene
                    pi_in = pi.unsqueeze(0).to(self.device)
                    pi_in = {'p': pi_in}
                    p_n = {}
                    for key in c.keys():
                        # normalized to the range of [0, 1]
                        p_n[key] = normalize_coord(pi.clone(), self.input_vol, plane=key).unsqueeze(0).to(self.device)
                    pi_in['p_n'] = p_n
                    with torch.no_grad():
                        occ_hat = self.model.decode(pi_in, c, **kwargs).logits
                    occ_hats.append(occ_hat.squeeze(0).detach().cpu())
            else:
                pi = pi.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    occ_hat = self.model.decode(pi, c, **kwargs).logits
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh