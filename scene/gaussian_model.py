#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import pickle
from utils.system_utils import mkdir_p
from utils.vis_utils import sort_smpl_rot, load_ply
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, HOIPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from knn_cuda import KNN
from nets.mlp_delta_body_pose import BodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
import torch.nn.functional as F


class GaussianModel:
    def __init__(self, sh_degree: int, smpl_type : str, motion_offset_flag : bool, motion_flag : bool,  actor_gender: str):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.labels = torch.empty(0)

        self.optimizer = None

        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


        self.device=torch.device('cuda', torch.cuda.current_device())
        # load MANO model
        if smpl_type == 'mano':
            neutral_smpl_path = os.path.join('mano/models', f'MANO_{actor_gender.upper()}.pkl')
            self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)


        # load knn module
        self.knn = KNN(k=1, transpose_mode=True)
        self.knn_near_2 = KNN(k=2, transpose_mode=True)

        self.motion_flag = motion_flag
        self.motion_offset_flag = motion_offset_flag
        if self.motion_offset_flag:
            # load pose correction module
            # import ipdb; ipdb.set_trace()
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3*(total_bones-1), mlp_width=128, mlp_depth=2)
            self.pose_decoder.to(self.device)

            # load lbs weight module
            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones=total_bones)
            self.lweight_offset_decoder.to(self.device)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_hand_xyz(self):
        return self._xyz[self.labels == 1]
    
    @property
    def get_obj_xyz(self):
        return self._xyz[self.labels == 0]
    
    @property
    def get_labels(self):
        return self.labels

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1, transform=None):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, transform)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale):
        if isinstance(pcd, HOIPointCloud):
            # TODO
            self.spatial_lr_scale = spatial_lr_scale
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])
            labels = torch.tensor(np.asarray(pcd.labels)).float().cuda()
            print("Number of hand : ", torch.sum(labels == 1).item(), "Number of object : ", (labels == 0).sum().item() )

            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.labels = labels
            
        elif isinstance(pcd, BasicPointCloud):
            self.spatial_lr_scale = 5.0
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        if not self.motion_offset_flag:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr, "name": "lweight_offset_decoder"},
            ] 

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_deformed_ply(self, save_name, iteration, path, viewpoint_camera, gaussians):
                        #    query_pts, params, t_params, t_vertices, lbs_weights=None, correct_Rs=None):
        mkdir_p(os.path.dirname(path))
        hand_means3D = gaussians.get_hand_xyz

        model_path_h = f'/data/home/acw773/Deformable-3D-Gaussians/output/ho3d/{save_name}'
        with open(model_path_h + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
            smpl_rot_unsorted = pickle.load(handle)
        smpl_rot, t_list = sort_smpl_rot(smpl_rot_unsorted)

        w2c = viewpoint_camera.world_view_transform.transpose(1, 2)
        # scene_data_h = load_hand_scene_data(model_path_h, iteration, smpl_rot, t_list, w2c)
        scene_data, _ = load_ply(os.path.join(model_path_h,
                                    "point_cloud",
                                    "iteration_" + str(iteration),
                                    "point_cloud.ply"), smpl_rot, t_list, w2c)

        

        # .detach().cpu().numpy()
        # xyz = s_means3D.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()

        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(path)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # import ipdb; ipdb.set_trace()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.labels = self.labels[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                   
        return optimizable_tensors

    def densification_postfix(self, new_labels, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        # import ipdb; ipdb.set_trace()
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.labels = torch.cat([self.labels, new_labels], dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        # import ipdb; ipdb.set_trace()
        new_labels = self.labels[selected_pts_mask].repeat(N)

        self.densification_postfix(new_labels, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # import ipdb; ipdb.set_trace()
        new_labels = self.labels[selected_pts_mask]

        self.densification_postfix(new_labels, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, gaussians_num, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size and gaussians_num>200000:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def coarse_deform_c2source(self, viewpoint_camera, query_pts, params, t_params, t_vertices, lbs_weights=None, correct_Rs=None, return_transl=False):
        # import ipdb; ipdb.set_trace()
        # print("1_query_pts:{}".format(query_pts))
        # print("1_query_pts.shape:{}".format(query_pts.shape))

        # project3D_2_2D(query_pts, viewpoint_camera, '0_1') 
        
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = t_vertices.shape[1]

        # Find nearest smpl vertex, all under world space
        smpl_pts = t_vertices
        _, vert_ids = self.knn(smpl_pts.float(), query_pts.float())

        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)#.cuda() # [bs, points_num, joints_num]
        else:
            # ensure the weights remain normalized and non-negative
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
            bweights = torch.log(bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        ### From Big To T Pose
        big_pose_params = t_params
        # TODO: find what A means?
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        # import ipdb; ipdb.set_trace()
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        # transforms from Big To T Pose
        transforms = R_inv

        # translation from Big To T Pose
        translation = None
        if return_transl: 
            translation = -A[..., :3, 3]
            translation = torch.matmul(R_inv, translation[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = big_pose_params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts - pose_offsets

            if return_transl: 
                translation -= pose_offsets

            # From mean shape to normal shape
            shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]#.cuda()
            shape_offset = torch.matmul(shapedirs.unsqueeze(0), torch.reshape(params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + shape_offset

            if return_transl: 
                translation += shape_offset

            # 从这里开始需要关注: From normal shape to target shape
            posedirs = self.SMPL_NEUTRAL['posedirs']#.cuda().float()
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])

            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, joints_num-1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

            # import ipdb; ipdb.set_trace()

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])#.cuda()
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1), posedirs.view(vertices_num*3, -1).transpose(1,0).unsqueeze(0)).view(batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3)) # [bs, N_rays*N_samples, 3]
            query_pts = query_pts + pose_offsets

            if return_transl: 
                translation += pose_offsets

        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)

        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]
        if return_transl: 
            translation_0 = torch.matmul(A[..., :3, :3], translation[..., None]).squeeze(-1)
            translation_1 = A[..., :3, 3]
        transforms = torch.matmul(A[..., :3, :3], transforms) 

        smpl_src_pts_1 = (smpl_src_pts + torch.tensor(params['Th'].reshape((1, 3))).cuda()).squeeze(0)
        if return_transl: 
            translation_1 += torch.tensor(params['Th'].reshape((1, 3))).cuda()

        B = torch.tensor(np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)).cuda()
        smpl_src_pts_1 = torch.matmul(smpl_src_pts_1, B.T)
        if return_transl:
            translation_0 = torch.matmul(translation_0, B.T)
            translation_1 = torch.matmul(translation_1, B.T)
        transforms = torch.matmul(B, transforms)

        E = np.eye(4)
        E[:3, :3] = np.transpose(viewpoint_camera.R)
        E[:3, 3] = viewpoint_camera.T
        ctw = np.linalg.inv(E)
        ctw_R = torch.tensor(ctw[:3, :3].astype(np.float32)).cuda()
        ctw_T = torch.tensor(ctw[:3, 3].astype(np.float32)).cuda()
        # import ipdb; ipdb.set_trace()
        smpl_src_pts_1_wor = torch.matmul(smpl_src_pts_1, ctw_R.T) + ctw_T
        if return_transl:
            translation_0 = torch.matmul(translation_0, ctw_R.T)
            translation_1 = torch.matmul(translation_1, ctw_R.T)
            translation_2 = ctw_T
            translation = translation_0 + translation_1 + translation_2
        transforms = torch.matmul(ctw_R, transforms)


        R_trans = torch.transpose(R, 0, 1)
        world_src_pts = torch.matmul(smpl_src_pts_1_wor, R_trans) + Th

        return smpl_src_pts_1_wor, world_src_pts, bweights, transforms, translation


def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat


def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3; joints 位置
    parents: 24; 父节点
    """
    # obtain the relative joints
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    # create the transformation matrix
    # transforms_mat: bs x 24 x 4 x 4
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)  #.to(rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    # rotate each part
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    # obtain the rigid transformation
    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)  #.to(rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """

    v_template = smpl['v_template']

    # add shape blend shapes
    shapedirs = smpl['shapedirs']
    betas = params['shapes']

    # computes the vertex positions adjusted for individual body shape. 
    v_shaped = v_template[None] + torch.sum(shapedirs[None][...,:betas.shape[-1]] * betas[:,None], axis=-1).float()

    # import ipdb; ipdb.set_trace()

    if rot_mats is None:
        # add pose blend shapes
        poses = params['poses'].reshape(-1, 3)
        # bs x 16 x 3 x 3
        # including the wrist rotation, root
        rot_mats = batch_rodrigues_torch(poses).view(params['poses'].shape[0], -1, 3, 3)

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, rot_mats.shape[1]-1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

    # obtain the joints
    # Calculates the positions of the joints based on the shaped vertex positions (v_shaped)
    joints = torch.matmul(smpl['J_regressor'][None], v_shaped) # [bs, 24 ,3]

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    # apply global transformation
    R = params['R'] 
    Th = params['Th'] 

    return A, R, Th, joints

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params