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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix_refine


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, FoVx, FoVy, image, image_hand, image_obj, hand_mask, obj_mask, gt_alpha_mask, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, depth=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image_hand = image_hand.clamp(0.0, 1.0).to(self.data_device)
        self.original_image_obj = image_obj.clamp(0.0, 1.0).to(self.data_device)
        self.hand_mask = hand_mask.clamp(0.0, 1.0).to(self.data_device)
        self.obj_mask = obj_mask.clamp(0.0, 1.0).to(self.data_device)
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(
            self.data_device)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
        #                                              fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix_refine(K=torch.Tensor(K).cuda(), H=self.image_height, 
                                                        W=self.image_width, znear=self.znear, zfar=self.zfar).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)


class HOICamera(nn.Module):
    def __init__(self, colmap_id, pose_id, R, T, K, FoVx, FoVy, image, image_hand, image_obj, hand_mask, obj_mask, gt_alpha_mask,
                 image_path, image_name, uid, fid,
                 bkgd_mask=None, bound_mask=None, smpl_param=None, 
                 world_vertex=None, world_bound=None, big_pose_smpl_param=None,
                 big_pose_world_vertex=None, big_pose_world_bound=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", depth=None
                 ):
        super(HOICamera, self).__init__()

        self.uid = uid
        self.data_device = torch.device(data_device)
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.pose_id = pose_id
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.bkgd_mask = bkgd_mask
        self.bound_mask = bound_mask
        self.image_path = image_path

        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # print("Cameras.image: ", image)
        # np_image = image.cpu().numpy()
        # print("np.unique(np_image): ", np.unique(np_image))
        self.original_image = image.clamp(0.0, 1.0)#.to(self.data_device)
        self.original_image_hand = image_hand.clamp(0.0, 1.0)
        self.original_image_obj = image_obj.clamp(0.0, 1.0)
        self.hand_mask = hand_mask.clamp(0.0, 1.0).to(self.data_device)
        self.obj_mask = obj_mask.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask#.to(self.data_device)
            self.original_image_hand *= gt_alpha_mask
            self.original_image_obj *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width)) #torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.original_image_hand *= torch.ones((1, self.image_height, self.image_width))
            self.original_image_obj *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = 1000 #100.0
        self.znear = 0.001 #0.01

        self.trans = trans
        self.scale = scale
        # TODO: check world_view_transform
        # self.world_view_transform: wTc
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.projection_matrix = getProjectionMatrix_refine(torch.Tensor(K).cuda(), self.image_height, self.image_width, self.znear, self.zfar).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # print("self.camera_center: ", self.camera_center)

        self.smpl_param = smpl_to_cuda(smpl_param, self.data_device)
        self.world_vertex = torch.tensor(world_vertex).to(self.data_device)
        self.world_bound = torch.tensor(world_bound).to(self.data_device)
        self.big_pose_smpl_param = smpl_to_cuda(big_pose_smpl_param, self.data_device)
        self.big_pose_world_vertex = torch.tensor(big_pose_world_vertex).to(self.data_device)
        self.big_pose_world_bound = torch.tensor(big_pose_world_bound).to(self.data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def smpl_to_cuda(param, device):
    for key in param:
        if torch.is_tensor(param[key]):
            param[key] = param[key].to(device)
        else:
            param[key] = torch.Tensor(param[key]).to(device)
    return param