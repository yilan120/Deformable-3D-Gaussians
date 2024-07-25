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
import os
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(iteration, case, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None, 
           transforms=None, translation=None, return_smpl_rot=False, sem=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # print(pc._xyz.shape)
    # print("pc.get_xyz.shape:{}".format(pc.get_xyz.shape))
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    labels = pc.get_labels

    hand_means3D = means3D[labels == 1]
    obj_means3D = means3D[labels == 0]
    indices_of_hand = torch.nonzero(labels == 1).squeeze()
    indices_of_obj = torch.nonzero(labels == 0).squeeze()

    if pc.motion_flag:
        if not pc.motion_offset_flag:
            _, means3D, _, transforms, _ = pc.coarse_deform_c2source(viewpoint_camera, hand_means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None])
        else:
            if transforms is None:
                # import ipdb; ipdb.set_trace()
                dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
                pose_out = pc.pose_decoder(dst_posevec)
                correct_Rs = pose_out['Rs']

                lbs_weights = pc.lweight_offset_decoder(hand_means3D[None].detach())
                lbs_weights = lbs_weights.permute(0,2,1)

                s_means3D, w_means3D, _, transforms, translation = pc.coarse_deform_c2source(viewpoint_camera, hand_means3D[None], viewpoint_camera.smpl_param,
                    viewpoint_camera.big_pose_smpl_param,
                    viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot)
                hand_means3D = s_means3D
            else:
                correct_Rs = None
                hand_means3D = torch.matmul(transforms, hand_means3D[..., None]).squeeze(-1) + translation

    # TODO
    # 是不是可以把hand的参数也加上一个小的offset
    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            obj_means3D = obj_means3D
        else:
            obj_means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        if isinstance(d_xyz, torch.Tensor):
            # print("obj_means3D.shape:{}".format(obj_means3D.shape))
            # print("d_xyz.shape:{}".format(d_xyz.shape))
            if case == "Object":
                obj_means3D = obj_means3D + d_xyz[indices_of_obj]
            elif case == "None":
                obj_means3D = obj_means3D + d_xyz
            elif case == "Hand-obj":
                hand_means3D = hand_means3D + d_xyz[indices_of_hand]
                obj_means3D = obj_means3D + d_xyz[indices_of_obj]
            # hand_means3D = hand_means3D + d_xyz[indices_of_hand]
            # [indices_of_obj]
        elif isinstance(d_xyz, float):
            obj_means3D = obj_means3D + d_xyz
    
    means3D = torch.cat([hand_means3D, obj_means3D], dim=0)

    if iteration % 1000 == 0:
        # print("hand_means3D_depth_min:{}".format(hand_means3D[:, 2].min()))
        # print("hand_means3D_depth_max:{}".format(hand_means3D[:, 2].max()))
        # print("hand_means3D_depth_mean:{}".format(hand_means3D[:, 2].mean()))
        # print("obj_means3D_depth_min:{}".format(obj_means3D[:, 2].min()))
        # print("obj_means3D_depth_max:{}".format(obj_means3D[:, 2].max()))
        # print("obj_means3D_depth_mean:{}".format(obj_means3D[:, 2].mean()))
        point_cloud_path = "/data/home/acw773/Deformable-3D-Gaussians/output/ho3d/nomutual-save-rot/point_cloud_deformed/iteration_{}".format(iteration)
        pc.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # import ipdb; ipdb.set_trace()
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling + d_scaling
        # rotations = pc.get_rotation + d_rotation

        scales = pc.get_scaling
        rotations = pc.get_rotation
        if case == "Object":
            new_scales = scales.clone()
            new_rotations = rotations.clone()
            if isinstance(d_xyz, torch.Tensor):
                new_scales[indices_of_obj] = new_scales[indices_of_obj] + d_scaling[indices_of_obj]
                new_rotations[indices_of_obj] = new_rotations[indices_of_obj] + d_rotation[indices_of_obj]
            else:
                new_scales[indices_of_obj] = new_scales[indices_of_obj] + d_scaling
                new_rotations[indices_of_obj] = new_rotations[indices_of_obj] + d_rotation

            scales = new_scales
            rotations = new_rotations


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    if sem:
        rendered_image_hand, radii_hand, depth_hand = rasterizer(
            means3D=means3D[indices_of_hand],
            means2D=screenspace_points[indices_of_hand],
            means2D_densify=screenspace_points_densify,
            shs=shs[indices_of_hand],
            colors_precomp=colors_precomp,
            opacities=opacity[indices_of_hand],
            scales=scales[indices_of_hand],
            rotations=rotations[indices_of_hand],
            cov3D_precomp=cov3D_precomp)
        
        rendered_image_obj, radii_obj, depth_obj = rasterizer(
            means3D=means3D[indices_of_obj],
            means2D=screenspace_points[indices_of_obj],
            means2D_densify=screenspace_points_densify,
            shs=shs[indices_of_obj],
            colors_precomp=colors_precomp,
            opacities=opacity[indices_of_obj],
            scales=scales[indices_of_obj],
            rotations=rotations[indices_of_obj],
            cov3D_precomp=cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if not sem:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "viewspace_points_densify": screenspace_points_densify,
                "visibility_filter": radii > 0,
                "radii": radii,
                "transforms": transforms,
                "translation": translation,
                "correct_Rs": correct_Rs,
                "depth": depth,}
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "viewspace_points_densify": screenspace_points_densify,
                "visibility_filter": radii > 0,
                "radii": radii,
                "transforms": transforms,
                "translation": translation,
                "correct_Rs": correct_Rs,
                "depth": depth,
                "render_hand": rendered_image_hand,
                "radii_hand": radii_hand,
                "depth_hand": depth_hand,
                "render_obj": rendered_image_obj,
                "radii_obj": radii_obj,
                "depth_obj": depth_obj}
    
