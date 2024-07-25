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

from scene.cameras import Camera, HOICamera
import numpy as np
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
import json

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, K=cam_info.K, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  depth=cam_info.depth)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def loadHOICam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_rgb_hand = PILtoTorch(cam_info.image_hand, resolution)
    resized_image_rgb_obj = PILtoTorch(cam_info.image_obj, resolution)
    # resized_mask_hand = PILtoTorch(cam_info.hand_mask, resolution)
    # resized_mask_obj = PILtoTorch(cam_info.obj_mask, resolution)

    gt_image = resized_image_rgb[:3, ...]
    gt_image_hand = resized_image_rgb_hand[:3, ...]
    gt_image_obj = resized_image_rgb_obj[:3, ...]
    # import ipdb; ipdb.set_trace()
    gt_mask_hand = cam_info.hand_mask[0, ...]
    gt_mask_obj = cam_info.obj_mask[0, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.bound_mask is not None:
        resized_bound_mask = PILtoTorch(cam_info.bound_mask, resolution)
    else:
        resized_bound_mask = None

    if cam_info.bkgd_mask is not None:
        resized_bkgd_mask = PILtoTorch(cam_info.bkgd_mask, resolution)
    else:
        resized_bkgd_mask = None

    return HOICamera(colmap_id=cam_info.uid, pose_id=cam_info.pose_id, R=cam_info.R, T=cam_info.T, K=cam_info.K, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, image_hand=gt_image_hand, image_obj=gt_image_obj, hand_mask=gt_mask_hand, obj_mask=gt_mask_obj, gt_alpha_mask=loaded_mask,
                  image_path=cam_info.image_path, image_name=cam_info.image_name, uid=id, bkgd_mask=resized_bkgd_mask, 
                  bound_mask=resized_bound_mask, smpl_param=cam_info.smpl_param, 
                  world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound, 
                  big_pose_smpl_param=cam_info.big_pose_smpl_param, 
                  big_pose_world_vertex=cam_info.big_pose_world_vertex, 
                  big_pose_world_bound=cam_info.big_pose_world_bound, 
                  data_device=args.data_device,
                  depth=cam_info.depth, fid=cam_info.fid)



def cameraList_from_HOIcamInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadHOICam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )
