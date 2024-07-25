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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration, mkdir_p
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
# , GaussianModel_HOI
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_HOIcamInfos


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "hand_obj_full")):
            scene_info = sceneLoadTypeCallbacks["HOI"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # train, 会进入到这里
            # TODO
            if isinstance(scene_info.ply_path, list): 
                with open(scene_info.ply_path[0], 'rb') as src_file, open(os.path.join(self.model_path, "input_hand.ply"),
                                                                    'wb') as dest_file:
                    dest_file.write(src_file.read())
                with open(scene_info.ply_path[1], 'rb') as src_file, open(os.path.join(self.model_path, "input_obj.ply"),
                                                                    'wb') as dest_file:
                    dest_file.write(src_file.read())
            elif isinstance(scene_info.ply_path, str):
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                    'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if os.path.exists(os.path.join(args.source_path, "hand_obj_full")):
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_HOIcamInfos(scene_info.train_cameras, resolution_scale,
                                                                                args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_HOIcamInfos(scene_info.test_cameras, resolution_scale,
                                                                            args)
            else:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                                args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                            args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save_deformed(self, save_name, iteration, viewpoint_cam, gaussians):

        point_cloud_deformed_path = os.path.join(self.model_path, "point_cloud_deformed/iteration_{}".format(iteration))
        self.gaussians.save_deformed_ply(save_name, iteration, os.path.join(point_cloud_deformed_path, "point_cloud_deformed.ply"),
                                         viewpoint_cam, gaussians)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(iteration), "ckpt.pth")
            mkdir_p(os.path.dirname(model_path))
            torch.save({
                'iter': iteration,
                'pose_decoder': self.gaussians.pose_decoder.state_dict(),
                'lweight_offset_decoder': self.gaussians.lweight_offset_decoder.state_dict(),
            }, model_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
