import numpy as np
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
import sys
import math
import torch
import cv2
import imageio
import os
from plyfile import PlyData, PlyElement
from utils.system_utils import mkdir_p



def hand_construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


def save_hand_ply(xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation, path):
        mkdir_p(os.path.dirname(path))

        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # import ipdb; ipdb.set_trace()
        f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_dc = _features_dc
        # print("f_dc.shape:{}".format(f_dc.shape))
        # f_rest = _features_rest
        # print("f_rest.shape:{}".format(f_rest.shape))
        opacities = _opacity.detach().cpu().numpy()
        scale = _scaling.detach().cpu().numpy()
        rotation = _rotation.detach().cpu().numpy()

        l_list = hand_construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation)

        dtype_full = [(attribute, 'f4') for attribute in l_list]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


MANO_MODEL_PATH = '/data/home/acw773/GauHuman/mano/models/MANO_RIGHT.pkl'

def project3D_2_2D(means3D, viewpoint_camera, flag):

    image_height, image_width = 480, 640
    background_color = (255, 255, 255)

    gt_image = viewpoint_camera.original_image
    # print("gt_image.shape: ", gt_image.shape)
    test_image = (gt_image.clone().permute(1,2,0)*255).numpy().astype(np.uint8)
    # print("test_image.shape: ", test_image.shape)
    # imageio.imwrite('test_image.png', test_image)
    

    # print("!!!!!!means3D: ", means3D.shape)

    points_3D = means3D.cpu().detach().numpy()
    if len(points_3D.shape) == 3:
        points_3D = points_3D[0]
        
    E = np.hstack((np.transpose(viewpoint_camera.R), viewpoint_camera.T.reshape(3,1)))
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
    K = viewpoint_camera.K
    
    points_2D_hom_cam = np.dot(points_3D_hom, E.T)
    points_2D_hom_wor = np.dot(points_2D_hom_cam, K.T)
    points_2D = points_2D_hom_wor[:, :2] / points_2D_hom_wor[:, 2].reshape(-1, 1)
    # print(points_2D)

    # # Check if any points are out of bounds
    # x_out_of_bounds = np.any((points_2D[:, 0] < 0) | (points_2D[:, 0] >= 640))
    # y_out_of_bounds = np.any((points_2D[:, 1] < 0) | (points_2D[:, 1] >= 480))
    # if x_out_of_bounds or y_out_of_bounds:
    #     print("Some points are out of the image boundaries.")
    # else:
    #     print("All points are within the image boundaries.")


    # Create an empty image
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    # Draw each point on the image
    for (x, y) in points_2D:
        cv2.circle(test_image, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)  # Red points
    cv2.imwrite('projected_points_{}.png'.format(flag), test_image)
    # print("write to image: projected_points_{}.png".format(flag))


    return 

def apply_global_tfm_to_camera(E, Rh, Th, B):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )
        
    Returns:
        - Array (3, 3)
    """

    # E: camera to world
    # Rh: hand rotation hand to camera
    # Th: hand translation hand to camera

    global_tfms = np.eye(4)  #(4, 4)

    # global_tfms A : hand to camera
    global_rot_OpenGL = cv2.Rodrigues(Rh)[0]
    # global_rot = np.matmul(global_rot_OpenGL, B.T)
    global_trans = Th
    global_tfms[:3, :3] = global_rot_OpenGL
    global_tfms[:3, 3] = global_trans

    # B_4x4 = np.eye(4, dtype=np.float32)  # Create a 4x4 identity matrix
    # B_4x4[:3, :3] = B 
    # hTc = np.dot(global_tfms, B_4x4)
    # hTw = np.dot(E, hTc)
    # np.linalg.inv(global_tfms), A-1 : hand to camera
    # return hTw
    return global_tfms



def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = imgInOrg.cpu().numpy().copy()

    cv2.imwrite('filename.png', imgIn)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]
    
    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)
    print("gtIn: ", gtIn.shape)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):
            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn



def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera intrisic matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    # pts3D_np = pts3D.numpy()

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    coord_change_mat_tensor = torch.tensor(coord_change_mat, dtype=torch.float32).double().cuda()

    if is_OpenGL_coords:
        pts3D = pts3D.matmul(coord_change_mat_tensor.T).cpu().numpy()
        # pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts


def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts, mesh
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''

    fullpose = fullpose.reshape(48,)
    trans = trans.reshape(3,)
    beta = beta.reshape(10,)

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta

    return m.J_transformed.r, m







def apply_external_transformations(v_template, J_regressor, betas, pose, trans, shapedirs, posedirs):
    # Recreate the transformations as they are applied internally within the MANO model
    v_shaped = v_template + shapedirs.dot(betas)
    J = J_regressor.dot(v_shaped)  # Calculate joints

    # Pose-dependent vertex offsets (posedirs are typically applied after shape adjustments)
    v_posed = v_shaped + posedirs.dot(pose)

    # Apply rotation and translation
    rotation_matrix, _ = cv2.Rodrigues(pose[:3])  # Assuming pose[:3] contains the rotation in axis-angle format
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = trans

    # Apply the transformation to the posed vertices
    v_posed_homogeneous = np.hstack([v_posed, np.ones((v_posed.shape[0], 1))])
    v_transformed = (transformation_matrix @ v_posed_homogeneous.T).T[:, :3]

    return v_transformed









import numpy as np

def transform_point(point, matrix):
    """ Transform a 3D point using a 4x4 transformation matrix """
    x, y, z = point
    transformed = matrix @ np.array([x, y, z, 1])
    return transformed[:3] / transformed[3]

def in_frustum(point, view_matrix, proj_matrix, flag):
    """ Check if a point is within the view frustum """
    # Transform the point to camera coordinates
    p_view = transform_point(point, view_matrix)
    # print("p_view[2]_{}:{}".format(flag, p_view[2]))
    
    # Check if the point is in front of the camera
    if p_view[2] < 0.2:
        return False
    
    # Transform the point to clip space
    p_clip = transform_point(p_view, proj_matrix)
    
    # Check clip space bounds
    if np.any(p_clip < -1) or np.any(p_clip > 1):
        return False
    
    return True

def mark_visible(points3D, view_matrix, proj_matrix, flag):
    """ Mark points as visible based on the view frustum """
    visible = np.array([in_frustum(pt, view_matrix, proj_matrix, flag) for pt in points3D])
    return visible

# Define the view and projection matrices (example placeholders)
# view_matrix = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, -5],
#     [0, 0, 0, 1]
# ])
# proj_matrix = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, -1, -0.2],
#     [0, 0, -1, 0]
# ])

# Example large array of points in 3D
# points3D = np.random.randn(778, 3)  # Generate random points for illustration

# Get visibility of points
# visibility = mark_visible(points3D, view_matrix, proj_matrix)
# print("Visibility:", visibility)
