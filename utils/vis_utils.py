from plyfile import PlyData, PlyElement
import numpy as np
import torch


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def sort_smpl_rot(smpl_rot_unsorted):
    for name in smpl_rot_unsorted.keys():  # Iterate over 'train' and 'test'
        for pose_id in smpl_rot_unsorted[name].keys():  # Iterate over pose IDs (assuming only one)
            for key in ['transforms', 'translation']:
                # Retrieve and sort the keys
                sorted_keys = sorted(smpl_rot_unsorted[name][pose_id][key].keys())
                if name == 'train':
                    t_list = sorted_keys
                # import ipdb; ipdb.set_trace()
                # Create a new ordered dictionary with the sorted keys
                sorted_dict = {k: smpl_rot_unsorted[name][pose_id][key][k] for k in sorted_keys}
                # Assign the sorted dictionary back to the original structure
                smpl_rot_unsorted[name][pose_id][key] = sorted_dict
    return smpl_rot_unsorted, t_list

def get_deformed_data(means3D, smpl_rot, t):
    # import ipdb; ipdb.set_trace()
    # means3D = scene_data['means3D']
    name = 'train'
    pose_id = 1
    transforms, translation = smpl_rot[name][pose_id]['transforms'][t], smpl_rot[name][pose_id]['translation'][t]
    deformed_means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation
    # scene_data['means3D'] = means3D.squeeze()
    return deformed_means3D.squeeze()


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result



def load_ply(path, smpl_rot, t_list, w2c):
    max_sh_degree = 3

    plydata = PlyData.read(path)

    # import ipdb; ipdb.set_trace()

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))

    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    scene_data = []

    t_num = len(smpl_rot['train'][1]['transforms'])
    for t in range(t_num):
        # print("t_list[t]:{}".format(t_list[t]))
        # import ipdb; ipdb.set_trace()
        deformed_xyz = get_deformed_data(torch.tensor(xyz, dtype=torch.float, device="cuda"), smpl_rot, t_list[t])

        # import ipdb; ipdb.set_trace()
        _xyz = deformed_xyz
        _features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        _features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        _opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        # 这个scale是经过log的，与dynaimicGS一样
        _scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        _rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        _shs = torch.cat((_features_dc, _features_rest), dim=1)

        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        camera_center = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda().inverse()[3, :3]

        shs_view = _shs.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
        dir_pp = (_xyz - camera_center.repeat(_shs.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(max_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        rendervar = {
                'means3D': _xyz,
                'colors_precomp': colors_precomp,
                'rotations': torch.nn.functional.normalize(_rotation),
                'opacities': torch.sigmoid(_opacity),
                # 需要加，见gaussian_model.py line 44, activation function is torch.exp
                # and get-scaling will call activation function
                'scales': torch.exp(_scaling),
                'means2D': torch.zeros_like(torch.empty(t_num, 3)).cuda(),
                # 'shs': _shs,
                # '_features_dc': _features_dc,
                # '_features_rest': _features_rest,
            }
        scene_data.append(rendervar)
    active_sh_degree = max_sh_degree
    return scene_data, active_sh_degree