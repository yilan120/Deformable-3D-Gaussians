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


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_excluding_mask(original_image, rendered_image, mask):
    # import ipdb; ipdb.set_trace()
    # Ensure mask is a binary mask
    mask = mask > 0.5
    mask = mask.expand_as(original_image)
    
    # Extract the unmasked regions
    original_valid = original_image[mask]
    rendered_valid = rendered_image[mask]

    mse = torch.mean((original_valid - rendered_valid) ** 2)

    # mse = (((original_valid - rendered_valid)) ** 2).view(original_valid.shape[0], -1).mean(1, keepdim=True)
    
    # Calculate PSNR
    # PIXEL_MAX = 255.0
    # psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

