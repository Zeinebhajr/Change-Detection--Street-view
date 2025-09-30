import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pytorch3d.renderer import AlphaCompositor, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, PointsRenderer
from pytorch3d.structures import Pointclouds

import modules.geometry
import numpy as np
import os
from modules.geometry import plot_mask_warping_comparison
from utils import save_occlusion_masks
class FeatureRegisterationModule(nn.Module):
    def __init__(self, args,intraclasse):
        super().__init__()
        self.args = args
        self.feature_warper = DifferentiableFeatureWarper()
        self.intraclasse=intraclasse
    def register_3d_features(
        self,
        batch,
        features1,
        features2,
    ):
        i=0
        using_camera_parameters = [x is not None for x in batch["intrinsics1"]]
        K_inv_1 = torch.zeros(len(batch["intrinsics1"]), 3, 3).type_as(features1)
        K_inv_2 = torch.zeros(len(batch["intrinsics2"]), 3, 3).type_as(features2)
        Rt_1_to_2 = torch.zeros(len(batch["intrinsics1"]), 4, 4).type_as(features1)
        Rt_2_to_1 = torch.zeros(len(batch["intrinsics2"]), 4, 4).type_as(features2)
        if sum(using_camera_parameters) > 0:
            _K_inv_1, _K_inv_2, _Rt_1_to_2, _Rt_2_to_1 = estimate_Rt_using_camera_parameters(
                torch.stack([batch["intrinsics1"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["intrinsics2"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["rotation1"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["rotation2"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["position1"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["position2"][i] for i, x in enumerate(using_camera_parameters) if x]),
            )
            K_inv_1[using_camera_parameters] = _K_inv_1
            K_inv_2[using_camera_parameters] = _K_inv_2
            Rt_1_to_2[using_camera_parameters] = _Rt_1_to_2
            Rt_2_to_1[using_camera_parameters] = _Rt_2_to_1
            os.makedirs('demo_data/examples_tranformations',exist_ok=True)
            i+=1
            np.save(os.path.join('demo_data/tranformations',f'transformation1-2_{i}_cameraparams.npy'),Rt_1_to_2)
            np.save(os.path.join('demo_data/tranformations',f'transformation2-1_{i}_cameraparams.npy'),Rt_2_to_1)

        using_points = [x is None for x in batch["intrinsics1"]]
        if sum(using_points) > 0:
            _K_inv_1, _K_inv_2, _Rt_1_to_2, _Rt_2_to_1 = estimate_Rt_using_points(
                [batch["points1"][i] for i, x in enumerate(using_points) if x],
                [batch["points2"][i] for i, x in enumerate(using_points) if x],
                batch["depth1"][using_points],
                batch["depth2"][using_points],
                [batch['focale1'][i] for i, x in enumerate(using_points) if x],
                [batch['focale2'][i] for i, x in enumerate(using_points) if x],
            )

            K_inv_1[using_points] = _K_inv_1
            K_inv_2[using_points] = _K_inv_2
            Rt_1_to_2[using_points] = _Rt_1_to_2
            Rt_2_to_1[using_points] = _Rt_2_to_1
            os.makedirs('demo_data/tranformations',exist_ok=True)
            i+=1
            np.save(os.path.join('demo_data/tranformations',f'transformation1to2_{i}_usingpoints.npy'),Rt_1_to_2)
            np.save(os.path.join('demo_data/tranformations',f'transformation2to1_{i}_usingpoints.npy'),Rt_2_to_1)
        #sauvegarder les matrices de transformations:
        nearest_resize = K.augmentation.Resize(features1.shape[-2:], resample=0, align_corners=None, keepdim=True)
        depth1 = nearest_resize(batch["depth1"])
        depth2 = nearest_resize(batch["depth2"])
        print(type(depth1),depth1.shape)
        image1_warped_onto_image2_plusmask = self.feature_warper.warp_with_mask(features1, depth1,batch['mask1'], K_inv_1, K_inv_2, Rt_1_to_2)
        image2_warped_onto_image1_plusmask = self.feature_warper.warp_with_mask(features2, depth2,batch['mask2'], K_inv_2, K_inv_1, Rt_2_to_1)
        image1_warped_onto_image2 = image1_warped_onto_image2_plusmask[:, :-2, :, :]  # features originales
        image2_warped_onto_image1 = image2_warped_onto_image1_plusmask[:, :-2, :, :] 
        
        mask1_warped_onto_mask2 = image1_warped_onto_image2_plusmask[:, -2:-1, :, :]  # masque de classes
        mask2_warped_onto_mask1 = image2_warped_onto_image1_plusmask[:, -2:-1, :, :]
        
        depth1_warped_onto_depth2 = image1_warped_onto_image2_plusmask[:, -1:, :, :]   # depth warpée
        depth2_warped_onto_depth1 = image2_warped_onto_image1_plusmask[:, -1:, :, :]

        def transform_points_1_to_2(points, index_in_batch):
            points = points.unsqueeze(0)
            return modules.geometry.convert_world_to_image_coordinates(
                modules.geometry.convert_image_coordinates_to_world(
                    image_coords=points,
                    depth=modules.geometry.sample_depth_for_given_points(batch["depth1"][index_in_batch].unsqueeze(0), points),
                    K_inv=K_inv_1[index_in_batch].unsqueeze(0),
                    Rt=Rt_1_to_2[index_in_batch].unsqueeze(0),
                ),
                K_inv_2[index_in_batch].unsqueeze(0),
                torch.eye(4).unsqueeze(0).repeat(1, 1, 1).type_as(Rt_1_to_2),
                keep_depth=False,
            )[0]

        def transform_points_2_to_1(points, index_in_batch):
            points = points.unsqueeze(0)
            return modules.geometry.convert_world_to_image_coordinates(
                modules.geometry.convert_image_coordinates_to_world(
                    image_coords=points,
                    depth=modules.geometry.sample_depth_for_given_points(batch["depth2"][index_in_batch].unsqueeze(0), points),
                    K_inv=K_inv_2[index_in_batch].unsqueeze(0),
                    Rt=Rt_2_to_1[index_in_batch].unsqueeze(0),
                ),
                K_inv_1[index_in_batch].unsqueeze(0),
                torch.eye(4).unsqueeze(0).repeat(1, 1, 1).type_as(Rt_1_to_2),
                keep_depth=False,
            )[0]

        return (image1_warped_onto_image2, image2_warped_onto_image1, 
        mask1_warped_onto_mask2, mask2_warped_onto_mask1,
        depth1_warped_onto_depth2, depth2_warped_onto_depth1,
        transform_points_1_to_2, transform_points_2_to_1)

    def register_2d_features(self, batch, features1, features2):
        M_1_to_2 = []
        M_2_to_1 = []
        for i, (p1, p2) in enumerate(zip(batch["points1"], batch["points2"])):
            if p1 is not None:
                p1, p2 = p1.unsqueeze(0), p2.unsqueeze(0)
                M_1_to_2.append(modules.geometry.estimate_linear_warp(p1, p2).squeeze(0))
                M_2_to_1.append(modules.geometry.estimate_linear_warp(p2, p1).squeeze(0))
            else:
                M_1_to_2.append(batch["transfm2d_1_to_2"][i])
                M_2_to_1.append(batch["transfm2d_2_to_1"][i])
        
        M_1_to_2 = torch.stack(M_1_to_2)
        M_2_to_1 = torch.stack(M_2_to_1)

        b, _, h, w = features1.shape
        image_coords = rearrange(
            modules.geometry.get_index_grid(h, w, batch=b, type_as=features1),
            "b h w t -> b (h w) t",
        )
        image1_points_warped = modules.geometry.transform_points(M_1_to_2, image_coords, keep_depth=True)
        image2_points_warped = modules.geometry.transform_points(M_2_to_1, image_coords, keep_depth=True)
        image1_warped_onto_image2 = self.feature_warper.render_features_from_points(image1_points_warped, features1)
        image2_warped_onto_image1 = self.feature_warper.render_features_from_points(image2_points_warped, features2)

        def transform_points_1_to_2(points, index_in_batch):
            points = points.unsqueeze(0)
            return modules.geometry.transform_points(M_1_to_2[index_in_batch].unsqueeze(0), points, keep_depth=False)[0]

        def transform_points_2_to_1(points, index_in_batch):
            points = points.unsqueeze(0)
            return modules.geometry.transform_points(M_2_to_1[index_in_batch].unsqueeze(0), points, keep_depth=False)[0]

        return image1_warped_onto_image2, image2_warped_onto_image1, transform_points_1_to_2, transform_points_2_to_1

    def register_identity_features(self, batch, features1, features2):
        b, _, h, w = features1.shape
        image_coords = rearrange(
            modules.geometry.get_index_grid(h, w, batch=b, type_as=features1),
            "b h w t -> b (h w) t",
        )
        image_coords = F.pad(image_coords, (0, 1), value=1)
        image1_warped_onto_image2 = self.feature_warper.render_features_from_points(image_coords, features1)
        image2_warped_onto_image1 = self.feature_warper.render_features_from_points(image_coords, features2)

        def transform_points(points, index_in_batch):
            return points

        return image1_warped_onto_image2, image2_warped_onto_image1, transform_points, transform_points 

    def register_features(self, batch, image1, image2, strategy):
        if len(image1) == 0:
            return [], [], None, None

        b, c, h, w = image1.shape
        print(b, c, h, w)

        if strategy == "3d":
            visibility = torch.ones((b, 1, h, w), requires_grad=False).type_as(image1)
            image1_warped_onto_image2, image2_warped_onto_image1, mask1_warped_onto_mask2, mask2_warped_onto_mask1, d1_to2, d2_to1, trasform_points_1_to_2, transform_points_2_to_1 = self.register_3d_features(
                batch, torch.cat([image1, visibility], dim=1), torch.cat([image2, visibility], dim=1)
            )
            
            print('9bal l9assan', image2_warped_onto_image1.shape)
            
            # Extract visibility channels
            visibility1 = image1_warped_onto_image2[:, -1:, :, :]
            visibility2 = image2_warped_onto_image1[:, -1:, :, :]
            visibility_orig1=visibility1
            visibility_orig2=visibility2
            image1_warped_onto_image2_m = image1_warped_onto_image2[:, :-1, :, :]
            image2_warped_onto_image1_m = image2_warped_onto_image1[:, :-1, :, :]
            
            # Setup resize operation
            nearest_resize = K.augmentation.Resize(image1.shape[-2:], resample=0, align_corners=None, keepdim=True)
            
            # Process masks
            mask1 = batch["mask1"].float()  # [b, h, w]
            mask2 = batch["mask2"].float()  # [b, h, w]
            mask1_channel = mask1.unsqueeze(1)  # [b, 1, h, w]
            mask2_channel = mask2.unsqueeze(1)
            mask2_c = nearest_resize(mask2_channel)
            mask1_c = nearest_resize(mask1_channel)
            inv_mask2 = (1 - mask2_c).float()
            inv_mask1 = (1 - mask1_c).float()
            
            # Process depth maps
            depth1_orig = nearest_resize(batch["depth1"])  # This gives [b, h, w]
            depth2_orig = nearest_resize(batch["depth2"])  # This gives [b, h, w]
            
            # Convert depth to [b, 1, h, w] to match d1_to2 and d2_to1 dimensions
            depth1_orig_4d = depth1_orig.unsqueeze(1)  # [b, 1, h, w]
            depth2_orig_4d = depth2_orig.unsqueeze(1)  # [b, 1, h, w]
            
            # Apply masking (both depth and mask are now [b, 1, h, w])
            depth1_orig_masked = depth1_orig_4d 
            depth2_orig_masked = depth2_orig_4d 
            # Compute occlusion masks (all tensors are now [b, 1, h, w])
            depth_threshold = 0.1
            occlusion_mask_1 = ((d1_to2 - depth2_orig_masked) > depth_threshold)
            occlusion_mask_2 = ((d2_to1 - depth1_orig_masked) > depth_threshold) 
            occlusion_mask_1_filtered = occlusion_mask_1 & (inv_mask2 > 0.5) 
            occlusion_mask_2_filtered = occlusion_mask_2 & (inv_mask1 > 0.5) 
            # Apply occlusion masks to visibility
            final_visibility1 = visibility1 * (~occlusion_mask_1_filtered).float()
            final_visibility2 = visibility2 * (~occlusion_mask_2_filtered).float()
            # Compute differences

            image1_diff = final_visibility2 * (image1 - image2_warped_onto_image1_m)
            image2_diff = final_visibility1 * (image2 - image1_warped_onto_image2_m)
            if self.intraclasse:
                mask1 = batch["mask1"].float()  # [b, h, w]
                mask2 = batch["mask2"].float()  # [b, h, w]
                mask1_channel = mask1.unsqueeze(1)  # [b, 1, h, w]
                mask2_channel = mask2.unsqueeze(1)
                mask1_warped = mask1_warped_onto_mask2 # [b, 1, h, w]
                mask2_warped = mask2_warped_onto_mask1
                mask2_c = F.interpolate(mask2_channel, size=(visibility2.shape[2], visibility2.shape[3]), mode='nearest')
                mask1_c = F.interpolate(mask1_channel, size=(visibility2.shape[2], visibility2.shape[3]), mode='nearest')
                im1_mask=visibility2*(mask1_c-mask2_warped)
                im2_mask=visibility1*(mask2_c-mask1_warped)
                image1=image1_diff*im1_mask
                image2=image2_diff*im2_mask
            else:
                image1=image1_diff
                image2=image2_diff
            pair_idx=0
            """ if (h,w)!=(64,64):
                print('hello')
            else:
                for batch_idx in range(b):
                # Créer la visualisation comparative
                    save_occlusion_masks(
                    occlusion_mask_1, occlusion_mask_2,
                    occlusion_mask_1_filtered, occlusion_mask_2_filtered,
                    combined_mask_1, combined_mask_2,
                    visibility_orig1,visibility_orig2,
                    visibility1,visibility2,
                    final_visibility1, final_visibility2,
                    d1_to2, d2_to1,
                    depth1_orig_masked, depth2_orig_masked, save_dir=f'OCCCLUSION3/feature_{h}x{w}',
                    batch_idx=batch_idx, pair_idx=pair_idx
                )
                    pair_idx+=1 """
        elif strategy == "2d":
            visibility = torch.ones((b, 1, h, w), requires_grad=False).type_as(image1)
            image1_warped_onto_image2, image2_warped_onto_image1, trasform_points_1_to_2, transform_points_2_to_1 = self.register_2d_features(
                batch, torch.cat([image1, visibility], dim=1), torch.cat([image2, visibility], dim=1)
            )
            visibility1 = image1_warped_onto_image2[:, -1:, :, :]
            visibility2 = image2_warped_onto_image1[:, -1:, :, :]
            image1_warped_onto_image2 = image1_warped_onto_image2[:, :-1, :, :]
            image2_warped_onto_image1 = image2_warped_onto_image1[:, :-1, :, :]
        elif strategy == "identity":
            image1_warped_onto_image2, image2_warped_onto_image1, trasform_points_1_to_2, transform_points_2_to_1 = self.register_identity_features(batch, image1, image2)
            visibility1 = torch.ones((b, 1, h, w), requires_grad=False).type_as(image1)
            visibility2 = torch.ones((b, 1, h, w), requires_grad=False).type_as(image1)
        return image1, image2, trasform_points_1_to_2, transform_points_2_to_1

    def forward(self, batch, image1, image2):
        reg_3d = [s == "3d" for s in batch["registration_strategy"]]
        reg_2d = [s == "2d" for s in batch["registration_strategy"]]
        reg_id = [s == "identity" for s in batch["registration_strategy"]]
        image1_3d, image2_3d, trasform_points_1_to_2_3d, transform_points_2_to_1_3d = self.register_features(slice_batch_given_bool_array(batch, reg_3d), image1[reg_3d], image2[reg_3d], "3d")
        image1_2d, image2_2d, trasform_points_1_to_2_2d, transform_points_2_to_1_2d = self.register_features(slice_batch_given_bool_array(batch, reg_2d), image1[reg_2d], image2[reg_2d], "2d")
        image1_id, image2_id, trasform_points_1_to_2_id, transform_points_2_to_1_id = self.register_features(slice_batch_given_bool_array(batch, reg_id), image1[reg_id], image2[reg_id], "identity")

        image1 = torch.zeros_like(image1)
        image2 = torch.zeros_like(image2)

        def transform_points_1_to_2(points, index_in_batch):
            if reg_3d[index_in_batch]:
                actual_index = 0
                for i in range(index_in_batch):
                    if reg_3d[i]:
                        actual_index += 1
                return trasform_points_1_to_2_3d(points, actual_index)
            elif reg_2d[index_in_batch]:
                actual_index = 0
                for i in range(index_in_batch):
                    if reg_2d[i]:
                        actual_index += 1
                return trasform_points_1_to_2_2d(points, actual_index)
            elif reg_id[index_in_batch]:
                actual_index = 0
                for i in range(index_in_batch):
                    if reg_id[i]:
                        actual_index += 1
                return trasform_points_1_to_2_id(points, actual_index)
        
        def transform_points_2_to_1(points, index_in_batch):
            if reg_3d[index_in_batch]:
                actual_index = 0
                for i in range(index_in_batch):
                    if reg_3d[i]:
                        actual_index += 1
                return transform_points_2_to_1_3d(points, actual_index)
            elif reg_2d[index_in_batch]:
                actual_index = 0
                for i in range(index_in_batch):
                    if reg_2d[i]:
                        actual_index += 1
                return transform_points_2_to_1_2d(points, actual_index)
            elif reg_id[index_in_batch]:
                actual_index = 0
                for i in range(index_in_batch):
                    if reg_id[i]:
                        actual_index += 1
                return transform_points_2_to_1_id(points, actual_index)

        if len(image1_3d) > 0:
            image1[reg_3d] = image1_3d
            image2[reg_3d] = image2_3d
        if len(image1_2d) > 0:
            image1[reg_2d] = image1_2d
            image2[reg_2d] = image2_2d
        if len(image1_id) > 0:
            image1[reg_id] = image1_id
            image2[reg_id] = image2_id

        batch["transform_points_1_to_2"] = transform_points_1_to_2
        batch["transform_points_2_to_1"] = transform_points_2_to_1

        return image1, image2


def estimate_Rt_using_camera_parameters(intrinsics1, intrinsics2, rotation1, rotation2, position1, position2):
    K_inv_1 = intrinsics1.inverse()
    K_inv_2 = intrinsics2.inverse()
    Rt_1_to_2, Rt_2_to_1 = modules.geometry.get_relative_pose(
        rotation1,
        rotation2,
        position1,
        position2,
        as_single_matrix=True,
    )
    return K_inv_1, K_inv_2, Rt_1_to_2, Rt_2_to_1

def estimate_Rt_using_points(points1, points2, depth1, depth2,foc1,foc2):
    print(foc1,foc2,type(foc1),type(foc2))
    K_inv, Rt = modules.geometry.setup_canonical_cameras(len(points1), tensor_to_infer_type_from=points1[0])
    if foc1!=None and foc2!=None:
        if isinstance(foc1, list):
            foc1 = torch.tensor(foc1, dtype=torch.float32)
        if isinstance(foc2, list):
            foc2 = torch.tensor(foc2, dtype=torch.float32)
        print(f"Focales reçues - foc1: {foc1}, foc2: {foc2}")
        print(f"Ratio f1/f2: {foc1/foc2}")
        print(f"Ratio f1/f2: {foc2/foc1}")
        # Calcul du ratio de focales (f1/f2)
        ratio_f1_over_f2 = foc1 / foc2
        b = len(points1)
        device_type = points1[0]
        
        # Matrices fixes pour les deux directions (cohérence garantie)
        # Caméra 1 = référence canonique (identité)
        K1_inv = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).type_as(device_type)
        
        # Caméra 2 = relative à caméra 1 (ratio f1/f2)
        K2_inv = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).type_as(device_type)
        K2_inv[:, 0, 0] = ratio_f1_over_f2  # f1/f2
        K2_inv[:, 1, 1] = ratio_f1_over_f2  # f1/f2
    else:
        K1_inv,K2_inv=K_inv,K_inv
    
    batch_points1_in_world_coordinates = []
    batch_points2_in_world_coordinates = []
    for i in range(len(points1)):
        points1_in_world_coordinates = modules.geometry.convert_image_coordinates_to_world(
            image_coords=points1[i].unsqueeze(0),
            depth=modules.geometry.sample_depth_for_given_points(depth1[i].unsqueeze(0), points1[i].unsqueeze(0)),
            K_inv=K1_inv[i].unsqueeze(0),
            Rt=Rt[i].unsqueeze(0),
        ).squeeze(0)
        points2_in_world_coordinates = modules.geometry.convert_image_coordinates_to_world(
            image_coords=points2[i].unsqueeze(0),
            depth=modules.geometry.sample_depth_for_given_points(depth2[i].unsqueeze(0), points2[i].unsqueeze(0)),
            K_inv=K2_inv[i].unsqueeze(0),
            Rt=Rt[i].unsqueeze(0),
        ).squeeze(0)
        batch_points1_in_world_coordinates.append(points1_in_world_coordinates)
        batch_points2_in_world_coordinates.append(points2_in_world_coordinates)
    Rt_1_to_2 = modules.geometry.estimate_linear_warp(batch_points1_in_world_coordinates, batch_points2_in_world_coordinates)
    Rt_2_to_1 = modules.geometry.estimate_linear_warp(batch_points2_in_world_coordinates, batch_points1_in_world_coordinates)
    return K1_inv, K2_inv, Rt_1_to_2, Rt_2_to_1

def slice_batch_given_bool_array(batch, mask):
    sliced_batch = {}
    for key in batch.keys():
        if "transform" in key:
            continue
        if isinstance(batch[key], list):
            sliced_batch[key] = [batch[key][i] for i in range(len(batch[key])) if mask[i]]
            if "bbox" in key or "point" in key:
                continue
            if len(sliced_batch[key]) > 0 and isinstance(sliced_batch[key][0], torch.Tensor):
                sliced_batch[key] = rearrange(sliced_batch[key], "... -> ...")
        else:
            sliced_batch[key] = batch[key][mask]
    return sliced_batch

class DifferentiableFeatureWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def render(self, point_cloud, device, image_hw):
        raster_settings = PointsRasterizationSettings(
            image_size=image_hw,
            radius=float(1.5) / min(image_hw) * 2.0,
            bin_size=0,
            points_per_pixel=8,
        )
        canonical_cameras = PerspectiveCameras(
            R=rearrange(torch.eye(3), "r c -> 1 r c"),
            T=rearrange(torch.zeros(3), "n -> 1 n"),
        )
        canonical_rasterizer = PointsRasterizer(cameras=canonical_cameras, raster_settings=raster_settings)
        canonical_renderer = PointsRenderer(rasterizer=canonical_rasterizer, compositor=AlphaCompositor())
        canonical_renderer.to(device)
        rendered_features = rearrange(canonical_renderer(point_cloud, eps=1e-5), "b h w c -> b c h w")
        return rendered_features

    def setup_given_cameras(self, batch):
        src_camera_K_inv = torch.linalg.inv(batch["intrinsics1"])
        dst_camera_K_inv = torch.linalg.inv(batch["intrinsics2"])
        src_camera_Rt = modules.geometry.construct_Rt_matrix(batch["rotation1"], batch["position1"])
        dst_camera_Rt = modules.geometry.construct_Rt_matrix(batch["rotation2"], batch["position2"])
        return src_camera_K_inv, dst_camera_K_inv, src_camera_Rt, dst_camera_Rt

    def warp(self, features_src, depth_src, src_camera_K_inv, dst_camera_K_inv, Rt_src_to_dst):
        b, _, h, w = features_src.shape
        image_coords = rearrange(
            modules.geometry.get_index_grid(h, w, batch=b, type_as=features_src),
            "b h w t -> b (h w) t",
        )
        src_points_in_dst_camera_coords = modules.geometry.convert_world_to_image_coordinates(
            modules.geometry.convert_image_coordinates_to_world(
                image_coords=image_coords,
                depth=rearrange(depth_src, "b h w -> b (h w)"),
                K_inv=src_camera_K_inv,
                Rt=Rt_src_to_dst,
            ),
            dst_camera_K_inv,
            torch.eye(4).unsqueeze(0).repeat(b, 1, 1).type_as(Rt_src_to_dst),
            keep_depth=True,
        )
        return self.render_features_from_points(src_points_in_dst_camera_coords, features_src)

    def render_features_from_points(self, points_in_3d, features):
        b, _, h, w = features.shape
        src_point_cloud = Pointclouds(
            points=modules.geometry.convert_to_pytorch3d_coordinate_system(points_in_3d),
            features=rearrange(features, "b c h w -> b (h w) c"),
        )
        return self.render(src_point_cloud, features.device, (h, w))

    def render_warped_images_from_ground_truth_data(self, batch):
        (
            image1_camera_K_inv,
            image2_camera_K_inv,
            image1_camera_Rt,
            image2_camera_Rt,
        ) = self.setup_given_cameras(batch)
        Rt_1_to_2 = torch.einsum("bij,bjk->bik", torch.linalg.inv(image2_camera_Rt), image1_camera_Rt)
        Rt_2_to_1 = torch.einsum("bij,bjk->bik", torch.linalg.inv(image1_camera_Rt), image2_camera_Rt)
        image1_warped_onto_image2 = self.warp(batch["image1"], batch["depth1"], image1_camera_K_inv, image2_camera_K_inv, Rt_1_to_2)
        image2_warped_onto_image1 = self.warp(batch["image2"], batch["depth2"], image2_camera_K_inv, image1_camera_K_inv, Rt_2_to_1)
        return image1_warped_onto_image2, image2_warped_onto_image1
    def warp_with_mask(self, features_src, depth_src, mask_2d_classes, src_camera_K_inv, dst_camera_K_inv, Rt_src_to_dst):
        b, c, h, w = features_src.shape
        
        # Convertir en tensor PyTorch
        if mask_2d_classes is None:
            print("mask_2d_classes est None, création d'une matrice de 1s")
            mask_2d_classes = torch.ones(b, h, w, dtype=torch.float32, device=features_src.device)
        else:
            if isinstance(mask_2d_classes, list):
                if all(m is None for m in mask_2d_classes):
                    print("Tous les masques de la liste sont None, création d'une matrice de 1s")
                    mask_2d_classes = torch.ones(b, h, w, dtype=torch.float32, device=features_src.device)
                else:
                    processed_masks = []
                    for i, m in enumerate(mask_2d_classes):
                        if m is None:
                            print(f"Masque {i} est None, création d'une matrice de 1s")
                            processed_masks.append(torch.ones(h, w, dtype=torch.float32))
                        else:
                            if isinstance(m, np.ndarray):
                                processed_masks.append(torch.from_numpy(m))
                            else:
                                processed_masks.append(torch.tensor(m))
                    mask_2d_classes = torch.stack(processed_masks)
            elif isinstance(mask_2d_classes, np.ndarray):
                mask_2d_classes = torch.from_numpy(mask_2d_classes)
            elif not torch.is_tensor(mask_2d_classes):
                mask_2d_classes = torch.tensor(mask_2d_classes)

        mask_2d_classes = mask_2d_classes.float().to(features_src.device)

        # --- Resize selon ta logique ---
        if mask_2d_classes.shape[-2:] != (h, w):
            print(f"Resize masque avec Kornia: {mask_2d_classes.shape} → ({h}, {w})")
            resize = K.augmentation.Resize((h, w), resample=2, keepdim=True)
            mask_2d_classes = resize(mask_2d_classes.unsqueeze(1)).squeeze(1)
            mask_2d_binary = (mask_2d_classes > 0).float()
        else:
            mask_2d_binary = (mask_2d_classes > 0).float()

        # --- Ajuster batch si nécessaire ---
        if mask_2d_classes.shape[0] != b:
            if mask_2d_classes.shape[0] > b:
                mask_2d_classes = mask_2d_classes[:b]
                mask_2d_binary = mask_2d_binary[:b]
            else:
                mask_2d_classes = mask_2d_classes.repeat(b // mask_2d_classes.shape[0] + 1, 1, 1)[:b]
                mask_2d_binary = mask_2d_binary.repeat(b // mask_2d_binary.shape[0] + 1, 1, 1)[:b]

        # --- Appliquer le masque binaire sur features et depth ---
        masked_depth = depth_src * mask_2d_binary
        mask_features = mask_2d_binary.unsqueeze(1)  # [b, 1, h, w]
        masked_features = features_src * mask_features
        # --- Ajouter le masque multi-classes comme canal supplémentaire ---
        mask_classes_channel = mask_2d_classes.unsqueeze(1)
        masked_depth_channel=masked_depth.unsqueeze(1)
        features_plus_mask = torch.cat([masked_features, mask_classes_channel,masked_depth_channel], dim=1)
        #features_plus_mask_plus_depth=torch.cat([features_plus_mask,masked_depth_channel],dim=1)
        return self.warp(features_plus_mask, depth_src, src_camera_K_inv, dst_camera_K_inv, Rt_src_to_dst)
            