import math
import os
import pickle
import types
from typing import Tuple

import kornia as K
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
from easydict import EasyDict
from loguru import logger as L
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from segmentation_models_pytorch.unet.model import UnetDecoder
from einops import rearrange

from modules.building_blocks import DownSamplingBlock, FeatureFusionBlock, Sequence2SpatialBlock
from modules.registeration_module import FeatureRegisterationModule

class Model(nn.Module):
    def __init__(self, args, intraclasse,load_weights_from=None):
        super().__init__()
        self.args = args
        model = build_model(args)
        self.feature_backbone = FeatureBackbone(args, model)
        self.registeration_module = FeatureRegisterationModule(args,intraclasse)
        self.bicubic_resize = K.augmentation.Resize((64, 64), resample=2, keepdim=True)
        self.unet_encoder = nn.ModuleList([DownSamplingBlock(i, j) for i, j in args.decoder.downsampling_blocks])
        self.unet_decoder = UnetDecoder(
            encoder_channels=args.decoder.encoder_channels,
            decoder_channels=args.decoder.decoder_channels,
            n_blocks=len(args.decoder.decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type="scse",
            num_coam_layers=0,
            return_features=False,
        )
        self.feature_fusion_block = FeatureFusionBlock(input_dims=64 + 768, hidden_dims=256, output_dims=64, output_resolution=[224,224])
        self.centernet_head = CenterNetHead(
            in_channels=64,
            feat_channels=64,
            num_classes=1,
            test_cfg=EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100}),
        )
        self.centernet_head.init_weights()
        if load_weights_from is not None:
            self.safely_load_state_dict(torch.load(load_weights_from))

    def safely_load_state_dict(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in checkpoint_state_dict:
            if k in model_state_dict:
                if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                    L.log(
                        "INFO",
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {checkpoint_state_dict[k].shape}",
                    )
                    checkpoint_state_dict[k] = model_state_dict[k]
            else:
                L.log("INFO", f"Dropping parameter {k}")
        self.load_state_dict(checkpoint_state_dict, strict=False)

    def forward(self, batch):
        image1_dino_features = self.feature_backbone(batch["image1"])
        image2_dino_features = self.feature_backbone(batch["image2"])
        image1_last_layer = self.bicubic_resize(image1_dino_features[-1])
        image2_last_layer = self.bicubic_resize(image2_dino_features[-1])
        image1_encoded_features = [[], image1_last_layer]
        image2_encoded_features = [[], image2_last_layer]
        for layer in self.unet_encoder:
            image1_encoded_features.append(layer(image1_encoded_features[-1]))
            image2_encoded_features.append(layer(image2_encoded_features[-1]))
        for i in range(len(self.unet_encoder)+1):
            image1_encoded_features[i + 1], image2_encoded_features[i + 1] = self.registeration_module(
                batch, image1_encoded_features[i + 1], image2_encoded_features[i + 1]
            )
        image1_decoded_features = self.unet_decoder(*image1_encoded_features)
        image2_decoded_features = self.unet_decoder(*image2_encoded_features)
        image1_decoded_features = self.feature_fusion_block(image1_dino_features[0], image1_decoded_features)
        image2_decoded_features = self.feature_fusion_block(image2_dino_features[0], image2_decoded_features)
        return (
            self.centernet_head([image1_decoded_features]),
            self.centernet_head([image2_decoded_features]),
        )

    def get_bboxes_from_logits(self, image1_outputs, image2_outputs, batch):
        """
        Version corrigée pour MMDetection 3.x
        """
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        # Préparer les data_samples
        data_samples = []
        for meta in batch["query_metadata"]:
            sample = DetDataSample()
            sample.set_metainfo(meta)
            data_samples.append(sample)

        def extract_predictions(outputs, data_samples_batch):
            """Extraire les prédictions d'un batch d'outputs"""
            results = []
            
            # Extraire les tensors des listes retournées par CenterNetHead
            try:
                if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
                    # Chaque output est une liste, prendre le premier élément (single scale)
                    center_heatmap_preds = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
                    wh_preds = outputs[1][0] if isinstance(outputs[1], list) else outputs[1]  
                    offset_preds = outputs[2][0] if isinstance(outputs[2], list) else outputs[2]
                else:
                    raise ValueError(f"Expected 3 outputs (center, wh, offset), got {len(outputs) if isinstance(outputs, (list, tuple)) else 'single tensor'}")
            except Exception as e:
                return self._create_empty_results(data_samples_batch)
            
            # Utiliser la méthode directe de décodage
            batch_size = center_heatmap_preds.shape[0]
            
            for i in range(batch_size):
                # Extraire les prédictions pour cette image
                center_pred = center_heatmap_preds[i:i+1]  # Garder la dimension batch
                wh_pred = wh_preds[i:i+1]
                offset_pred = offset_preds[i:i+1]
                
                # Obtenir les métadonnées de l'image
                img_meta = data_samples_batch[i].metainfo
                img_shape = img_meta['img_shape']
                
                # Décoder manuellement les prédictions
                decoded_bboxes, decoded_scores, decoded_labels = self._decode_single_image(
                    center_pred, wh_pred, offset_pred, img_shape
                )
                
                # Créer l'InstanceData
                pred_instances = InstanceData()
                pred_instances.bboxes = decoded_bboxes
                pred_instances.scores = decoded_scores
                pred_instances.labels = decoded_labels
                
                # Créer le résultat
                result_sample = DetDataSample()
                result_sample.set_metainfo(img_meta)
                result_sample.pred_instances = pred_instances
                results.append(result_sample)
            
            return results

        try:
            image1_predicted_bboxes = extract_predictions(image1_outputs, data_samples)
            image2_predicted_bboxes = extract_predictions(image2_outputs, data_samples)
            
        except Exception as e:
            # Fallback : retourner des résultats vides
            image1_predicted_bboxes = self._create_empty_results(data_samples)
            image2_predicted_bboxes = self._create_empty_results(data_samples)

        return image1_predicted_bboxes, image2_predicted_bboxes

    def _decode_single_image(self, center_heatmap_pred, wh_pred, offset_pred, img_shape):
        """
        Décoder les prédictions pour une seule image
        """
        try:
            # Importer les fonctions utilitaires - essayer plusieurs sources
            try:
                from mmdet.models.utils.gaussian_target import (
                    get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat
                )
            except ImportError:
                try:
                    from mmdet.models.dense_heads.utils import (
                        get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat
                    )
                except ImportError:
                    # Fallback - essayer de les importer depuis le module principal
                    from mmdet.models.utils import gaussian_radius, gen_gaussian_target
                    from mmdet.models.dense_heads.centernet_head import (
                        get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat
                    )
            
            # Configuration de test
            k = self.centernet_head.test_cfg.get('topk', 100)
            kernel = self.centernet_head.test_cfg.get('local_maximum_kernel', 3)
            
            # Dimensions
            height, width = center_heatmap_pred.shape[2:]
            inp_h, inp_w = img_shape[:2]
            
            # Appliquer sigmoid à la heatmap
            #center_heatmap_pred = center_heatmap_pred.sigmoid()
            
            # Trouver les maxima locaux
            center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)
            
            # Obtenir les top-k détections
            *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
            batch_scores, batch_index, batch_topk_labels = batch_dets
            
            # Rassembler les features WH et offset
            wh = transpose_and_gather_feat(wh_pred, batch_index)
            offset = transpose_and_gather_feat(offset_pred, batch_index)
            
            # Ajuster les positions avec offset
            topk_xs = topk_xs + offset[..., 0]
            topk_ys = topk_ys + offset[..., 1]
            
            # Calculer les bounding boxes
            tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
            tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
            br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
            br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
            
            # Stack pour former les bboxes [x1, y1, x2, y2]
            batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
            
            # Retourner les résultats pour la première image du batch
            return batch_bboxes[0], batch_scores[0], batch_topk_labels[0]
            
        except Exception as e:
            # Retourner des résultats vides en cas d'erreur
            device = center_heatmap_pred.device
            return (
                torch.empty(0, 4, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

    def _create_empty_results(self, data_samples):
        """Créer des résultats vides pour le fallback"""
        from mmengine.structures import InstanceData
        
        results = []
        for data_sample in data_samples:
            pred_instances = InstanceData()
            pred_instances.bboxes = torch.empty(0, 4)
            pred_instances.scores = torch.empty(0)
            pred_instances.labels = torch.empty(0, dtype=torch.long)
            
            result_sample = DetDataSample()
            result_sample.set_metainfo(data_sample.metainfo)
            result_sample.pred_instances = pred_instances
            results.append(result_sample)
        
        return results

    @torch.no_grad()
    def predict(self, batch):
        self.eval()
        image1_outputs, image2_outputs = self(batch)
        batch_image1_predicted_bboxes, batch_image2_predicted_bboxes = self.get_bboxes_from_logits(
            image1_outputs, image2_outputs, batch
        )
        return batch_image1_predicted_bboxes, batch_image2_predicted_bboxes

    def compute_loss(self, batch, image1_outputs, image2_outputs):
        """
        Version adaptée pour MMDetection 3.x
        """
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData
        
        # Préparer les data_samples pour l'entraînement
        data_samples_1 = []
        data_samples_2 = []
        
        for i, meta in enumerate(batch["query_metadata"]):
            # Sample pour image 1
            sample1 = DetDataSample()
            sample1.set_metainfo(meta)
            gt_instances1 = InstanceData()
            gt_instances1.bboxes = batch["target_bbox_1"][i]
            gt_instances1.labels = batch["target_bbox_labels1"][i]
            sample1.gt_instances = gt_instances1
            data_samples_1.append(sample1)
            
            # Sample pour image 2
            sample2 = DetDataSample()
            sample2.set_metainfo(meta)
            gt_instances2 = InstanceData()
            gt_instances2.bboxes = batch["target_bbox_2"][i]
            gt_instances2.labels = batch["target_bbox_labels2"][i]
            sample2.gt_instances = gt_instances2
            data_samples_2.append(sample2)

        # Calculer les losses
        try:
            # Essayer avec la nouvelle API MMDet 3.x
            image1_losses = self.centernet_head.loss(image1_outputs, data_samples_1)
            image2_losses = self.centernet_head.loss(image2_outputs, data_samples_2)
        except Exception as e:
            # Fallback vers l'ancienne API
            image1_losses = self.centernet_head.loss(
                *image1_outputs,
                batch["target_bbox_1"],
                batch["target_bbox_labels1"],
                img_metas=batch["query_metadata"],
            )
            image2_losses = self.centernet_head.loss(
                *image2_outputs,
                batch["target_bbox_2"], 
                batch["target_bbox_labels2"],
                img_metas=batch["query_metadata"],
            )

        # Sommer les losses
        overall_loss = 0
        for key in image1_losses:
            overall_loss += image1_losses[key] + image2_losses[key]
        return overall_loss

class FeatureBackbone(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.sequence_to_spatial = nn.ModuleList([Sequence2SpatialBlock(args) for _ in args.vit_feature_layers])
        self._features = []
        self.register_hooks(args.vit_feature_layers)

    def register_hooks(self, hook_layers):
        for index in hook_layers:

            def _hook(module, input, output):
                qkv = rearrange(output, "b n (t c) -> t b n c", t=3)
                self._features.append(qkv[1])

            self.model.blocks[index].attn.qkv.register_forward_hook(_hook)

    def forward(self, x):
        self.model.forward_features(x)  # desired features will get stored in self._features
        output = [self.sequence_to_spatial[i](feature) for i, feature in enumerate(self._features)]
        self._features.clear()  # clear for next forward pass
        return output

def build_model(args, frozen=True):
    model = timm.create_model("vit_base_patch8_224_dino", pretrained=True)
    model = patch_vit_resolution(model, image_hw=[224,224], stride=args.encoder.stride)
    if frozen:
        for _, value in model.named_parameters():
            value.requires_grad = False
    return model

def patch_vit_resolution(model: nn.Module, image_hw, stride: int) -> nn.Module:
    """
    change resolution of model output by changing the stride of the patch extraction.
    :param model: the model to change resolution for.
    :param stride: the new stride parameter.
    :return: the adjusted model
    """
    patch_size = model.patch_embed.patch_size
    if stride == patch_size:  # nothing to do
        return model

    stride = nn_utils._pair(stride)
    assert all([(p // s_) * s_ == p for p, s_ in zip(patch_size, stride)]), f"stride {stride} should divide patch_size {patch_size}"

    # fix the stride
    model.patch_embed.proj.stride = stride
    # fix the positional encoding code
    model._pos_embed = types.MethodType(fix_pos_enc(patch_size, image_hw, stride), model)
    return model

def fix_pos_enc(patch_size: Tuple[int, int], image_hw, stride_hw: Tuple[int, int]):
    """
    Creates a method for position encoding interpolation.
    :param patch_size: patch size of the model.
    :param stride_hw: A tuple containing the new height and width stride respectively.
    :return: the interpolation method
    """

    def interpolate_pos_encoding(self, x) -> torch.Tensor:
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        h, w = image_hw
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size[1]) // stride_hw[1]
        h0 = 1 + (h - patch_size[1]) // stride_hw[0]
        assert (
            w0 * h0 == npatch
        ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return x + torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding


# Fonctions utilitaires pour la compatibilité MMDet 3.x
def extract_bbox_results(det_data_samples):
    """
    Extraire les bboxes des DetDataSample pour compatibilité avec votre code d'inférence
    
    Usage: 
    image1_bboxes = extract_bbox_results(batch_image1_predicted_bboxes)
    
    Peut accepter:
    - Un seul DetDataSample
    - Une liste de DetDataSample
    """
    from mmdet.structures import DetDataSample
    
    # Vérifier si c'est un seul DetDataSample ou une liste
    if isinstance(det_data_samples, DetDataSample):
        # Un seul DetDataSample - le convertir en liste
        det_data_samples = [det_data_samples]
    elif not isinstance(det_data_samples, (list, tuple)):
        # Si ce n'est ni un DetDataSample ni une liste, essayer de l'itérer
        try:
            det_data_samples = list(det_data_samples)
        except TypeError:
            return [np.empty((0, 5))]
    
    results = []
    for data_sample in det_data_samples:
        if hasattr(data_sample, 'pred_instances') and hasattr(data_sample.pred_instances, 'bboxes'):
            bboxes = data_sample.pred_instances.bboxes.cpu().numpy()
            scores = data_sample.pred_instances.scores.cpu().numpy()
            
            # Vérifier que nous avons des détections et les bonnes formes
            if len(bboxes) > 0 and len(scores) > 0 and bboxes.ndim == 2 and bboxes.shape[1] == 4:
                # Combiner bboxes et scores [x1, y1, x2, y2, score]
                bbox_results = np.column_stack([bboxes, scores])
            else:
                # Pas de détections ou forme incorrecte
                bbox_results = np.empty((0, 5))
            results.append(bbox_results)
        else:
            # Résultat vide si pas de pred_instances
            results.append(np.empty((0, 5)))
    
    return results


def extract_single_bbox_result(det_data_sample):
    """
    Extraire les bboxes d'un seul DetDataSample - Version simplifiée
    
    Usage:
    bbox_array = extract_single_bbox_result(det_data_sample)
    """
    from mmdet.structures import DetDataSample
    
    if not isinstance(det_data_sample, DetDataSample):
        return np.empty((0, 5))
    
    if hasattr(det_data_sample, 'pred_instances') and hasattr(det_data_sample.pred_instances, 'bboxes'):
        bboxes = det_data_sample.pred_instances.bboxes.cpu().numpy()
        scores = det_data_sample.pred_instances.scores.cpu().numpy()
        
        if len(bboxes) > 0 and len(scores) > 0 and bboxes.ndim == 2 and bboxes.shape[1] == 4:
            # Combiner bboxes et scores [x1, y1, x2, y2, score]
            result = np.column_stack([bboxes, scores])
            return result
    
    # Retourner un array vide si pas de détections
    return np.empty((0, 5))