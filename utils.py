import kornia as K
from etils import epath
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import imageio.v2 as imageio
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import os
from modules.segmentation import segment_image
from modules.depthpromodule import depth,unidepths
from modules.geometry import get_exif_focal_length
from torchvision.utils import save_image
import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
def create_batch_from_metadata(metadata):
    list_of_items = metadata["batch"]
    batch = {}
    all_keys = [
        "image1",
        "image2",
        "depth1",
        "depth2",
        "intrinsics1",
        "intrinsics2",
        "position1",
        "position2",
        "rotation1",
        "rotation2",
        "transfm2d_1_to_2",
        "transfm2d_2_to_1",
        "registration_strategy",
        "mask1",
        "mask2",
        "focale1",
        "focale2"
    ]
    for key in all_keys:
        batch[key] = []
    for item in list_of_items:
        for key in all_keys:
            value = item.get(key, None)
            if value is None:
                batch[key].append(None)
                continue
            if "image" in key:
                value = read_image_as_pilrgb(value)
            if "depth" in key:
                value = read_depth_as_tensor(value)
            for k in ["position","rotation","intrinsics","transfm2d","mask"]:
                if k in key:
                    value = torch.tensor(np.load(value))
            batch[key].append(value)
    _sanity_test_batch(batch, list_of_items)
    return batch

def read_image_as_pilrgb(path_to_image):
    assert path_to_image is not None
    with open(path_to_image, "rb") as file:
        pil_image = Image.open(file).convert("RGB")
    return pil_image
def read_image_as_tensor(pil_image):
    image_as_tensor = pil_to_tensor(pil_image).float()
    return image_as_tensor

def read_depth_as_tensor(path_to_depth):
    assert path_to_depth is not None
    if ".tiff" in path_to_depth:
        return _read_depth_from_tiff(path_to_depth)
    return _read_depth_from_png(path_to_depth)

@torch.no_grad()
def fill_in_the_missing_information(batch,correspondence_extractor,output_dir,depthpro,unidepth,image_processor=None,model=None,depth_predictor=None,use_mask=True):
    j=0
    for i in range(len(batch["image1"])):
        j+=1
        if use_mask:
            if batch['mask1'][i]==None and batch['mask2'][i]==None :
                batch['mask1'][i]=segment_image(batch['image1'][i],output_dir,2*j)
                batch['mask2'][i]=segment_image(batch['image2'][i],output_dir,2*j+1)
        if batch['focale1'][i]==None:
            batch['focale1'][i]=get_exif_focal_length(batch['image1'][i])
        if batch['focale2'][i]==None:
            batch['focale2'][i]=get_exif_focal_length(batch['image2'][i])
        batch['image1'][i]=read_image_as_tensor(batch['image1'][i])
        batch['image2'][i]=read_image_as_tensor(batch['image2'][i])
        if batch["registration_strategy"][i] == "3d":
            assert (batch["depth1"][i] is None) == (batch["depth2"][i] is None)
            if batch["depth1"][i] is None and batch["depth2"][i] is None:
                if depthpro:
                    batch["depth1"][i],_=depth(batch["image1"][i],image_processor,model)
                    batch["depth2"][i],_=depth(batch["image2"][i],image_processor,model)
                    depth1=batch["depth1"][i]
                    depth2=batch["depth2"][i]
                    batch["depth1"][i]=torch.from_numpy(batch["depth1"][i])
                    batch["depth2"][i]=torch.from_numpy(batch["depth2"][i])
                elif unidepth:
                    batch["depth1"][i]=unidepths(batch["image1"][i],model)
                    batch["depth2"][i]=unidepths(batch["image2"][i],model)
                    depth1=batch["depth1"][i]
                    depth2=batch["depth2"][i]
                    batch["depth1"][i]=torch.from_numpy(batch["depth1"][i])
                    batch["depth2"][i]=torch.from_numpy(batch["depth2"][i])
                else:
                    batch["depth1"][i] = depth_predictor.infer(batch["image1"][i].unsqueeze(0)).squeeze()
                    batch["depth2"][i] = depth_predictor.infer(batch["image2"][i].unsqueeze(0)).squeeze()
                    depth1=batch["depth1"][i].cpu().numpy()
                    depth2=batch["depth2"][i].cpu().numpy()
                batch['image1'][i]=batch['image1'][i]/255.0
                batch['image2'][i]=batch['image2'][i]/255.0
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(depth1, cmap='magma_r')
                plt.colorbar(label='Depth')
                plt.title('Depth Map Image 1')
                plt.axis('off')
    
                plt.subplot(1, 2, 2)
                plt.imshow(depth2, cmap='magma_r')
                plt.colorbar(label='Depth')
                plt.title('Depth Map Image 2')
                plt.axis('off')
    
                plt.tight_layout()
                os.makedirs(output_dir+'/depth',exist_ok=True)
                plt.savefig(os.path.join(output_dir+'/depth', f'depth_maps_{i}.png'), bbox_inches='tight')
                plt.close()
        else:
            batch['image1'][i]=batch['image1'][i]/255.0
            batch['image2'][i]=batch['image2'][i]/255.0
    batch = correspondence_extractor(batch)
    return batch


def normalise_image(img_as_tensor):
    imagenet_normalisation = K.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = rearrange(img_as_tensor, "c h w -> 1 c h w")
    img = imagenet_normalisation(img)
    return img.squeeze()

def undo_imagenet_normalization(image_as_tensor):
    """
    Undo the imagenet normalization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_as_tensor = image_as_tensor * std + mean
    return image_as_tensor

def convert_kornia_transformation_matrix_to_normalised_coordinates(matrix, original_hw, new_hw):
    scale_up = torch.Tensor([[original_hw[1], 0, 0], [0, original_hw[0], 0], [0, 0, 1]])
    scale_down = torch.Tensor([[1 / new_hw[1], 0, 0], [0, 1 / new_hw[0], 0], [0, 0, 1]])
    return scale_down @ matrix @ scale_up

def _read_depth_from_png(path_to_depth):
    with open(path_to_depth, "rb") as file:
        pil_image = Image.open(path_to_depth)
        image_as_tensor = pil_to_tensor(pil_image).float()
    if image_as_tensor.ndim == 3:
        image_as_tensor = image_as_tensor.squeeze(0)
    return image_as_tensor

def _read_depth_from_tiff(path_to_depth):
    filename = epath.Path(path_to_depth)
    img = imageio.imread(filename.read_bytes(), format="tiff")
    if img.ndim == 2:
        img = img[:, :, None]
    return K.image_to_tensor(img).float().squeeze()

def visualise_predictions(
    left_image,
    right_image,
    left_predicted_bboxes,
    right_predicted_bboxes,
    left_score,
    right_score,
    save_path="./results.png"
):
    TARGET_COLOUR = "#1E88E5"
    PREDICTED_COLOUR = "#FFC107"
    figure, plot = plt.subplots(1, 2)
    plot[0].imshow(K.tensor_to_image(left_image))
    for i,bbox in enumerate(left_predicted_bboxes):
        bbox[[0,2]] = bbox[[0, 2]] * (left_image.shape[-1] / 224)
        bbox[[1,3]] = bbox[[1, 3]] * (left_image.shape[-2] / 224)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[0].add_patch(rect)
        if left_score is not None and i < len(left_score):
            confidence = left_score[i]
            # Positionner le texte au-dessus du bbox (légèrement au-dessus du coin supérieur gauche)
            text_x = bbox[0]
            text_y = bbox[1] - 5  # 5 pixels au-dessus
            
            # Ajouter un fond noir semi-transparent pour le texte
            plot[0].text(
                text_x, text_y, f'{confidence:.2f}',
                color=PREDICTED_COLOUR,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='bottom'
            )
    plot[0].axis("off")
    
    # Image de droite
    plot[1].imshow(K.tensor_to_image(right_image))
    plot[0].axis("off")

    plot[1].imshow(K.tensor_to_image(right_image))
    for i,bbox in enumerate(right_predicted_bboxes):
        bbox[[0,2]] = bbox[[0, 2]] * (right_image.shape[-1] / 224)
        bbox[[1,3]] = bbox[[1, 3]] * (right_image.shape[-2] / 224)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[1].add_patch(rect)
        if right_score is not None and i < len(right_score):
            confidence = right_score[i]
            # Positionner le texte au-dessus du bbox (légèrement au-dessus du coin supérieur gauche)
            text_x = bbox[0]
            text_y = bbox[1] - 5  # 5 pixels au-dessus
            
            # Ajouter un fond noir semi-transparent pour le texte
            plot[1].text(
                text_x, text_y, f'{confidence:.2f}',
                color=PREDICTED_COLOUR,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='bottom')
    plot[1].axis("off")
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
def filter_bboxes_by_mask_overlap(bboxes, mask,scores, threshold=0.3):
    filtered_bboxes = []
    filtered_scores = []
    
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = map(int, bbox[:4])
        # Vérifier les limites du masque pour éviter des index out of bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)

        cropped_mask = mask[y1:y2, x1:x2]
        if cropped_mask.size == 0:
            continue
        
        overlap_ratio = cropped_mask.sum() / cropped_mask.size
        if overlap_ratio >= threshold:
            # on ré-attache le score à la bbox
            filtered_bboxes.append(np.array([x1,y1,x2,y2,score]))
            filtered_scores.append(score)
            
    if len(filtered_bboxes) == 0:
        return np.empty((0,5)), np.empty((0,))
    
    return np.array(filtered_bboxes), np.array(filtered_scores)

def plot_correspondences(source_image, target_image, source_points, target_points, save_path="./correspondences.png"):
    """
    Helper function to plot correspondences.
    """
    fig, axarr = plt.subplots(1,2)
    if torch.is_tensor(source_image):
        source_image = K.tensor_to_image(source_image)
    if torch.is_tensor(target_image):
        target_image = K.tensor_to_image(target_image)
    axarr[0].imshow(source_image)
    plt.axis('off')
    axarr[1].imshow(target_image)
    plt.axis('off')
    source_points = source_points * torch.tensor([source_image.shape[1], source_image.shape[0]])
    target_points = target_points * torch.tensor([target_image.shape[1], target_image.shape[0]])

    for i, (pt_q, pt_t) in enumerate(zip(source_points, target_points)):
            col = (np.random.random(), np.random.random(), np.random.random())
            con = ConnectionPatch(pt_t, pt_q,
                                  coordsA='data', coordsB='data',
                                  axesA=axarr[1], axesB=axarr[0],
                                  color='g', linewidth=0.5)
            axarr[1].add_artist(con)
            axarr[0].plot(pt_q[0], pt_q[1], c=col, marker='x')
            axarr[1].plot(pt_t[0], pt_t[1], c=col, marker='x')
    axarr[0].axis("off")
    axarr[1].axis("off")
    plt.savefig(save_path, bbox_inches="tight")

def _sanity_test_batch(batch, list_of_items):
    keys_and_their_existance = [
        {
            "keys": ["image1", "image2", "registration_strategy"],
            "possible_values": [
                [True, True, True]
            ],
        },
        {
            "keys": ["depth1", "depth2", "intrinsics1", "intrinsics2", "position1", "position2", "rotation1", "rotation2"],
            "possible_values": [
                [True, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False],
                [True, True, False, False, False, False, False, False],
            ]
        },
        {
            "keys": ["transfm2d_1_to_2", "transfm2d_2_to_1"],
            "possible_values": [
                [True, True],
                [False, False],
            ]
        },
        {
            "keys": ["mask1", "mask2"],
            "possible_values": [
                [True, True],
                [False, False],
            ]
        }
    ]
    for i in range(len(list_of_items)):
        for dict in keys_and_their_existance:
            keys = dict["keys"]
            keys_exist = [batch[key][i] is not None for key in keys]
            assert keys_exist in dict["possible_values"]

def disable_batchnorm_contamination(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            print(f"BatchNorm mis en eval: {name}")
            module.eval()
    return model


def prepare_batch_for_model(batch, output_dir="batch_outputs", batch_id=None):
    """
    Modified function that saves each key element and tracks outputs
    """
    # Create output directory
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    batch_output_dir = Path(output_dir) / f"batch_{batch_id}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking DataFrame
    tracking_data = []
    
    nearest_resize = K.augmentation.Resize((224,224), resample=0, align_corners=None, keepdim=True)
    bicubic_resize = K.augmentation.Resize((224,224), resample=2, keepdim=True)
    
    # Store original data before processing
    original_data = {}
    
    for i in range(len(batch["image1"])):
        item_dir = batch_output_dir / f"paire_{i:04d}"
        item_dir.mkdir(exist_ok=True)
        
        # Track this item
        item_data = {
            "item_index": f'paire_{i}',
            "item_directory": str(item_dir)
        }
        
        # Save original dimensions
        original_hw1 = batch["image1"][i].shape[-2:]
        original_hw2 = batch["image2"][i].shape[-2:]
        
        # Save original images before processing
        
        # Process images
        batch["image1"][i] = bicubic_resize(normalise_image(batch["image1"][i]))
        batch["image2"][i] = bicubic_resize(normalise_image(batch["image2"][i]))
        
        
        # Handle depth maps
        if batch["depth1"][i] is not None:
            original_depth_hw1 = batch["depth1"][i].shape[-2:]
            batch["depth1"][i] = nearest_resize(batch["depth1"][i])
            save_tensor_data(batch["depth1"][i], item_dir / f"depth1_paire{i}.pt")
            item_data["depth1"] = str(item_dir / f"depth1_paire{i}.pt")
            
        if batch["depth2"][i] is not None:
            original_depth_hw2 = batch["depth2"][i].shape[-2:]
            batch["depth2"][i] = nearest_resize(batch["depth2"][i])
            save_tensor_data(batch["depth2"][i], item_dir / f"depth2_paire{i}.pt")
            item_data[f"depth2"] = str(item_dir / f"depth2_paire{i}.pt")
        
        # Handle intrinsics
        if batch["intrinsics1"][i] is not None:
            assert original_hw1 == original_depth_hw1
            transformation = nearest_resize.transform_matrix.squeeze()
            transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(
                transformation, original_hw1, (224, 224)
            )
            batch["intrinsics1"][i] = transformation @ batch["intrinsics1"][i]
            save_tensor_data(batch["intrinsics1"][i], item_dir / f"intrinsics1_paire{i}.pt")
            item_data["intrinsics1"] = str(item_dir / f"intrinsics1_paire{i}.pt")

        if batch["intrinsics2"][i] is not None:
            assert original_hw2 == original_depth_hw2
            transformation = nearest_resize.transform_matrix.squeeze()
            transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(
                transformation, original_hw2, (224, 224)
            )
            batch["intrinsics2"][i] = transformation @ batch["intrinsics2"][i]
            save_tensor_data(batch["intrinsics2"][i], item_dir / f"intrinsics2_paire{i}.pt")
            item_data["intrinsics2"] = str(item_dir / f"intrinsics2_paire{i}.pt")
        
        # Handle masks
        for mask_key in ["mask1", "mask2"]:
            if batch[mask_key][i] is not None:
                # Save original mask
                if isinstance(batch[mask_key][i], np.ndarray):
                    mask = torch.from_numpy(batch[mask_key][i]).float()
                else:
                    mask = batch[mask_key][i]
                # Process mask
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                mask = nearest_resize(mask).squeeze(0)
                batch[mask_key][i] = mask
                
                save_tensor_data(mask, item_dir / f"{mask_key}_paire{i}.pt")
                item_data[f'{mask_key}'] = str(item_dir / f"{mask_key}_paire{i}.pt")
            else:
                item_data[f"{mask_key}"] = None
        for keys in batch.keys():
            if (keys not in item_data):
                if batch[keys][i] is None:
                    item_data[str(keys)]=None
                elif isinstance(batch[keys][i],str) or isinstance(batch[keys][i],int) or isinstance(batch[keys][i],float):
                    item_data[str(keys)]=batch[keys][i]
                else:
                    save_tensor_data(batch[keys][i],item_dir/f'{keys}_paire{i}.pt')
                    item_data[str(keys)]=str(item_dir/f'{keys}_paire{i}.pt')

        tracking_data.append(item_data)
    
    # Stack images as in original function
    for keys in ["image1", "image2"]:
        batch[keys] = torch.stack(batch[keys])
    
    # Save stacked tensors
    save_tensor_data(batch["image1"], batch_output_dir / "stacked_image1.pt")
    save_tensor_data(batch["image2"], batch_output_dir / "stacked_image2.pt")
    
    # Generate query metadata
    batch["query_metadata"] = []
    for i in range(len(batch["image1"])):
        # Get original dimensions for this item
        original_h, original_w = original_hw1 if i == 0 else batch["image1"][i].shape[-2:]
        metadata = {
            "img_shape": (224, 224),
            "ori_shape": (original_h, original_w),
            "pad_shape": (224, 224),
            "scale_factor": (224.0/original_h, 224.0/original_w),
            "border": np.array([0, 0, 0, 0]),
            "batch_input_shape": (224, 224),
        }
        batch["query_metadata"].append(metadata)
    
    # Save metadata
    save_metadata(batch["query_metadata"], batch_output_dir / "query_metadata.json")
    
    # Create and save tracking DataFrame
    df = pd.DataFrame(tracking_data)
    
    # Add batch-level information
    for i, row in df.iterrows():
        df.at[i, "stacked_image1_path"] = str(batch_output_dir / "stacked_image1.pt")
        df.at[i, "stacked_image2_path"] = str(batch_output_dir / "stacked_image2.pt")
        df.at[i, "query_metadata_path"] = str(batch_output_dir / "query_metadata.json")
    
    # Save tracking CSV
    csv_path = batch_output_dir / f"batch_{batch_id}_tracking.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Batch {batch_id} processed and saved to {batch_output_dir}")
    print(f"Tracking CSV saved to {csv_path}")
    
    return batch


def save_tensor_data(data, filepath):
    """Save tensor data to file"""
    if isinstance(data, torch.Tensor):
        torch.save(data, filepath)
    elif isinstance(data, np.ndarray):
        np.save(filepath.with_suffix('.npy'), data)
    else:
        # For other data types, use pickle
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)


def save_metadata(metadata, filepath):
    """Save metadata as JSON"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    converted_metadata = convert_numpy(metadata)
    
    with open(filepath, 'w') as f:
        json.dump(converted_metadata, f, indent=2)
def save_occlusion_masks(occlusion_mask_1, occlusion_mask_2, occlusion_mask_1_filtered, occlusion_mask_2_filtered,
                        combined_mask_1, combined_mask_2,orig_vis1,orig_vis2,visibility1,visibility2, final_visibility1, final_visibility2,
                        d1_to2, d2_to1, depth1_orig_masked, depth2_orig_masked,
                        save_dir='demo_data/occlusion_masks', batch_idx=0, pair_idx=0):
    """
    Save and visualize all occlusion-related masks for debugging
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    def save_mask_image(mask_tensor, filename, title=""):
        """Helper function to save a mask tensor as image"""
        if mask_tensor.dim() == 4:  # [b, 1, h, w]
            mask_np = mask_tensor[batch_idx, 0].cpu().numpy()
        elif mask_tensor.dim() == 3:  # [b, h, w]
            mask_np = mask_tensor[batch_idx].cpu().numpy()
        else:
            mask_np = mask_tensor.cpu().numpy()
            
        # Normalize to 0-1 range for visualization
        if mask_np.dtype == bool:
            mask_np = mask_np.astype(float)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(mask_np, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f"{title} - Batch {batch_idx}, Pair {pair_idx}")
        plt.savefig(os.path.join(save_dir, f"{filename}_b{batch_idx}_p{pair_idx}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save as tensor image (0-1 normalized)
        if isinstance(mask_tensor, torch.Tensor):
            mask_normalized = (mask_tensor[batch_idx:batch_idx+1].float() - mask_tensor[batch_idx:batch_idx+1].float().min()) / \
                            (mask_tensor[batch_idx:batch_idx+1].float().max() - mask_tensor[batch_idx:batch_idx+1].float().min() + 1e-8)
            save_image(mask_normalized, os.path.join(save_dir, f"{filename}_tensor_b{batch_idx}_p{pair_idx}.png"))
    
    def save_depth_comparison(depth_orig1,depth_warped, depth_orig2, filename_prefix, title_prefix):
        """Save depth comparison and difference"""
        depth_o1 = depth_orig1[batch_idx, 0].cpu().numpy() if depth_orig1.dim() == 4 else depth_orig1[batch_idx].cpu().numpy()
        depth_w = depth_warped[batch_idx, 0].cpu().numpy() if depth_warped.dim() == 4 else depth_warped[batch_idx].cpu().numpy()
        depth_o = depth_orig2[batch_idx, 0].cpu().numpy() if depth_orig2.dim() == 4 else depth_orig2[batch_idx].cpu().numpy()
        depth_diff = depth_w - depth_o
        
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        im1 = axes[0].imshow(depth_o1, cmap='plasma')
        axes[0].set_title(f"{title_prefix} Depth 1")
        plt.colorbar(im1, ax=axes[0])
        # Warped depth
        im1 = axes[1].imshow(depth_w, cmap='plasma')
        axes[0].set_title(f"{title_prefix} Warped Depth")
        plt.colorbar(im1, ax=axes[0])
        
        # Original depth
        im2 = axes[2].imshow(depth_o, cmap='plasma')
        axes[1].set_title(f"{title_prefix} Original Depth2")
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        im3 = axes[3].imshow(depth_diff, cmap='RdBu_r', vmin=-2, vmax=2)
        axes[2].set_title(f"{title_prefix} Depth Difference")
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(f"Depth Analysis - Batch {batch_idx}, Pair {pair_idx}")
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_depth_analysis_b{batch_idx}_p{pair_idx}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saving occlusion masks for batch {batch_idx}, pair {pair_idx}...")
    
    # Save basic occlusion masks
    def save_combined_masks_comparison():
        """Save all masks in a single comparison figure"""
        fig, axes = plt.subplots(2, 6, figsize=(20, 10))
        
        # Helper function to display mask on axis
        def show_mask_on_axis(ax, mask_tensor, title):
            if mask_tensor.dim() == 4:  # [b, 1, h, w]
                mask_np = mask_tensor[batch_idx, 0].cpu().numpy()
            elif mask_tensor.dim() == 3:  # [b, h, w]
                mask_np = mask_tensor[batch_idx].cpu().numpy()
            else:
                mask_np = mask_tensor.cpu().numpy()
                
            if mask_np.dtype == bool:
                mask_np = mask_np.astype(float)
            
            im = ax.imshow(mask_np, cmap='viridis', interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            return im
        
        # Row 1: Direction 1 masks (image1 -> image2)
        im01=show_mask_on_axis(axes[0,0], orig_vis1, "visibilité géometrique intiale")
        im1 = show_mask_on_axis(axes[0,1], occlusion_mask_1, "Raw Occlusion 1\n(d1->d2)")
        im2 = show_mask_on_axis(axes[0,2], occlusion_mask_1_filtered, "Filtered Occlusion 1")
        im3 = show_mask_on_axis(axes[0,3], combined_mask_1, "Combined Semantic 1")
        im9=show_mask_on_axis(axes[0,4], visibility1, " Visibility 1 geometrique+combined mask")
        im4 = show_mask_on_axis(axes[0,5], final_visibility1, "Final Visibility 1")
        
        # Row 2: Direction 2 masks (image2 -> image1)
        im02=show_mask_on_axis(axes[1,0], orig_vis2, "visibilité géometrique intiale")
        im5 = show_mask_on_axis(axes[1,1], occlusion_mask_2, "Raw Occlusion 2\n(d2->d1)")
        im6 = show_mask_on_axis(axes[1,2], occlusion_mask_2_filtered, "Filtered Occlusion 2")
        im7 = show_mask_on_axis(axes[1,3], combined_mask_2, "Combined Semantic 2")
        im10=show_mask_on_axis(axes[1,4], visibility2, " Visibility 2 geometrique+combined mask")
        im8 = show_mask_on_axis(axes[1,5], final_visibility2, "Final Visibility 2")
        
        # Add colorbars
        for i, im in enumerate([im01,im1, im2, im3, im9,im4, im02,im5, im6, im7, im10,im8]):
            plt.colorbar(im, ax=axes.flat[i], shrink=0.6)
        
        plt.suptitle(f"Occlusion Masks Comparison - Batch {batch_idx}, Pair {pair_idx}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"combined_masks_comparison_b{batch_idx}_p{pair_idx}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    save_combined_masks_comparison()
    
    # Save depth comparisons
    save_depth_comparison(depth1_orig_masked,d1_to2, depth2_orig_masked, "d1_to_d2", "Image1->Image2")
    save_depth_comparison(depth2_orig_masked,d2_to1, depth1_orig_masked, "d2_to_d1", "Image2->Image1")
    
    # Save occlusion statistics
    
    print(f"Saved all occlusion visualizations to {save_dir}")