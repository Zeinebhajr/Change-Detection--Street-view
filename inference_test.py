import yaml
from easydict import EasyDict
import fix_uint16
from modules.model import Model
from utils import create_batch_from_metadata, fill_in_the_missing_information, prepare_batch_for_model, visualise_predictions, plot_correspondences, undo_imagenet_normalization
from modules.correspondence_extractor import CorrespondenceExtractor
import torch
from modules.geometry import remove_bboxes_with_area_less_than, suppress_overlapping_bboxes, get_height_width,load_annotations_for_image
from modules.model import extract_single_bbox_result
from modules.eval2 import evaluate_model_predictions,visualise_predictions_with_ground_truth,visualise_predictions
import numpy as np
import random
import os
from torch.hub import load_state_dict_from_url
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from unidepth.models import UniDepthV2
print("finished importing")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(
    config_file: str = "config.yml",
    input_metadata: str = "input_metadata/inputdchan_ToutesPaires.yml",
    output_dir='Resultats',
    load_weights_from: str = None,
    filter_predictions_with_area_under: int = 400,
    max_predictions_to_display: int = 5,
    seed: int = 42,
    depthpro: bool = False,
    annotations_dir: str = "annotations",  
    evaluate_results: bool = True,  
    iou_thresholds: list = [0.1, 0.3, 0.5],
    roma: bool = True,
    visualize_ground_truth: bool = True,
    intraclasse: bool = False,
    unidepth=True,
    use_mask: bool = True
):
    
    configs = get_easy_dict_from_yaml_file(config_file)
    set_seed(seed)
    model = Model(configs,intraclasse, load_weights_from=load_weights_from)
    correspondence_extractor = CorrespondenceExtractor(roma)

    depth_predictor = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if depthpro:
        image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        model_dpt = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    elif unidepth:
        type_ = "l"  # available types: s, b, l
        name = f"unidepth-v2-vit{type_}14"
        model_dpt = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
        image_processor=None
    else: #zoedepth
        image_processor, model_dpt = None, None
    
    os.makedirs(output_dir, exist_ok=True)
    batch_metadata = get_easy_dict_from_yaml_file(input_metadata)
    
    batch = create_batch_from_metadata(batch_metadata)
    batch = fill_in_the_missing_information(batch,correspondence_extractor, output_dir, depthpro,unidepth ,image_processor, model_dpt, depth_predictor,use_mask)
    batch = prepare_batch_for_model(batch,output_dir=f'{output_dir}/batch_outputs')

    batch_image1_predicted_bboxes, batch_image2_predicted_bboxes = model.predict(batch)
    
    # Initialize prediction storage for evaluation
    all_predictions_img1 = []
    all_predictions_img2 = []
    
    # Process predictions for each image pair
    for i, (image1_det_sample, image2_det_sample) in enumerate(zip(batch_image1_predicted_bboxes, batch_image2_predicted_bboxes)):
        # Plot correspondences
        plot_correspondences(
            undo_imagenet_normalization(batch["image1"][i]), 
            undo_imagenet_normalization(batch["image2"][i]), 
            batch["points1"][i], 
            batch["points2"][i], 
            save_path=os.path.join(output_dir, f"correspondences_{i}.png")
        )     

        # Initialize empty bounding boxes arrays
        image1_bboxes = np.empty((0, 4))
        image2_bboxes = np.empty((0, 4))
        scores1 = np.empty((0,))
        scores2 = np.empty((0,))
        
        # Extract bounding boxes if they exist
        extracted_image1_bboxes = extract_single_bbox_result(image1_det_sample)
        extracted_image2_bboxes = extract_single_bbox_result(image2_det_sample)

        # Process image1 bboxes if valid
        if len(extracted_image1_bboxes) > 0 and extracted_image1_bboxes.ndim == 2:
            # Filter predictions by area
            filtered_image1_bboxes = remove_bboxes_with_area_less_than(extracted_image1_bboxes, filter_predictions_with_area_under)
            
            if len(filtered_image1_bboxes) > 0:
                # Apply non-maxima suppression
                image1_bboxes, scores1 = suppress_overlapping_bboxes(filtered_image1_bboxes[:, :4], filtered_image1_bboxes[:, 4])
        
        if len(extracted_image2_bboxes) > 0 and extracted_image2_bboxes.ndim == 2:
            # Filter predictions by area
            filtered_image2_bboxes = remove_bboxes_with_area_less_than(extracted_image2_bboxes, filter_predictions_with_area_under)
            
            if len(filtered_image2_bboxes) > 0:
                image2_bboxes, scores2 = suppress_overlapping_bboxes(filtered_image2_bboxes[:, :4], filtered_image2_bboxes[:, 4])
        
        if len(image1_bboxes) > 0:
            image1_final = np.column_stack([image1_bboxes[:max_predictions_to_display], scores1[:max_predictions_to_display]])
        else:
            image1_final = np.empty((0, 5))
            
        if len(image2_bboxes) > 0:
            image2_final = np.column_stack([image2_bboxes[:max_predictions_to_display], scores2[:max_predictions_to_display]])
        else:
            image2_final = np.empty((0, 5))
        
        # Store predictions for evaluation (même si vides)
        all_predictions_img1.append(image1_final)
        all_predictions_img2.append(image2_final)
        
        # VISUALISATION - Toujours exécutée, même avec des bounding boxes vides
        if visualize_ground_truth:
            # Main visualization with ground truth overlay
            visualise_predictions_with_ground_truth(
                undo_imagenet_normalization(batch["image1"][i]), 
                undo_imagenet_normalization(batch["image2"][i]), 
                image1_bboxes[:max_predictions_to_display] if len(image1_bboxes) > 0 else np.empty((0, 4)), 
                image2_bboxes[:max_predictions_to_display] if len(image2_bboxes) > 0 else np.empty((0, 4)), 
                scores1[:max_predictions_to_display] if len(scores1) > 0 else np.empty((0,)), 
                scores2[:max_predictions_to_display] if len(scores2) > 0 else np.empty((0,)),
                batch_metadata=batch_metadata,
                image_pair_index=i,
                annotations_dir=annotations_dir,
                save_path=f"{output_dir}/prediction_vs_gt_{i}.png"
            )
            
        # Original visualization (predictions only) - Toujours exécutée
        visualise_predictions(
            undo_imagenet_normalization(batch["image1"][i]), 
            undo_imagenet_normalization(batch["image2"][i]), 
            image1_bboxes[:max_predictions_to_display] if len(image1_bboxes) > 0 else np.empty((0, 4)), 
            image2_bboxes[:max_predictions_to_display] if len(image2_bboxes) > 0 else np.empty((0, 4)), 
            scores1[:max_predictions_to_display] if len(scores1) > 0 else np.empty((0,)), 
            scores2[:max_predictions_to_display] if len(scores2) > 0 else np.empty((0,)), 
            save_path=f"{output_dir}/prediction_depthpro{i}.png"
        )
    
    # Run evaluation using the extracted module
    if evaluate_results:
        evaluation_results = evaluate_model_predictions(
            batch_metadata=batch_metadata,
            all_predictions_img1=all_predictions_img1,
            all_predictions_img2=all_predictions_img2,
            output_dir=output_dir,
            annotations_dir=annotations_dir,
            iou_thresholds=iou_thresholds,
            model_size=224
        )
        
        return evaluation_results
    
    return None

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """Reads a yaml and returns it as an easy dict."""
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)