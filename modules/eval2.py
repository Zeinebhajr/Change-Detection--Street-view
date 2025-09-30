import os
import numpy as np
import pandas as pd
from modules.geometry import load_annotations_for_image, evaluate_predictions, calculate_per_image_metrics, get_height_width
import kornia as K
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import stats


def evaluate_model_predictions(
    batch_metadata,
    all_predictions_img1,
    all_predictions_img2,
    output_dir,
    annotations_dir="annotations",
    iou_thresholds=[0.1, 0.3, 0.5],
    model_size=224
):
    """
    Comprehensive evaluation function for model predictions with combined results and statistics.
    
    Args:
        batch_metadata: Metadata containing image paths and information
        all_predictions_img1: List of prediction arrays for image1 in each pair
        all_predictions_img2: List of prediction arrays for image2 in each pair
        output_dir: Directory to save evaluation results
        annotations_dir: Directory containing ground truth annotations
        iou_thresholds: List of IoU thresholds for evaluation
        model_size: Model input size for scale factor calculation
        
    Returns:
        dict: Comprehensive evaluation results including combined metrics
    """
    
    # Prepare ground truth data and image IDs
    all_ground_truths_img1 = []
    all_ground_truths_img2 = []
    all_combined_predictions = []
    all_combined_ground_truths = []
    image_ids_img1 = []
    image_ids_img2 = []
    per_image_results = []
    
    for i, batch_item in enumerate(batch_metadata.batch):
        # Calculate scale factors
        height1, width1 = get_height_width(batch_item.image1)
        height2, width2 = get_height_width(batch_item.image2)
        gt_scale_factor1 = model_size / max(height1, width1)
        gt_scale_factor2 = model_size / max(height2, width2)
        
        # Generate image IDs
        #image1_id = os.path.splitext(os.path.basename(batch_item.image1))[0] + "_combined_mask.png"
        #image2_id = os.path.splitext(os.path.basename(batch_item.image2))[0] + "_combined_mask.png"
        image1_base = os.path.splitext(os.path.basename(batch_item.image1))[0]
        print(image1_base)
        image1_id = image1_base 
        print(image1_id,image1_id+'.txt')
        if not os.path.exists(os.path.join(annotations_dir, image1_id+ '.txt')):
            image1_id = image1_base + "_combined_mask.png"

        # Image 2
        image2_base = os.path.splitext(os.path.basename(batch_item.image2))[0]
        image2_id = image2_base 
        if not os.path.exists(os.path.join(annotations_dir, image2_id+ '.txt')):
            image2_id = image2_base + "_combined_mask.png"

        # Load ground truth annotations
        gt1 = load_annotations_for_image(image1_id, annotations_dir, gt_scale_factor1)
        gt2 = load_annotations_for_image(image2_id, annotations_dir, gt_scale_factor2)
        
        all_ground_truths_img1.append(gt1)
        all_ground_truths_img2.append(gt2)
        image_ids_img1.append(image1_id)
        image_ids_img2.append(image2_id)
        
        # Get current predictions (handle empty cases)
        img1_preds = all_predictions_img1[i] if i < len(all_predictions_img1) else np.empty((0, 5))
        img2_preds = all_predictions_img2[i] if i < len(all_predictions_img2) else np.empty((0, 5))
        
        # Combine predictions and ground truths for global evaluation
        combined_preds = np.vstack([img1_preds, img2_preds]) if len(img1_preds) > 0 or len(img2_preds) > 0 else np.empty((0, 5))
        combined_gts = np.vstack([gt1, gt2]) if len(gt1) > 0 or len(gt2) > 0 else np.empty((0, 4))
        
        all_combined_predictions.append(combined_preds)
        all_combined_ground_truths.append(combined_gts)
        
        # Store basic info for per-image results
        per_image_results.append({
            'image_pair_id': i,
            'image1_path': batch_item.image1,
            'image2_path': batch_item.image2,
            'image1_id': image1_id,
            'image2_id': image2_id,
            'num_predictions_img1': len(img1_preds),
            'num_predictions_img2': len(img2_preds),
            'num_predictions_combined': len(combined_preds),
            'num_gt_img1': len(gt1),
            'num_gt_img2': len(gt2),
            'num_gt_combined': len(combined_gts),
            'scale_factor_img1': gt_scale_factor1,
            'scale_factor_img2': gt_scale_factor2
        })
    
    # Evaluate predictions for individual images
    results1, evaluator1 = evaluate_predictions(
        all_predictions_img1, all_ground_truths_img1, image_ids_img1,
        "image1", output_dir, iou_thresholds
    )
    
    results2, evaluator2 = evaluate_predictions(
        all_predictions_img2, all_ground_truths_img2, image_ids_img2,
        "image2", output_dir, iou_thresholds
    )
    
    # Evaluate combined predictions
    combined_image_ids = [f"pair_{i}_combined" for i in range(len(all_combined_predictions))]
    results_combined, evaluator_combined = evaluate_predictions(
        all_combined_predictions, all_combined_ground_truths, combined_image_ids,
        "combined", output_dir, iou_thresholds
    )
    
    # Calculate per-image metrics for each IoU threshold (including combined)
    _calculate_per_image_metrics_enhanced(
        per_image_results, 
        all_predictions_img1, 
        all_predictions_img2,
        all_combined_predictions,
        all_ground_truths_img1, 
        all_ground_truths_img2,
        all_combined_ground_truths,
        iou_thresholds
    )
    
    # Calculate comprehensive statistics
    statistics_summary = _calculate_comprehensive_statistics(
        per_image_results, results1, results2, results_combined, iou_thresholds
    )
    
    # Generate performance curves
    _generate_performance_curves(
        per_image_results, statistics_summary, output_dir, iou_thresholds
    )
    
    # Save results to CSV files
    _save_evaluation_results_enhanced(per_image_results, output_dir, iou_thresholds)
    
    # Print and save comprehensive summary
    _print_and_save_comprehensive_summary(
        results1, results2, results_combined, statistics_summary, output_dir
    )
    
    return {
        'image1_results': results1,
        'image2_results': results2,
        'combined_results': results_combined,
        'per_image_results': per_image_results,
        'statistics_summary': statistics_summary,
        'evaluator1': evaluator1,
        'evaluator2': evaluator2,
        'evaluator_combined': evaluator_combined
    }


def _calculate_per_image_metrics_enhanced(per_image_results, all_predictions_img1, all_predictions_img2,
                                        all_combined_predictions, all_ground_truths_img1, all_ground_truths_img2,
                                        all_combined_ground_truths, iou_thresholds):
    """Calculate enhanced metrics for each image individually including combined results and F1."""
    for i, img_result in enumerate(per_image_results):
        img1_preds = all_predictions_img1[i] if i < len(all_predictions_img1) else np.empty((0, 5))
        img1_gts = all_ground_truths_img1[i] if i < len(all_ground_truths_img1) else np.empty((0, 4))
        img2_preds = all_predictions_img2[i] if i < len(all_predictions_img2) else np.empty((0, 5))
        img2_gts = all_ground_truths_img2[i] if i < len(all_ground_truths_img2) else np.empty((0, 4))
        combined_preds = all_combined_predictions[i] if i < len(all_combined_predictions) else np.empty((0, 5))
        combined_gts = all_combined_ground_truths[i] if i < len(all_combined_ground_truths) else np.empty((0, 4))
        
        for iou_threshold in iou_thresholds:
            # Image1 metrics
            try:
                img1_metrics = calculate_per_image_metrics(img1_preds, img1_gts, iou_threshold)
                img_result[f'img1_mAP_IoU_{iou_threshold}'] = img1_metrics.get('mAP', 0.0)
                img_result[f'img1_recall_IoU_{iou_threshold}'] = img1_metrics.get('recall', 0.0)
                img_result[f'img1_precision_IoU_{iou_threshold}'] = img1_metrics.get('precision', 0.0)
                img_result[f'img1_f1_IoU_{iou_threshold}'] = img1_metrics.get('f1', 0.0)
            except Exception as e:
                print(f"Error calculating img1 metrics for image {i}: {e}")
                img_result[f'img1_mAP_IoU_{iou_threshold}'] = 0.0
                img_result[f'img1_recall_IoU_{iou_threshold}'] = 0.0
                img_result[f'img1_precision_IoU_{iou_threshold}'] = 0.0
                img_result[f'img1_f1_IoU_{iou_threshold}'] = 0.0
            
            # Image2 metrics
            try:
                img2_metrics = calculate_per_image_metrics(img2_preds, img2_gts, iou_threshold)
                img_result[f'img2_mAP_IoU_{iou_threshold}'] = img2_metrics.get('mAP', 0.0)
                img_result[f'img2_recall_IoU_{iou_threshold}'] = img2_metrics.get('recall', 0.0)
                img_result[f'img2_precision_IoU_{iou_threshold}'] = img2_metrics.get('precision', 0.0)
                img_result[f'img2_f1_IoU_{iou_threshold}'] = img2_metrics.get('f1', 0.0)
            except Exception as e:
                print(f"Error calculating img2 metrics for image {i}: {e}")
                img_result[f'img2_mAP_IoU_{iou_threshold}'] = 0.0
                img_result[f'img2_recall_IoU_{iou_threshold}'] = 0.0
                img_result[f'img2_precision_IoU_{iou_threshold}'] = 0.0
                img_result[f'img2_f1_IoU_{iou_threshold}'] = 0.0
            
            # Combined metrics
            try:
                combined_metrics = calculate_per_image_metrics(combined_preds, combined_gts, iou_threshold)
                img_result[f'combined_mAP_IoU_{iou_threshold}'] = combined_metrics.get('mAP', 0.0)
                img_result[f'combined_recall_IoU_{iou_threshold}'] = combined_metrics.get('recall', 0.0)
                img_result[f'combined_precision_IoU_{iou_threshold}'] = combined_metrics.get('precision', 0.0)
                img_result[f'combined_f1_IoU_{iou_threshold}'] = combined_metrics.get('f1', 0.0)
            except Exception as e:
                print(f"Error calculating combined metrics for image {i}: {e}")
                img_result[f'combined_mAP_IoU_{iou_threshold}'] = 0.0
                img_result[f'combined_recall_IoU_{iou_threshold}'] = 0.0
                img_result[f'combined_precision_IoU_{iou_threshold}'] = 0.0
                img_result[f'combined_f1_IoU_{iou_threshold}'] = 0.0
            
            # Calculate average metrics (img1 + img2) / 2
            img1_map = img_result[f'img1_mAP_IoU_{iou_threshold}']
            img2_map = img_result[f'img2_mAP_IoU_{iou_threshold}']
            img_result[f'avg_mAP_IoU_{iou_threshold}'] = (img1_map + img2_map) / 2
            
            img1_recall = img_result[f'img1_recall_IoU_{iou_threshold}']
            img2_recall = img_result[f'img2_recall_IoU_{iou_threshold}']
            img_result[f'avg_recall_IoU_{iou_threshold}'] = (img1_recall + img2_recall) / 2
            
            img1_precision = img_result[f'img1_precision_IoU_{iou_threshold}']
            img2_precision = img_result[f'img2_precision_IoU_{iou_threshold}']
            img_result[f'avg_precision_IoU_{iou_threshold}'] = (img1_precision + img2_precision) / 2
            
            img1_f1 = img_result[f'img1_f1_IoU_{iou_threshold}']
            img2_f1 = img_result[f'img2_f1_IoU_{iou_threshold}']
            img_result[f'avg_f1_IoU_{iou_threshold}'] = (img1_f1 + img2_f1) / 2


def _calculate_comprehensive_statistics(per_image_results, results1, results2, results_combined, iou_thresholds):
    """Calculate comprehensive statistics across all metrics and IoU thresholds."""
    statistics = {}
    
    for iou_threshold in iou_thresholds:
        stats_key = f'IoU_{iou_threshold}'
        statistics[stats_key] = {}
        
        # Extract metrics for this IoU threshold
        metrics = ['mAP', 'recall', 'precision', 'f1']
        image_types = ['img1', 'img2', 'combined', 'avg']
        
        for image_type in image_types:
            for metric in metrics:
                metric_key = f'{image_type}_{metric}_IoU_{iou_threshold}'
                
                # Only include images that have ground truth
                values = []
                for result in per_image_results:
                    # Check if this image type has ground truth
                    if image_type == 'img1' and result.get('num_gt_img1', 0) > 0:
                        metric_value = result.get(metric_key, 0.0)
                        if metric_value is not None and not np.isnan(metric_value):
                            values.append(metric_value)
                    elif image_type == 'img2' and result.get('num_gt_img2', 0) > 0:
                        metric_value = result.get(metric_key, 0.0)
                        if metric_value is not None and not np.isnan(metric_value):
                            values.append(metric_value)
                    elif image_type == 'combined' and result.get('num_gt_combined', 0) > 0:
                        metric_value = result.get(metric_key, 0.0)
                        if metric_value is not None and not np.isnan(metric_value):
                            values.append(metric_value)
                    elif image_type == 'avg':
                        # For average, include if either img1 or img2 has GT
                        if (result.get('num_gt_img1', 0) > 0 or result.get('num_gt_img2', 0) > 0):
                            metric_value = result.get(metric_key, 0.0)
                            if metric_value is not None and not np.isnan(metric_value):
                                values.append(metric_value)
                
                if values:
                    statistics[stats_key][f'{image_type}_{metric}'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'count': len(values)
                    }
                else:
                    statistics[stats_key][f'{image_type}_{metric}'] = {
                        'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                        'q25': 0.0, 'q75': 0.0, 'count': 0
                    }
    
    # Overall dataset statistics
    images_with_gt_img1 = sum(1 for result in per_image_results if result.get('num_gt_img1', 0) > 0)
    images_with_gt_img2 = sum(1 for result in per_image_results if result.get('num_gt_img2', 0) > 0)
    images_with_gt_combined = sum(1 for result in per_image_results if result.get('num_gt_combined', 0) > 0)
    
    statistics['dataset_stats'] = {
        'total_image_pairs': len(per_image_results),
        'images_with_gt_img1': images_with_gt_img1,
        'images_with_gt_img2': images_with_gt_img2,
        'images_with_gt_combined': images_with_gt_combined,
        'total_predictions_img1': sum(result.get('num_predictions_img1', 0) for result in per_image_results),
        'total_predictions_img2': sum(result.get('num_predictions_img2', 0) for result in per_image_results),
        'total_predictions_combined': sum(result.get('num_predictions_combined', 0) for result in per_image_results),
        'total_gt_img1': sum(result.get('num_gt_img1', 0) for result in per_image_results),
        'total_gt_img2': sum(result.get('num_gt_img2', 0) for result in per_image_results),
        'total_gt_combined': sum(result.get('num_gt_combined', 0) for result in per_image_results),
        'avg_predictions_per_image': np.mean([result.get('num_predictions_combined', 0) for result in per_image_results]),
        'avg_gt_per_image': np.mean([result.get('num_gt_combined', 0) for result in per_image_results if result.get('num_gt_combined', 0) > 0]) if images_with_gt_combined > 0 else 0.0
    }
    
    return statistics


def _generate_performance_curves(per_image_results, statistics_summary, output_dir, iou_thresholds):
    """Generate comprehensive performance curves and visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. mAP vs IoU Threshold Curve
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # mAP curves
    image_types = ['img1', 'img2', 'combined', 'avg']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (img_type, color) in enumerate(zip(image_types, colors)):
        map_values = []
        for iou_threshold in iou_thresholds:
            stats_key = f'IoU_{iou_threshold}'
            metric_key = f'{img_type}_mAP'
            map_value = statistics_summary[stats_key][metric_key]['mean']
            map_values.append(map_value)
        
        axes[0, 0].plot(iou_thresholds, map_values, marker='o', linewidth=2, 
                       label=f'{img_type.replace("_", " ").title()}', color=color)
    
    axes[0, 0].set_xlabel('IoU Threshold')
    axes[0, 0].set_ylabel('mAP')
    axes[0, 0].set_title('mAP vs IoU Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score curves
    for i, (img_type, color) in enumerate(zip(image_types, colors)):
        f1_values = []
        for iou_threshold in iou_thresholds:
            stats_key = f'IoU_{iou_threshold}'
            metric_key = f'{img_type}_f1'
            f1_value = statistics_summary[stats_key][metric_key]['mean']
            f1_values.append(f1_value)
        
        axes[0, 1].plot(iou_thresholds, f1_values, marker='s', linewidth=2,
                       label=f'{img_type.replace("_", " ").title()}', color=color)
    
    axes[0, 1].set_xlabel('IoU Threshold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score vs IoU Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall curves at IoU=0.5
    iou_05_key = 'IoU_0.5'
    if iou_05_key in statistics_summary:
        precision_values = []
        recall_values = []
        for img_type in image_types:
            precision = statistics_summary[iou_05_key][f'{img_type}_precision']['mean']
            recall = statistics_summary[iou_05_key][f'{img_type}_recall']['mean']
            precision_values.append(precision)
            recall_values.append(recall)
        
        for i, (img_type, color) in enumerate(zip(image_types, colors)):
            axes[1, 0].scatter(recall_values[i], precision_values[i], s=100, 
                             label=f'{img_type.replace("_", " ").title()}', color=color)
    
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Recall (IoU=0.5)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # Performance distribution boxplot (mAP at IoU=0.5)
    if iou_05_key in statistics_summary:
        box_data = []
        box_labels = []
        for img_type in image_types:
            metric_key = f'{img_type}_mAP_IoU_0.5'
            values = [result.get(metric_key, 0.0) for result in per_image_results]
            values = [v for v in values if v is not None and not np.isnan(v)]
            if values:
                box_data.append(values)
                box_labels.append(img_type.replace('_', ' ').title())
        
        if box_data:
            bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    
    axes[1, 1].set_ylabel('mAP')
    axes[1, 1].set_title('mAP Distribution (IoU=0.5)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed metrics heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data for heatmap
    heatmap_data = []
    row_labels = []
    col_labels = ['mAP', 'Recall', 'Precision', 'F1']
    
    for iou_threshold in iou_thresholds:
        stats_key = f'IoU_{iou_threshold}'
        for img_type in image_types:
            row_data = []
            for metric in ['mAP', 'recall', 'precision', 'f1']:
                metric_key = f'{img_type}_{metric}'
                value = statistics_summary[stats_key][metric_key]['mean']
                row_data.append(value)
            heatmap_data.append(row_data)
            row_labels.append(f'{img_type} (IoU={iou_threshold})')
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black" if heatmap_data[i, j] < 0.5 else "white")
    
    ax.set_title('Performance Metrics Heatmap')
    plt.colorbar(im, ax=ax, label='Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def _save_evaluation_results_enhanced(per_image_results, output_dir, iou_thresholds):
    """Save enhanced evaluation results including combined metrics."""
    # Save per-image results to separate CSV files for each IoU threshold
    for iou_threshold in iou_thresholds:
        iou_results = []
        
        for img_result in per_image_results:
            iou_row = {
                'image_pair_id': img_result['image_pair_id'],
                'image1_path': img_result['image1_path'],
                'image2_path': img_result['image2_path'],
                'image1_id': img_result['image1_id'],
                'image2_id': img_result['image2_id'],
                'num_predictions_img1': img_result['num_predictions_img1'],
                'num_predictions_img2': img_result['num_predictions_img2'],
                'num_predictions_combined': img_result['num_predictions_combined'],
                'num_gt_img1': img_result['num_gt_img1'],
                'num_gt_img2': img_result['num_gt_img2'],
                'num_gt_combined': img_result['num_gt_combined'],
                'scale_factor_img1': img_result['scale_factor_img1'],
                'scale_factor_img2': img_result['scale_factor_img2'],
                
                # Individual image metrics
                'img1_mAP': img_result[f'img1_mAP_IoU_{iou_threshold}'],
                'img1_recall': img_result[f'img1_recall_IoU_{iou_threshold}'],
                'img1_precision': img_result[f'img1_precision_IoU_{iou_threshold}'],
                'img1_f1': img_result[f'img1_f1_IoU_{iou_threshold}'],
                'img2_mAP': img_result[f'img2_mAP_IoU_{iou_threshold}'],
                'img2_recall': img_result[f'img2_recall_IoU_{iou_threshold}'],
                'img2_precision': img_result[f'img2_precision_IoU_{iou_threshold}'],
                'img2_f1': img_result[f'img2_f1_IoU_{iou_threshold}'],
                
                # Combined metrics
                'combined_mAP': img_result[f'combined_mAP_IoU_{iou_threshold}'],
                'combined_recall': img_result[f'combined_recall_IoU_{iou_threshold}'],
                'combined_precision': img_result[f'combined_precision_IoU_{iou_threshold}'],
                'combined_f1': img_result[f'combined_f1_IoU_{iou_threshold}'],
                
                # Average metrics
                'avg_mAP': img_result[f'avg_mAP_IoU_{iou_threshold}'],
                'avg_recall': img_result[f'avg_recall_IoU_{iou_threshold}'],
                'avg_precision': img_result[f'avg_precision_IoU_{iou_threshold}'],
                'avg_f1': img_result[f'avg_f1_IoU_{iou_threshold}']
            }
            iou_results.append(iou_row)
        
        # Save to CSV
        df_iou = pd.DataFrame(iou_results)
        csv_path = os.path.join(output_dir, f"per_image_results_IoU_{iou_threshold}.csv")
        df_iou.to_csv(csv_path, index=False)

    
    # Save complete results with all IoU thresholds
    df_complete = pd.DataFrame(per_image_results)
    csv_complete_path = os.path.join(output_dir, "per_image_results_complete.csv")
    df_complete.to_csv(csv_complete_path, index=False)


def _print_and_save_comprehensive_summary(results1, results2, results_combined, statistics_summary, output_dir):
    """Print and save comprehensive evaluation summary."""
    print(f"\n{'='*80}")
    print("📋 COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Print dataset statistics
    dataset_stats = statistics_summary['dataset_stats']
    print(f"   Total Image Pairs: {dataset_stats['total_image_pairs']}")
    print(f"   Total Predictions (Combined): {dataset_stats['total_predictions_combined']}")
    print(f"   Total Ground Truth (Combined): {dataset_stats['total_gt_combined']}")
    print(f"   Avg Predictions per Image: {dataset_stats['avg_predictions_per_image']:.2f}")
    print(f"   Avg Ground Truth per Image: {dataset_stats['avg_gt_per_image']:.2f}")
    
    # Print mAP@0.5 results
    print(f"\n🎯 mAP@0.5 RESULTS:")
    if 'IoU_0.5' in statistics_summary:
        iou_05_stats = statistics_summary['IoU_0.5']
        for img_type in ['img1', 'img2', 'combined', 'avg']:
            metric_key = f'{img_type}_mAP'
            if metric_key in iou_05_stats:
                mean_map = iou_05_stats[metric_key]['mean']
                std_map = iou_05_stats[metric_key]['std']
                print(f"   {img_type.replace('_', ' ').title()}: {mean_map:.4f} ± {std_map:.4f}")
    
    # Print F1@0.5 results
    print(f"\n🎯 F1@0.5 RESULTS:")
    if 'IoU_0.5' in statistics_summary:
        iou_05_stats = statistics_summary['IoU_0.5']
        for img_type in ['img1', 'img2', 'combined', 'avg']:
            metric_key = f'{img_type}_f1'
            if metric_key in iou_05_stats:
                mean_f1 = iou_05_stats[metric_key]['mean']
                std_f1 = iou_05_stats[metric_key]['std']
                print(f"   {img_type.replace('_', ' ').title()}: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Print all IoU thresholds summary
    print(f"\n📈 PERFORMANCE ACROSS IoU THRESHOLDS:")
    for iou_threshold in [0.1, 0.3, 0.5]:  # Common thresholds
        if f'IoU_{iou_threshold}' in statistics_summary:
            print(f"\n   IoU {iou_threshold}:")
            iou_stats = statistics_summary[f'IoU_{iou_threshold}']
            
            # Show combined results
            if 'combined_mAP' in iou_stats:
                combined_map = iou_stats['combined_mAP']['mean']
                combined_f1 = iou_stats['combined_f1']['mean']
                combined_precision = iou_stats['combined_precision']['mean']
                combined_recall = iou_stats['combined_recall']['mean']
                
                print(f"     Combined - mAP: {combined_map:.4f}, F1: {combined_f1:.4f}, "
                      f"Precision: {combined_precision:.4f}, Recall: {combined_recall:.4f}")
    
    # Save comprehensive evaluation summary
    summary_path = os.path.join(output_dir, "comprehensive_evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Dataset statistics
        f.write("DATASET STATISTICS:\n")
        f.write("-"*20 + "\n")
        for key, value in dataset_stats.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Detailed results for each IoU threshold
        for iou_threshold in [0.1, 0.3, 0.5]:
            if f'IoU_{iou_threshold}' in statistics_summary:
                f.write(f"IoU THRESHOLD {iou_threshold}:\n")
                f.write("-"*20 + "\n")
                
                iou_stats = statistics_summary[f'IoU_{iou_threshold}']
                for img_type in ['img1', 'img2', 'combined', 'avg']:
                    f.write(f"\n{img_type.replace('_', ' ').title().upper()}:\n")
                    
                    for metric in ['mAP', 'recall', 'precision', 'f1']:
                        metric_key = f'{img_type}_{metric}'
                        if metric_key in iou_stats:
                            stats = iou_stats[metric_key]
                            f.write(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                                   f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}, "
                                   f"median: {stats['median']:.4f})\n")
                f.write("\n" + "="*50 + "\n\n")
    
    # Save statistics summary as JSON for programmatic access
    import json
    stats_json_path = os.path.join(output_dir, "statistics_summary.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    stats_serializable = convert_numpy_types(statistics_summary)
    
    with open(stats_json_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\n📄 Comprehensive summary saved: {summary_path}")
    print(f"📄 Statistics JSON saved: {stats_json_path}")
    print(f"{'='*80}")



# Keep original visualization functions for backward compatibility
def visualise_predictions_with_ground_truth(
    left_image,
    right_image,
    left_predicted_bboxes,
    right_predicted_bboxes,
    left_score,
    right_score,
    batch_metadata=None,
    image_pair_index=None,
    annotations_dir="annotations",
    model_size=224,
    save_path="./results_with_gt.png"
):
    """
    Visualize predictions alongside ground truth annotations.
    
    Args:
        left_image: Left input image tensor
        right_image: Right input image tensor
        left_predicted_bboxes: Predicted bounding boxes for left image
        right_predicted_bboxes: Predicted bounding boxes for right image
        left_score: Confidence scores for left predictions
        right_score: Confidence scores for right predictions
        batch_metadata: Metadata containing image paths (optional, for GT loading)
        image_pair_index: Index of current image pair in batch (optional, for GT loading)
        annotations_dir: Directory containing ground truth annotations
        model_size: Model input size for scale factor calculation
        save_path: Path to save the visualization
    """
    
    # Color scheme
    PREDICTED_COLOUR = "#FFC107"  # Yellow for predictions
    GT_COLOUR = "#E53935"         # Red for ground truth
    
    figure, plot = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get ground truth if metadata is provided
    left_gt_bboxes = None
    right_gt_bboxes = None
    
    if batch_metadata is not None and image_pair_index is not None:
        try:
            # Calculate scale factors for ground truth
            batch_item = batch_metadata.batch[image_pair_index]
            
            height1, width1 = get_height_width(batch_item.image1)
            height2, width2 = get_height_width(batch_item.image2)
            gt_scale_factor1 = model_size / max(height1, width1)
            gt_scale_factor2 = model_size / max(height2, width2)
            
            # Generate image IDs
            image1_base = os.path.splitext(os.path.basename(batch_item.image1))[0]
            print(image1_base)
            image1_id = image1_base 
            print(image1_id,image1_id+'.txt')
            if not os.path.exists(os.path.join(annotations_dir, image1_id+ '.txt')):
                image1_id = image1_base + "_combined_mask.png"

            # Image 2
            image2_base = os.path.splitext(os.path.basename(batch_item.image2))[0]
            image2_id = image2_base 
            if not os.path.exists(os.path.join(annotations_dir, image2_id+ '.txt')):
                image2_id = image2_base + "_combined_mask.png"

            
            # Load ground truth annotations
            left_gt_bboxes = load_annotations_for_image(image1_id, annotations_dir, gt_scale_factor1)
            right_gt_bboxes = load_annotations_for_image(image2_id, annotations_dir, gt_scale_factor2)
            
        except Exception as e:
            print(f"Warning: Could not load ground truth annotations: {e}")
    
    # Process left image
    plot[0].imshow(K.tensor_to_image(left_image))
    plot[0].set_title("Left Image: Predictions (Yellow) vs Ground Truth (Red)", fontsize=12, pad=20)
    
    # Draw ground truth bounding boxes first (so they appear behind predictions)
    if left_gt_bboxes is not None and len(left_gt_bboxes) > 0:
        for gt_bbox in left_gt_bboxes:
            # Scale ground truth bboxes to image size
            gt_scaled = gt_bbox.copy()
            gt_scaled[[0, 2]] = gt_scaled[[0, 2]] * (left_image.shape[-1] / model_size)
            gt_scaled[[1, 3]] = gt_scaled[[1, 3]] * (left_image.shape[-2] / model_size)
            
            w_gt = gt_scaled[2] - gt_scaled[0]
            h_gt = gt_scaled[3] - gt_scaled[1]
            
            rect_gt = patches.Rectangle(
                gt_scaled[:2], w_gt, h_gt, 
                linewidth=3, edgecolor=GT_COLOUR, facecolor="none", linestyle='--'
            )
            plot[0].add_patch(rect_gt)
            
            # Add "GT" label
            plot[0].text(
                gt_scaled[0], gt_scaled[1] - 15,
                'GT',
                color=GT_COLOUR,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
    
    # Draw predicted bounding boxes
    for i, bbox in enumerate(left_predicted_bboxes):
        bbox_scaled = bbox.copy()
        bbox_scaled[[0, 2]] = bbox_scaled[[0, 2]] * (left_image.shape[-1] / model_size)
        bbox_scaled[[1, 3]] = bbox_scaled[[1, 3]] * (left_image.shape[-2] / model_size)
        
        w = bbox_scaled[2] - bbox_scaled[0]
        h = bbox_scaled[3] - bbox_scaled[1]
        
        rect = patches.Rectangle(
            bbox_scaled[:2], w, h, 
            linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[0].add_patch(rect)
        
        if left_score is not None and i < len(left_score):
            confidence = left_score[i]
            plot[0].text(
                bbox_scaled[0], bbox_scaled[1] - 5,
                f'Pred: {confidence:.2f}',
                color=PREDICTED_COLOUR,
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7),
                verticalalignment='bottom'
            )
    
    plot[0].axis("off")
    
    # Process right image
    plot[1].imshow(K.tensor_to_image(right_image))
    plot[1].set_title("Right Image: Predictions (Yellow) vs Ground Truth (Red)", fontsize=12, pad=20)
    
    # Draw ground truth bounding boxes first
    if right_gt_bboxes is not None and len(right_gt_bboxes) > 0:
        for gt_bbox in right_gt_bboxes:
            # Scale ground truth bboxes to image size
            gt_scaled = gt_bbox.copy()
            gt_scaled[[0, 2]] = gt_scaled[[0, 2]] * (right_image.shape[-1] / model_size)
            gt_scaled[[1, 3]] = gt_scaled[[1, 3]] * (right_image.shape[-2] / model_size)
            
            w_gt = gt_scaled[2] - gt_scaled[0]
            h_gt = gt_scaled[3] - gt_scaled[1]
            
            rect_gt = patches.Rectangle(
                gt_scaled[:2], w_gt, h_gt, 
                linewidth=3, edgecolor=GT_COLOUR, facecolor="none", linestyle='--'
            )
            plot[1].add_patch(rect_gt)
            
            # Add "GT" label
            plot[1].text(
                gt_scaled[0], gt_scaled[1] - 15,
                'GT',
                color=GT_COLOUR,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
            )
    
    # Draw predicted bounding boxes
    for i, bbox in enumerate(right_predicted_bboxes):
        bbox_scaled = bbox.copy()
        bbox_scaled[[0, 2]] = bbox_scaled[[0, 2]] * (right_image.shape[-1] / model_size)
        bbox_scaled[[1, 3]] = bbox_scaled[[1, 3]] * (right_image.shape[-2] / model_size)
        
        w = bbox_scaled[2] - bbox_scaled[0]
        h = bbox_scaled[3] - bbox_scaled[1]
        
        rect = patches.Rectangle(
            bbox_scaled[:2], w, h, 
            linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[1].add_patch(rect)
        
        if right_score is not None and i < len(right_score):
            confidence = right_score[i]
            plot[1].text(
                bbox_scaled[0], bbox_scaled[1] - 5,
                f'Pred: {confidence:.2f}',
                color=PREDICTED_COLOUR,
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7),
                verticalalignment='bottom'
            )
    
    plot[1].axis("off")
    
    # Add legend
    legend_elements = [
        patches.Patch(color=PREDICTED_COLOUR, label='Predictions'),
        patches.Patch(color=GT_COLOUR, label='Ground Truth')
    ]
    figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
    
    plt.tight_layout()
    figure.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(figure)



def visualise_predictions(
    left_image,
    right_image,
    left_predicted_bboxes,
    right_predicted_bboxes,
    left_score,
    right_score,
    save_path="./results.png"
):
    """
    Original visualization function (kept for backward compatibility).
    """
    TARGET_COLOUR = "#1E88E5"
    PREDICTED_COLOUR = "#FFC107"
    figure, plot = plt.subplots(1, 2)
    
    plot[0].imshow(K.tensor_to_image(left_image))
    for i, bbox in enumerate(left_predicted_bboxes):
        bbox[[0, 2]] = bbox[[0, 2]] * (left_image.shape[-1] / 224)
        bbox[[1, 3]] = bbox[[1, 3]] * (left_image.shape[-2] / 224)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[0].add_patch(rect)
        if left_score is not None and i < len(left_score):
            confidence = left_score[i]
            text_x = bbox[0]
            text_y = bbox[1] - 5
            plot[0].text(
                text_x, text_y, f'{confidence:.2f}',
                color=PREDICTED_COLOUR,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='bottom'
            )
    plot[0].axis("off")
    
    plot[1].imshow(K.tensor_to_image(right_image))
    for i, bbox in enumerate(right_predicted_bboxes):
        bbox[[0, 2]] = bbox[[0, 2]] * (right_image.shape[-1] / 224)
        bbox[[1, 3]] = bbox[[1, 3]] * (right_image.shape[-2] / 224)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            bbox[:2], w, h, linewidth=2, edgecolor=PREDICTED_COLOUR, facecolor="none"
        )
        plot[1].add_patch(rect)
        if right_score is not None and i < len(right_score):
            confidence = right_score[i]
            text_x = bbox[0]
            text_y = bbox[1] - 5
            plot[1].text(
                text_x, text_y, f'{confidence:.2f}',
                color=PREDICTED_COLOUR,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='bottom'
            )
    plot[1].axis("off")
    
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


