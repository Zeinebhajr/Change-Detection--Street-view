import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt


class BoundingBox:
    """Classe pour représenter une bounding box"""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 class_id: Union[int, str] = 0, confidence: float = 1.0, 
                 image_id: Optional[str] = None):
        self.x1 = x1
        self.y1 = y1  
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.confidence = confidence
        self.image_id = image_id
        
    def area(self) -> float:
        """Calcule l'aire de la bounding box"""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    def __repr__(self):
        return f"BBox({self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}, {self.class_id}, conf={self.confidence:.3f})"


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calcule l'IoU entre deux bounding boxes"""
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1.area()
    area2 = bbox2.area()
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


class mAPEvaluator:
    """Classe principale pour l'évaluation mAP"""
    
    def __init__(self, iou_thresholds: List[float] = [0.5], class_names: List[str] = None):
        self.iou_thresholds = iou_thresholds
        self.class_names = class_names or ["object"]
        self.predictions = []
        self.ground_truths = []
        
    def add_predictions_from_array(self, pred_array: np.ndarray, image_id: str, class_id: Union[int, str] = 0):
        """Ajoute des prédictions depuis un array numpy [N, 5] (x1, y1, x2, y2, score)"""
        if pred_array.size == 0:
            return
            
        for i in range(pred_array.shape[0]):
            bbox = BoundingBox(
                x1=float(pred_array[i, 0]),
                y1=float(pred_array[i, 1]),
                x2=float(pred_array[i, 2]),
                y2=float(pred_array[i, 3]),
                confidence=float(pred_array[i, 4]),
                class_id=class_id,
                image_id=image_id
            )
            self.predictions.append(bbox)
    
    def add_ground_truth_from_array(self, gt_array: np.ndarray, image_id: str, class_id: Union[int, str] = 0):
        """Ajoute des ground truth depuis un array numpy [N, 4] (x1, y1, x2, y2)"""
        if gt_array.size == 0:
            return
            
        for i in range(gt_array.shape[0]):
            bbox = BoundingBox(
                x1=float(gt_array[i, 0]),
                y1=float(gt_array[i, 1]),
                x2=float(gt_array[i, 2]),
                y2=float(gt_array[i, 3]),
                confidence=1.0,
                class_id=class_id,
                image_id=image_id
            )
            self.ground_truths.append(bbox)
    
    def calculate_ap_for_class(self, class_id: Union[int, str], iou_threshold: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calcule l'Average Precision pour une classe donnée à un seuil IoU"""
        
        class_predictions = [p for p in self.predictions if p.class_id == class_id]
        class_gt = [gt for gt in self.ground_truths if gt.class_id == class_id]
        
        if not class_predictions or not class_gt:
            return 0.0, np.array([]), np.array([])
        
        class_predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        num_predictions = len(class_predictions)
        tp = np.zeros(num_predictions)
        fp = np.zeros(num_predictions)
        
        gt_by_image = defaultdict(list)
        for gt in class_gt:
            gt_by_image[gt.image_id].append(gt)
        
        gt_matched = defaultdict(lambda: defaultdict(bool))
        
        for i, pred in enumerate(class_predictions):
            image_gts = gt_by_image.get(pred.image_id, [])
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(image_gts):
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        total_gt = len(class_gt)
        recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = self._calculate_ap_from_pr_curve(precisions, recalls)
        
        return ap, precisions, recalls
    
    def _calculate_ap_from_pr_curve(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Calcule l'AP depuis une courbe précision-rappel"""
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap
    
    def evaluate(self) -> Dict[str, Any]:
        """Évalue et retourne les métriques mAP"""
        results = {}
        
        all_classes = set()
        for p in self.predictions:
            all_classes.add(p.class_id)
        for gt in self.ground_truths:
            all_classes.add(gt.class_id)
        
        all_classes = sorted(list(all_classes))
        
        for iou_thresh in self.iou_thresholds:
            results[f'mAP@{iou_thresh}'] = {}
            aps = []
            
            for class_id in all_classes:
                ap, precisions, recalls = self.calculate_ap_for_class(class_id, iou_thresh)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                results[f'mAP@{iou_thresh}'][class_name] = {
                    'AP': ap,
                    'precision_curve': precisions,
                    'recall_curve': recalls
                }
                aps.append(ap)
            
            results[f'mAP@{iou_thresh}']['mean'] = np.mean(aps) if aps else 0.0
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Affiche les résultats de façon lisible"""
        print("\n" + "="*50)
        print("EVALUATION mAP RESULTS")
        print("="*50)
        
        for iou_key, iou_results in results.items():
            print(f"\n{iou_key}:")
            print("-" * 30)
            
            for class_name, metrics in iou_results.items():
                if class_name == 'mean':
                    print(f"Mean mAP: {metrics:.4f}")
                else:
                    print(f"{class_name}: AP = {metrics['AP']:.4f}")
    
    def plot_pr_curves(self, results: Dict[str, Any], save_path: str = None):
        """Plot les courbes précision-rappel"""
        fig, axes = plt.subplots(1, len(self.iou_thresholds), figsize=(6*len(self.iou_thresholds), 5))
        if len(self.iou_thresholds) == 1:
            axes = [axes]
        
        for i, (iou_key, iou_results) in enumerate(results.items()):
            ax = axes[i]
            
            for class_name, metrics in iou_results.items():
                if class_name != 'mean' and 'precision_curve' in metrics:
                    ax.plot(metrics['recall_curve'], metrics['precision_curve'], 
                           label=f"{class_name} (AP={metrics['AP']:.3f})")
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'P-R Curves {iou_key}')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def clear(self):
        """Reset les données"""
        self.predictions.clear()
        self.ground_truths.clear()