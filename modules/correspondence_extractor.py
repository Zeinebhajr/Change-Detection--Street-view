import torch
import torch.nn as nn
import kornia as K
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from modules.geometry import transform_points, convert_image_coordinates_to_world, sample_depth_for_given_points
import torch.nn.functional as F
import numpy as np
import sys
import os
import cv2

# Ajouter le chemin RoMa si nécessaire
roma_path = os.path.join(os.path.dirname(__file__), '..', 'RoMa')
if os.path.exists(roma_path) and roma_path not in sys.path:
    sys.path.insert(0, roma_path)

# Import conditionnel de RoMa pour éviter les importations circulaires
def _import_roma():
    try:
        # Essayer différentes méthodes d'import
        from romatch import roma_outdoor
        return roma_outdoor
    except ImportError:
        try:
            from RoMa.romatch.models.model_zoo import roma_outdoor
            return roma_outdoor
        except ImportError:
            try:
                from romatch.models import roma_outdoor
                return roma_outdoor
            except ImportError:
                raise ImportError("Impossible d'importer RoMa. Vérifiez l'installation et la structure du package.")

class CorrespondenceExtractor(nn.Module):
    def __init__(self, roma=False, nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024, 
                 superglue="outdoor", sinkhorn_iterations=20, match_threshold=0.2, resize=640,use_magsac=True, magsac_confidence=0.99, magsac_max_iters=10000):
        super().__init__()
        
        self.use_roma = roma
        self._resize = K.augmentation.Resize(resize, side="long").eval()
        self.use_magsac = use_magsac
        self.magsac_confidence = magsac_confidence  
        self.magsac_max_iters = magsac_max_iters
        # Initialiser SuperGlue/SuperPoint si roma=False
        if not self.use_roma:
            config = {
                'superpoint': {
                    'nms_radius': nms_radius,
                    'keypoint_threshold': keypoint_threshold,
                    'max_keypoints': max_keypoints
                },
                'superglue': {
                    'weights': superglue,
                    'sinkhorn_iterations': sinkhorn_iterations,
                    'match_threshold': match_threshold,
                }
            }
            self._matching = Matching(config).eval()
        else:
            self._matching = None
        
        # Initialiser RoMa si roma=True (import différé)
        if self.use_roma:
            try:
                roma_outdoor = _import_roma()
                self.roma_model = roma_outdoor(device='cuda')
                self.roma_model.eval()
                print("RoMa initialisé avec succès")
            except ImportError as e:
                print(f"Erreur lors de l'import de RoMa: {e}")
                print("Utilisation de SuperGlue à la place")
                self.use_roma = False
                self.roma_model = None
                # Fallback vers SuperGlue
                config = {
                    'superpoint': {
                        'nms_radius': nms_radius,
                        'keypoint_threshold': keypoint_threshold,
                        'max_keypoints': max_keypoints
                    },
                    'superglue': {
                        'weights': superglue,
                        'sinkhorn_iterations': sinkhorn_iterations,
                        'match_threshold': match_threshold,
                    }
                }
                self._matching = Matching(config).eval()
        else:
            self.roma_model = None

    @torch.no_grad()
    def forward(self, batch):
        batch_points1 = []
        batch_points2 = []
        
        for i in range(len(batch["image1"])):
            # Skip si pas besoin de calculer les correspondances
            if (batch["registration_strategy"][i] == "identity" or 
                batch["intrinsics1"][i] is not None or 
                batch["transfm2d_1_to_2"][i] is not None):
                batch_points1.append(None)
                batch_points2.append(None)
                continue
            
            # Nettoyage mémoire
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if hasattr(torch.backends.cudnn, 'reset_peak_memory_stats'):
                torch.backends.cudnn.reset_peak_memory_stats()
            
            # Préparation des images
            img1 = batch["image1"][i]  # Format RGB
            img2 = batch["image2"][i]  # Format RGB
            
            if self.use_roma:
                # Utiliser RoMa
                kpts1, kpts2 = self._extract_correspondences_roma(img1, img2)
            else:
                # Utiliser SuperGlue + SuperPoint
                kpts1, kpts2 = self._extract_correspondences_superglue(img1, img2)
            
            print(f'Nombre de points extraits: {len(kpts1)} correspondances')
            
            # Filtrage RANSAC
            if self.use_magsac:
                kpts1, kpts2 = filter_out_bad_correspondences_using_magsac(
                    batch["registration_strategy"][i], kpts1, kpts2, 
                    batch["depth1"][i], batch["depth2"][i],
                    confidence=self.magsac_confidence,
                    max_iters=self.magsac_max_iters
                )
                print('MAGSAC')
            else:
                kpts1, kpts2 = filter_out_bad_correspondences_using_ransac(
                    batch["registration_strategy"][i], kpts1, kpts2, 
                    batch["depth1"][i], batch["depth2"][i]
                )
            batch_points1.append(kpts1)
            batch_points2.append(kpts2)
                    
        batch["points1"] = batch_points1
        batch["points2"] = batch_points2
        return batch
    
    def _extract_correspondences_roma(self, img1, img2):
        """Extraction avec RoMa"""
        from PIL import Image
        import numpy as np
        
        # Sauvegarder le device original
        original_device = img1.device
        
        # Convertir les tenseurs PyTorch en images PIL RGB
        def tensor_to_pil_rgb(tensor):
            # Si c'est un tenseur [C, H, W]
            if tensor.dim() == 3:
                if tensor.shape[0] == 1:  # Grayscale -> RGB
                    tensor = tensor.repeat(3, 1, 1)
                # Convertir en [H, W, C]
                img_np = tensor.permute(1, 2, 0).cpu().numpy()
            # Si c'est un tenseur [H, W]
            elif tensor.dim() == 2:
                img_np = tensor.cpu().numpy()
                # Ajouter dimension channel pour RGB
                img_np = np.stack([img_np, img_np, img_np], axis=2)
            else:
                raise ValueError(f"Format de tenseur non supporté: {tensor.shape}")
            
            # Normaliser en [0, 255] pour PIL
            if img_np.max() <= 1.0:
                print("=======================================================")
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Créer image PIL RGB
            return Image.fromarray(img_np, mode='RGB')
        
        # Convertir en images PIL
        pil_img1 = tensor_to_pil_rgb(img1)
        pil_img2 = tensor_to_pil_rgb(img2)
        
        # Obtenir les dimensions
        W_A, H_A = pil_img1.size  # PIL: (width, height)
        W_B, H_B = pil_img2.size
        
        print(f"RoMa PIL images: img1={pil_img1.mode} {pil_img1.size}, img2={pil_img2.mode} {pil_img2.size}")
        
        # Matching avec RoMa (passer les images PIL)
        warp, certainty = self.roma_model.match(pil_img1, pil_img2, device='cuda')
        
        # Échantillonner les correspondances
        #matches, certainty = self.roma_model.sample(warp, certainty)
        """ min_cert = 0.05
        mask = certainty > min_cert

        # Obtenez les coordonnées des pixels qui passent le seuil
        coords_y, coords_x = torch.where(mask)

        # Récupérez les coordonnées correspondantes dans warp
        matches_filtered = []
        for i in range(len(coords_y)):
            y, x = coords_y[i], coords_x[i]
            # warp[y, x] contient [x_A, y_A, x_B, y_B] en coordonnées normalisées
            match = warp[y, x]  # shape: (4,)
            matches_filtered.append(match)

        matches = torch.stack(matches_filtered)  # shape: (N, 4)
        certainty_filtered = certainty[mask] """

        # Maintenant sample() fonctionne
        matches, certainty_filtered = self.roma_model.sample(warp, certainty)

        # Convert to pixel coordinates
        kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        
        # S'assurer que les points sont sur le bon device
        kpts1 = kpts1.to(original_device)
        kpts2 = kpts2.to(original_device)
        
        # Normaliser les coordonnées (0-1)
        kpts1[:, 0] /= W_A
        kpts1[:, 1] /= H_A
        kpts2[:, 0] /= W_B
        kpts2[:, 1] /= H_B
        
        # Trier par certitude
        if certainty is not None and len(certainty) > 0:
            certainty = certainty.to(original_device)
            conf_values = certainty.flatten() if certainty.dim() > 1 else certainty
            if len(conf_values) == len(kpts1):
                conf, sort_idx = conf_values.sort(descending=True)
                kpts1 = kpts1[sort_idx]
                kpts2 = kpts2[sort_idx]
        
        print('with roma')
        return kpts1, kpts2
    
    def _extract_correspondences_superglue(self, img1, img2):
        """Extraction avec SuperGlue + SuperPoint"""
        # Conversion en niveaux de gris
        inp1 = K.color.rgb_to_grayscale(img1).unsqueeze(0)
        inp2 = K.color.rgb_to_grayscale(img2).unsqueeze(0)
        
        # Redimensionnement
        fresh_resize = K.augmentation.Resize(640, side="long")
        inp1 = fresh_resize(inp1)
        inp2 = fresh_resize(inp2)
        
        # Matching
        pred = self._matching({'image0': inp1, 'image1': inp2})
        
        # Nettoyage mémoire
        torch.cuda.empty_cache()
        
        # Extraction des points clés et correspondances
        kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]
        matches, conf = pred['matches0'][0], pred['matching_scores0'][0]
        
        # Normalisation des coordonnées
        scale_1 = torch.tensor(inp1.shape[-2:]).flip(dims=(0,))
        scale_2 = torch.tensor(inp2.shape[-2:]).flip(dims=(0,))
        kpts1 /= scale_1
        kpts2 /= scale_2
        
        # Filtrage des correspondances valides
        valid = matches != -1
        conf = conf[valid]
        kpts1 = kpts1[valid]
        kpts2 = kpts2[matches[valid]]
        
        # Tri par confiance
        conf, sort_idx = conf.sort(descending=True)
        kpts1 = kpts1[sort_idx]
        kpts2 = kpts2[sort_idx]
        
        return kpts1, kpts2


def inliers_using_ransac(X, Y, n_iters=500):
    """RANSAC pour trouver les inliers"""
    best_inliers = None
    best_fit_error = None
    threshold = torch.median(torch.abs(Y - torch.median(Y)))
    
    for _ in range(n_iters):
        # Estimation de la transformation
        sample_indices = np.random.choice(np.arange(X.shape[0]), size=min(50, X.shape[0]), replace=False)
        sample_X = X[sample_indices]
        sample_Y = Y[sample_indices]
        
        X_ = F.pad(sample_X, (0, 1), value=1)
        Y_ = F.pad(sample_Y, (0, 1), value=1)
        X_pinv = torch.linalg.pinv(X_)
        M = torch.einsum("ij,jk->ki", X_pinv, Y_)
        
        # Recherche des inliers
        X_warped = transform_points(M, X)
        fit_error = torch.sum(torch.abs(X_warped - Y), dim=1)
        inliers = (fit_error < threshold).nonzero().squeeze()
        fit_error_of_inliers = fit_error[inliers].sum()
        
        if (best_fit_error is None or fit_error_of_inliers < best_fit_error) and torch.numel(inliers) >= 10:
            best_fit_error = fit_error_of_inliers
            best_inliers = (fit_error < threshold).nonzero().squeeze()
    
    if best_inliers is None:
        return torch.ones(X.shape[0]).bool()
    return best_inliers


def filter_out_bad_correspondences_using_ransac(registration_strategy, points1, points2, depth1=None, depth2=None):
    """Filtrage des correspondances avec RANSAC"""
    if registration_strategy == "3d":
        assert depth1 is not None and depth2 is not None
        
        # S'assurer que tous les tenseurs sont sur le même device
        device = points1.device
        if depth1.device != device:
            depth1 = depth1.to(device)
        if depth2.device != device:
            depth2 = depth2.to(device)
        if points2.device != device:
            points2 = points2.to(device)
            
        X = convert_image_coordinates_to_world(
            image_coords=points1.unsqueeze(0),
            depth=sample_depth_for_given_points(depth1.unsqueeze(0), points1.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points1).unsqueeze(0).to(device),
            Rt=torch.eye(4).type_as(points1).unsqueeze(0).to(device),
        ).squeeze(0)
        Y = convert_image_coordinates_to_world(
            image_coords=points2.unsqueeze(0),
            depth=sample_depth_for_given_points(depth2.unsqueeze(0), points2.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points2).unsqueeze(0).to(device),
            Rt=torch.eye(4).type_as(points2).unsqueeze(0).to(device),
        ).squeeze(0)
    elif registration_strategy == "2d":
        # S'assurer que points1 et points2 sont sur le même device
        device = points1.device
        if points2.device != device:
            points2 = points2.to(device)
        X = points1
        Y = points2
    else:
        raise NotImplementedError()
    
    inliers = inliers_using_ransac(X, Y)
    points1 = points1[inliers]
    points2 = points2[inliers]
    return points1, points2
def inliers_using_magsac(X, Y, img_shape=None, confidence=0.99, max_iters=10000):
    """MAGSAC++ pour trouver les inliers avec homographie"""
    if len(X) < 4:
        return torch.ones(X.shape[0]).bool()
    
    X_np = X.detach().cpu().numpy().astype(np.float32)
    Y_np = Y.detach().cpu().numpy().astype(np.float32)
    
    # Si coordonnées normalisées [0,1], les convertir avec les vraies dimensions
    if X_np.max() <= 1.0 and img_shape is not None:
        # img_shape doit être (height, width)
        X_np[:, 0] *= img_shape[1]  # width
        X_np[:, 1] *= img_shape[0]  # height
        Y_np[:, 0] *= img_shape[1]  # width  
        Y_np[:, 1] *= img_shape[0]  # height
    
    try:
        H, mask = cv2.findHomography(
            X_np, Y_np,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=1.0,
            confidence=confidence,
            maxIters=max_iters
        )
        
        if mask is not None:
            return torch.from_numpy(mask.flatten().astype(bool))
        else:
            return torch.ones(X.shape[0]).bool()
    except:
        return torch.ones(X.shape[0]).bool()

def filter_out_bad_correspondences_using_magsac(registration_strategy, points1, points2, 
                                                depth1=None, depth2=None, confidence=0.99, max_iters=10000):
    """Filtrage des correspondances avec MAGSAC++"""
    if registration_strategy == "3d":
        # Même code que RANSAC mais appeler inliers_using_magsac à la fin
        assert depth1 is not None and depth2 is not None
        device = points1.device
        if depth1.device != device:
            depth1 = depth1.to(device)
        if depth2.device != device:
            depth2 = depth2.to(device)
        if points2.device != device:
            points2 = points2.to(device)
            
        X = convert_image_coordinates_to_world(
            image_coords=points1.unsqueeze(0),
            depth=sample_depth_for_given_points(depth1.unsqueeze(0), points1.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points1).unsqueeze(0).to(device),
            Rt=torch.eye(4).type_as(points1).unsqueeze(0).to(device),
        ).squeeze(0)
        Y = convert_image_coordinates_to_world(
            image_coords=points2.unsqueeze(0),
            depth=sample_depth_for_given_points(depth2.unsqueeze(0), points2.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points2).unsqueeze(0).to(device),
            Rt=torch.eye(4).type_as(points2).unsqueeze(0).to(device),
        ).squeeze(0)
        img_shape = depth1.shape[-2:]
        inliers = inliers_using_magsac(X[:, :2], Y[:, :2], img_shape, confidence, max_iters)  # Projeter en 2D
    elif registration_strategy == "2d":
        device = points1.device
        if points2.device != device:
            points2 = points2.to(device)
        X = points1
        Y = points2
        inliers = inliers_using_magsac(X, Y,None, confidence, max_iters)
    else:
        raise NotImplementedError()
    
    points1 = points1[inliers]
    points2 = points2[inliers]
    return points1, points2