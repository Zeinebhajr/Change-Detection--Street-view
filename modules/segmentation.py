import os
from pathlib import Path
import requests
import argparse
import sys

from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import cv2


def load_model(model_name="facebook/mask2former-swin-large-mapillary-vistas-panoptic", device='cpu'):
    """Load the Mask2Former model and processor."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def load_image(image_path):
    """Load an image from the given path."""
    image = Image.open(image_path).convert('RGB')
    return image


def run_inference(processor, model, image, size, device='cpu'):
    """Run inference on the image and return panoptic segmentation output."""
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_panoptic_segmentation(outputs, target_sizes=size)[0]



def save_image(np_img, out_path):
    Image.fromarray(np_img.astype(np.uint8)).save(out_path)
def save_mask_file(mask,out_path):
    np.save(out_path,mask)
def RGB_to_BRG(img) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def clahe_color(img, is_rgb: bool = True) -> np.ndarray:
    if is_rgb:
        img_bgr = RGB_to_BRG(img)
    else:
        img_bgr = img
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
def segment_image(img,out_dir,i,device='cuda',with_clahe=True):
    TARGET_CLASSES = {
    # Classes existantes
    "building": {
        "labels": {"building"},
        "color": [255, 0, 0]  # Rouge pour les bâtiments
    },
    "traffic_light": {
        "labels": {"traffic light"},
        "color": [0, 255, 0]  # Vert pour les feux
    },
    
    # SIGNALISATION
    "traffic_sign": {
        "labels": {"traffic sign (front)", "traffic sign (back)"},
        "color": [255, 165, 0]  # Orange
    },
    "traffic_sign_support": {
        "labels": {"traffic sign frame"},
        "color": [75, 0, 130]
    },
    
    # SUPPORTS ET POTEAUX
    "support_pole": {
        "labels": {"pole", "utility pole"},
        "color": [128, 0, 128]  # Violet
    },
    "traffic_light_support": {
        "labels": {"traffic light support", "traffic lights support"},
        "color": [75, 0, 130]  # Indigo
    },
    
    # SURFACES PLATES (FLAT)
    "bike_lane": {
        "labels": {"bike lane"},
        "color": [0, 128, 0]  # Vert foncé
    },
    "service_lane": {
        "labels": {"service lane"},
        "color": [128, 128, 0]  # Olive
    },
    "rail_track": {
        "labels": {"rail track"},
        "color": [139, 69, 19]  # Marron
    },
    
    # STRUCTURES
    "structure_building": {
        "labels": {"building"},
        "color": [255, 0, 0]  # Rouge
    },
    "bridge": {
        "labels": {"bridge"},
        "color": [160, 82, 45]  # Brun
    },
    "tunnel": {
        "labels": {"tunnel"},
        "color": [47, 79, 79]  # Gris ardoise foncé
    },
    
    # MARQUAGES
    "marking_general": {
        "labels": {"lane marking - general"},
        "color": [255, 255, 224]  # Crème
    },
    "marking_crosswalk_zebra": {
        "labels": {"lane marking - crosswalk"},
        "color": [0, 0, 0]  # Noir
    },
    
    # OBJETS URBAINS
    "street_light": {
        "labels": {"street light"},
        "color": [255, 215, 0]  # Or
    },
    "object_traffic_light": {
        "labels": {"traffic light"},
        "color": [50, 205, 50]  # Vert lime
    },
    "banner": {
        "labels": {"banner"},
        "color": [220, 20, 60]  # Cramoisi
    },
    "billboard": {
        "labels": {"billboard"},
        "color": [255, 20, 147]  # Rose vif
    },
    "junction_box": {
        "labels": {"junction box"},
        "color": [184, 134, 11]  # Jaune foncé
    },
    "fire_hydrant": {
        "labels": {"fire hydrant"},
        "color": [255, 69, 0]  # Rouge-orange
    },
    "bench": {
        "labels": {"bench"},
        "color": [210, 180, 140]  # Tan
    },
    "bike_rack": {
        "labels": {"bike rack"},
        "color": [0, 100, 0]  # Vert foncé
    },
    "mailbox": {
        "labels": {"mailbox"},
        "color": [30, 144, 255]  # Bleu Dodger
    },
    "pothole": {
        "labels": {"pothole"},
        "color": [139, 0, 0]  # Rouge foncé
    },
    
    # BARRIÈRES
    "fence": {
        "labels": {"fence"},
        "color": [160, 160, 160]  # Gris
    },
    "wall": {
        "labels": {"wall"},
        "color": [188, 143, 143]  # Brun rosé
    },
    "other_barrier": {
        "labels": {"barrier"},
        "color": [205, 133, 63]  # Brun sable
    },
    "guard_rail": {
        "labels": {"guard rail"},
        "color": [176, 196, 222]  # Bleu acier clair
    }
}
    processor, model = load_model(device=device)

    # Mapper les labels aux IDs pour chaque classe
    id2label = model.config.id2label
    """ class_ids = {}
    for class_name, class_config in TARGET_CLASSES.items():
        class_ids[class_name] = {
            id for id, label in id2label.items() 
            if label.lower() in class_config["labels"]
        } """
    labelid2class = {}
    for class_name, class_cfg in TARGET_CLASSES.items():
        for id_, label in id2label.items():
            if label.lower() in class_cfg["labels"]:
                labelid2class[id_] = class_name

    # Assign a unique pixel value to each class (1,2,3,...)
    class2val = {cls: idx+1 for idx, cls in enumerate(TARGET_CLASSES.keys())}

    print("Début du traitement des images...")
        # Charger et préprocesser l'image

    if with_clahe==True:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_clahe_bgr = clahe_color(img_bgr)
        img = Image.fromarray(cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB))
        
        # Prédiction
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
        )[0]
        
    seg = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]
        
    # Sauvegarder la segmentation d'instance complète
    instance_img = np.zeros(seg.shape, dtype=np.uint8)
    os.makedirs(out_dir+'/mask',exist_ok=True)
    #print(model.config.id2label)
    for s in segments_info:
        #print(s, "=>", id2label[s["label_id"]])
        if s["label_id"] in labelid2class:   # ✅ filter unwanted classes
            class_name = labelid2class[s["label_id"]]
            val = class2val[class_name]      # assign unique value
            mask = seg == s["id"]
            instance_img[mask] = 1

    # --- SAVE FINAL MASK ---
    out_path = Path(out_dir) / f"mask{i}.png"
    #save_image(instance_img, out_path)
    Image.fromarray(instance_img.astype(np.uint8)).save(out_path)
    print(f"[INFO] Mask saved to {out_path} | shape={instance_img.shape}")
    plt.figure(figsize=(10, 10))
    plt.imshow(img)  # image originale
    plt.imshow(instance_img, alpha=0.5, cmap='jet')  # masque superposé
    plt.axis('off')

    # Sauvegarde
    out_path = f"mask_overlay{i}.png"
    plt.savefig(Path(out_dir) /out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return instance_img