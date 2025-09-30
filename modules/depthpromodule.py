from PIL import Image
import torch
import matplotlib.pyplot as plt
""" import rerun as rr
import rerun.blueprint as rrb """
from tqdm import tqdm
import os
from PIL.ExifTags import TAGS
import numpy as np
from torchvision.transforms.functional import to_pil_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def depth(image,image_processor,model):
    #image = Image.open(image_path)
    #image = to_pil_image(image) comment for compare
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)],
    )

    field_of_view = post_processed_output[0]["field_of_view"]
    focal_length = post_processed_output[0]["focal_length"]
    depth = post_processed_output[0]["predicted_depth"]
    depth = depth.detach().cpu().numpy()
    dp_im = np.clip(depth, 0, 100)
    image_depth = Image.fromarray(depth.astype("uint8"))
    
    return dp_im,focal_length

def get_exif_focal_length(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data and 37386 in exif_data:  # 37386 = FocalLength
        focal = exif_data[37386]
        if isinstance(focal, tuple):
            return focal[0] / focal[1] if focal[1] != 0 else None
        return float(focal)
    return None

def compare(im1, im2, image_processor, model):
    rerun_initialized = False
    # Profondeur + focale estimée
    im_depth1, focal1 = depth(im1, image_processor, model,rerun_initialized)
    im_depth2, focal2 = depth(im2, image_processor, model,rerun_initialized)
    # Focale en mm (valeur estimée par le modèle convertie)
    focal1_mm = focal1 *0.0021

    focal2_mm = focal2 *0.0021

    # Focale EXIF (si dispo)
    focal1_exif = get_exif_focal_length(im1)
    focal2_exif = get_exif_focal_length(im2)

    # Affichage
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(im_depth1, cmap='viridis')
    plt.axis('off')
    if focal1_exif:
        plt.title(f"Image 1\nFocale modèle: {focal1_mm:.2f} mm\nFocale EXIF: {focal1_exif:.2f} mm")
    else:
        plt.title(f"Image 1\nFocale modèle: {focal1_mm:.2f} mm")

    plt.subplot(1, 2, 2)
    plt.imshow(im_depth2, cmap='viridis')
    plt.axis('off')
    if focal2_exif:
        plt.title(f"Image 2\nFocale modèle: {focal2_mm:.2f} mm\nFocale EXIF: {focal2_exif:.2f} mm")
    else:
        plt.title(f"Image 2\nFocale modèle: {focal2_mm:.2f} mm")

    os.makedirs('./depth', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./depth/comparison.png')
    plt.show()
    return None

def unidepths(im1,model):
    print(f"Shape rgb1 NumPy: {im1.shape}")  # Devrait être (H, W, 3)
    rgb1_torch = torch.from_numpy(np.array(im1))
    print(f"Shape après permute: {rgb1_torch.shape}")  # Devrait être (3, H, W)
    predictions1 = model.infer(rgb1_torch)
    depth_pred1 = predictions1["depth"].squeeze().cpu().numpy()
    #depth_pred1=((depth_pred1-depth_pred1.min())/(depth_pred1.max()-depth_pred1.min()))
    #dp=depth_pred1/100
    #dp=np.clip(depth_pred1,0,100)
    #print(dp,type(dp))
    return depth_pred1