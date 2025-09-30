# fix_uint16.py
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor as original_pil_to_tensor

def fixed_pil_to_tensor(pic):
    """Fixed version that handles uint16 and other types"""
    print(f"PIL image mode: {pic.mode}, size: {pic.size}")

    # Forcer un dtype numérique valide
    np_array = np.asarray(pic, dtype=np.float32 if pic.mode == 'I;16' else np.uint8)

    print(f"np_array type: {type(np_array)}")  # devrait afficher numpy.ndarray
    print(f"np_array dtype: {np_array.dtype}")  # float32 ou uint8 attendu

    tensor = torch.tensor(np_array.copy(), dtype=torch.float32)


    if pic.mode == 'I;16' or len(np_array.shape) == 2:
        print(f"Depth/grayscale image - keeping single channel")
        print(f"Final tensor shape: {tensor.shape}")
        return tensor

    elif len(tensor.shape) == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
        print(f"RGB image - converted to [C, H, W]")
        return tensor

    elif len(tensor.shape) == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
        print(f"Single channel image - removed channel dim")
        return tensor

    else:
        if len(np_array.shape) == 2:
            np_array = np.stack([np_array] * 3, axis=-1)
            tensor = torch.from_numpy(np_array).permute(2, 0, 1)
        print(f"Other image type - default handling")
        return tensor


# Monkey patch
import torchvision.transforms.functional
torchvision.transforms.functional.pil_to_tensor = fixed_pil_to_tensor