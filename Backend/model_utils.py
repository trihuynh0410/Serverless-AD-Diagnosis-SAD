# model_utils.py

import torch, json
import nibabel as nib
import numpy as np
from skimage import exposure
from torch.nn import functional as F
from nni.nas.space import model_context
from model import MobileKAN

def load_model():
    with open('cell_arch.json', 'r') as f:
        exported_arch = json.load(f)
    
    with model_context(exported_arch):
        model = MobileKAN(
            width=18,
            num_cells=10,
            dataset='imagenet',
            auxiliary_loss=True, 
            drop_path_prob=0.2,
        )
    
    model.load_state_dict(torch.load('cell_weight.ckpt', map_location=torch.device('cpu')))
    model.eval()
    return model

def load_and_preprocess_image(file_path):
    img_3d = nib.load(file_path).dataobj
    
    data_range = np.nonzero(img_3d)
    min_slices = np.min(data_range, axis=1)
    max_slices = np.max(data_range, axis=1)
    
    slices = []
    
    for axis_idx in range(3):
        min_slice = min_slices[axis_idx]
        max_slice = max_slices[axis_idx]
        if axis_idx == 0:
            percentiles = [0.43, 0.44, 0.45, 0.55, 0.56, 0.57]
        elif axis_idx == 1:
            percentiles = [0.52, 0.53, 0.54, 0.55, 0.56]
        else:
            percentiles = [0.56, 0.57, 0.58, 0.59, 0.6]
        slice_indices = [min_slice + int((max_slice - min_slice) * percentile) for percentile in percentiles]
        
        for slice_idx in slice_indices:
            if axis_idx == 0:
                img_2d = img_3d[slice_idx, :, :]
            elif axis_idx == 1:
                img_2d = img_3d[:, slice_idx, :]
            else:
                img_2d = img_3d[:, :, slice_idx]
            
            img_2d = np.asarray(img_2d)
            img_2d = np.pad(img_2d, ((0, 224 - img_2d.shape[0]), (0, 224 - img_2d.shape[1])), mode='constant')
            img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())
            img_2d = exposure.equalize_hist(img_2d)
            img_2d = torch.tensor(img_2d, dtype=torch.float32).unsqueeze(0)
            slices.append(img_2d)
    
    slices = torch.stack(slices)
    return slices.unsqueeze(0)

def predict(model, image):
    with torch.no_grad():
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.squeeze().tolist()

model = load_model()