import torch, random, gc, torchmetrics, math

from typing import Optional, Callable

import numpy as np
import nibabel as nib
import torchvision.transforms.functional as TF

from skimage import exposure
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
from nni.nas.evaluator.pytorch import DataLoader, ClassificationModule

class AugmentTransform:
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.degrees = torch.rand(1).item() * 60 - 30  
        self.translate = (torch.rand(1).item() * 0.1 - 0.05, torch.rand(1).item() * 0.1 - 0.05)
        self.scale = torch.rand(1).item() * 0.2 + 0.9 
        self.shear = 0 
        self.flip_h = torch.rand(1).item() > 0.5
        self.flip_v = torch.rand(1).item() > 0.5

    def __call__(self, img):
        img = TF.affine(img, self.degrees, self.translate, self.scale, self.shear, interpolation=TF.InterpolationMode.BILINEAR)
        if self.flip_h:
            img = TF.hflip(img)
        if self.flip_v:
            img = TF.vflip(img)
        return img


class BalancedAugmentedDataset(Dataset):
    def __init__(self, dataset, mode='train', model_type = 'chain',target_count=None):
        self.dataset = dataset
        self.mode = mode
        self.model_type = model_type
        self.target_count = target_count
        
        if self.mode == 'train' and self.target_count is not None:
            self.class_indices = self._get_class_indices()
            self.balanced_indices, self.is_duplicate = self._balance_classes()
        else:
            self.balanced_indices = self.dataset
            self.is_duplicate = [False] * len(self.dataset)

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _balance_classes(self):
        balanced_indices = []
        is_duplicate = []
        for _, indices in self.class_indices.items():
            original_count = len(indices)
            duplication_factor = math.ceil(self.target_count / original_count)
            duplicated_indices = indices * duplication_factor
            balanced_indices.extend(duplicated_indices[:self.target_count])
            is_duplicate.extend([False] * original_count + [True] * (self.target_count - original_count))        
        return balanced_indices, is_duplicate

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        original_idx = self.balanced_indices[idx]
        is_duplicate = self.is_duplicate[idx]
        img_path, label = self.dataset.samples[original_idx]
        img_3d = nib.load(img_path).dataobj

        data_range = np.nonzero(img_3d)
        min_slices = np.min(data_range, axis=1)
        max_slices = np.max(data_range, axis=1)

        del data_range
        gc.collect()

        seed = random.randint(0, 1_000_000)
        augment_transform = AugmentTransform(seed)

        slices = []

        for axis_idx in range(3):
            min_slice = min_slices[axis_idx]
            max_slice = max_slices[axis_idx]        
            if axis_idx == 0:
                percentiles = [0.43, 0.44, 0.45, 0.55, 0.56, 0.57] if self.model_type == 'chain' else [0.44, 0.56]
            elif axis_idx == 1:
                percentiles = [0.52, 0.53, 0.54, 0.55, 0.56] if self.model_type == 'chain' else [0.53, 0.55]
            else:
                percentiles = [0.56, 0.57, 0.58, 0.59, 0.6] if self.model_type == 'chain' else [0.57, 0.59]
            slice_indices = [min_slice + int((max_slice - min_slice) * percentile) for percentile in percentiles]
      
            for slice_idx in slice_indices:
                if axis_idx == 0:
                    img_2d = img_3d[slice_idx, :, :]  # Sagittal slice
                elif axis_idx == 1:
                    img_2d = img_3d[:, slice_idx, :]  # Coronal slice
                else:
                    img_2d = img_3d[:, :, slice_idx]  # Axial slice
                
                img_2d = np.pad(img_2d, ((0, 224 - img_2d.shape[0]), (0, 224 - img_2d.shape[1])), mode='constant')
                
                if not is_duplicate:  # Original image                    
                    img_2d = exposure.equalize_hist(img_2d)
                    img_2d = torch.tensor(img_2d, dtype=torch.float32).unsqueeze(0)  
                else:  # Duplicate image
                    img_2d = torch.tensor(img_2d, dtype=torch.float64).unsqueeze(0)  
                    img_2d = augment_transform(img_2d)
                    img_2d = img_2d.numpy().astype(np.float64)
                    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min())
                    img_2d = exposure.equalize_hist(img_2d)
                    img_2d = torch.tensor(img_2d, dtype=torch.float32)
                
                slices.append(img_2d)
        
        slices = torch.stack(slices) 

        return slices, label

  
class DartsClassificationModule(ClassificationModule):
    def __init__(self, 
            learning_rate: float = 0.001, 
            weight_decay: float = 0.,
            auxiliary_loss_weight: float = 0.,
            num_classes: int = 3,
            label_smoothing: bool = False,
            optimizer_class: Optional[Callable] = None,
            optimizer_params: Optional[dict] = None,
            scheduler_class: Optional[Callable] = None,
            scheduler_params: Optional[dict] = None
        ):
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=True, num_classes=num_classes)
        self.save_hyperparameters(ignore=['optimizer_class', 'scheduler_class', 'scheduler_params'])
        self.optimizer_class = optimizer_class or torch.optim.SGD
        self.optimizer_params = optimizer_params or {}
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params or {}
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.train_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', task="multiclass")
        self.val_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', task="multiclass")
        self.flex_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) if label_smoothing else self.criterion

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay,
            **self.optimizer_class
        )
        
        if self.scheduler_class:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.auxiliary_loss_weight:
            y_hat, y_aux = self(x)
            loss_main = self.flex_criterion(y_hat, y)
            loss_aux = self.flex_criterion(y_aux, y)
            self.log('train_loss_main', loss_main, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            self.log('train_loss_aux', loss_aux, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.flex_criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.log('train_f1', self.val_f1(y_hat, y), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log('val_' + name, metric(y_hat, y), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_f1', self.val_f1(y_hat, y), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

  
def dataloader_3d_to_2d(train_dir, val_dir, target_count, model_type, batch_size=32, num_workers=32):
    train_dataset = DatasetFolder(
        root=train_dir,
        loader=lambda x: x,
        extensions='.nii.gz'
    )

    val_dataset = DatasetFolder(
        root=val_dir,
        loader=lambda x: x,
        extensions='.nii.gz'
    )

    train_data = BalancedAugmentedDataset(train_dataset, mode='train', model_type=model_type, target_count=target_count)
    val_data = BalancedAugmentedDataset(val_dataset, mode='val', model_type=model_type)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, 
                                 num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, persistent_workers=True)
    return train_loader, val_loader