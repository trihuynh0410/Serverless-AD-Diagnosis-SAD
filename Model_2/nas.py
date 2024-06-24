import json
import random
import nibabel as nib
import numpy as np

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import DatasetFolder

from nni.nas.evaluator.pytorch import ClassificationModule, Lightning, Trainer, DataLoader
from nni.nas.strategy import DARTS
from nni.nas.experiment import NasExperiment
from nni.nas.hub.pytorch.nasnet import NDSStageDifferentiable

from model_space import *

class AugmentTransform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __call__(self, img):
        return self.transforms(img)

class BalancedAugmentedDataset(Dataset):
    def __init__(self, dataset, target_count, pad_size=218, slice_range=(108, 110)):
        self.dataset = dataset
        self.target_count = target_count
        self.class_indices = self._get_class_indices()
        self.augment_transform = AugmentTransform()
        self.balanced_indices = self._balance_classes()
        self.pad_size = pad_size
        self.slice_range = slice_range
        self.slices_per_image = 3 * (self.slice_range[1] - self.slice_range[0])
     
    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _balance_classes(self):
        balanced_indices = []
        for label, indices in self.class_indices.items():
            original_count = len(indices)
            if original_count <= self.target_count:
                additional_indices = random.choices(indices, k=self.target_count - original_count)
                balanced_indices.extend(indices + additional_indices)
            else:
                selected_indices = random.sample(indices, self.target_count)
                balanced_indices.extend(selected_indices)
        random.shuffle(balanced_indices)
        return balanced_indices

    def __len__(self):
        return len(self.balanced_indices) * self.slices_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.slices_per_image
        slice_in_image = idx % self.slices_per_image
        axis_idx = slice_in_image // (self.slice_range[1] - self.slice_range[0])
        slice_idx = slice_in_image % (self.slice_range[1] - self.slice_range[0])

        original_idx = self.balanced_indices[img_idx]
        img_path, label = self.dataset.samples[original_idx]
        img = nib.load(img_path).get_fdata()

        # Pad the image if necessary
        padded_img = np.pad(img, ((0, self.pad_size - img.shape[0]), (0, self.pad_size - img.shape[1]), (0, self.pad_size - img.shape[2])), mode='constant')

        if axis_idx == 0:
            img_2d = padded_img[:, :, slice_idx - 25 + self.slice_range[0]]  # Axial slice
        elif axis_idx == 1:
            img_2d = padded_img[:, slice_idx + self.slice_range[0], :]  # Sagittal slice
        else:
            img_2d = padded_img[slice_idx - 5 + self.slice_range[0], :, :]  # Coronal slice

        img_2d = torch.tensor(img_2d, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        img_2d = self.augment_transform(img_2d)

        return img_2d, label

def dataloader_3d_to_2d(data_dir,target_count, train_ratio, batch_size=32, num_workers=32):

    full_dataset = DatasetFolder(
        root=data_dir, 
        loader=lambda x: x,
        extensions='.nii.gz'
    )

    balanced_augmented_dataset = BalancedAugmentedDataset(full_dataset, target_count)

    train_size = int(train_ratio * len(balanced_augmented_dataset))
    val_size = len(balanced_augmented_dataset) - train_size

    train_dataset, val_dataset = random_split(balanced_augmented_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader

class DartsClassificationModule(ClassificationModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.,
        auxiliary_loss_weight: float = 0.4,
        max_epochs: int = 600,
        num_classes: int = 3
    ):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # Training length will be used in LR scheduler
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False,num_classes=num_classes)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss."""
        x, y = batch
        if self.auxiliary_loss_weight:
            y_hat, y_aux = self(x)
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        # Set drop path probability before every epoch. This has no effect if drop path is not enabled in model.
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)

        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

if __name__ == '__main__':

    data_dir = '/workspace/data'

    train_loader, val_loader = dataloader_3d_to_2d(data_dir=data_dir,target_count=200,train_ratio=0.5)
    
    max_epochs = 70
    torch.set_float32_matmul_precision('high')

    evaluator = Lightning(
        DartsClassificationModule(0.025, 3e-5, 0., max_epochs),
        Trainer(
            accelerator='gpu', 
            devices=1,
            max_epochs=max_epochs,
        ),
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    strategy = DARTS(mutation_hooks=[NDSStageDifferentiable.mutate],gradient_clip_val=5.)

    model_space = MKNAS(
        width=16,
        num_cells=4,
        dataset='imagenet'
    )

    experiment = NasExperiment(model_space, evaluator, strategy)
    experiment.run()

    exported_arch = experiment.export_top_models(formatter='dict')[0]
    file_path = '/workspace/Serverless-AD-Diagnosis-SAD/Model/exported_arch.json'
    with open(file_path, 'w') as json_file:
        json.dump(exported_arch, json_file, indent=4)
    print(f'Exported architecture saved to {file_path}')