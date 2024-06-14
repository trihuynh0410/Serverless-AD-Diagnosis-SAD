import torch
from torch.utils.data import Dataset, random_split
from nni.nas.evaluator.pytorch import DataLoader
import random
import nibabel as nib
import numpy as np
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from skimage import exposure
import os
import shutil

# Define the base directory
data_dir = '/workspace/data'

# Define class directories
class_dirs = ['CN', 'MCI', 'Mild']
for class_dir in class_dirs:
    os.makedirs(os.path.join(data_dir, class_dir), exist_ok=True)

# List all files in the base directory
for filename in os.listdir(data_dir):
    # Check if the file has the correct format and is not a directory
    if filename.endswith('.nii.gz') and os.path.isfile(os.path.join(data_dir, filename)):
        # Determine the class from the prefix
        class_prefix = filename.split('_')[0]
        if class_prefix in class_dirs:
            # Define source and destination paths
            src_path = os.path.join(data_dir, filename)
            dest_path = os.path.join(data_dir, class_prefix, filename)
            # Move the file
            shutil.move(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")
        else:
            print(f"Skipping {filename}, unknown class prefix")

import torch
from torch.utils.data import Dataset, random_split
from nni.nas.evaluator.pytorch import DataLoader

import random
import nibabel as nib
import numpy as np
from torchvision import transforms
from torchvision.datasets import DatasetFolder

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
    def __init__(self, dataset, target_count, pad_size=218, slice_range=(104, 114)):
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
        # random.shuffle(balanced_indices)
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
data_dir = '/workspace/data'

full_dataset = DatasetFolder(
    root=data_dir, 
    loader=lambda x: x,
    extensions='.nii.gz'
)

target_count = 200
balanced_augmented_dataset = BalancedAugmentedDataset(full_dataset, target_count)

train_ratio = 0.5
train_size = int(train_ratio * len(balanced_augmented_dataset))
val_size = len(balanced_augmented_dataset) - train_size

train_dataset, val_dataset = random_split(balanced_augmented_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=32)
val_loader = DataLoader(val_dataset, batch_size=96, shuffle=True, num_workers=32)

print(len(train_loader))
import torch
from nni.nas.evaluator.pytorch import ClassificationModule

class DartsClassificationModule(ClassificationModule):
    def __init__(self, learning_rate=0.001, weight_decay=0., auxiliary_loss_weight=0.4, max_epochs=600, num_classes=3):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=True, num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-4)
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log('val_' + name, metric(y_hat, y), prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log('test_' + name, metric(y_hat, y), prog_bar=True, logger=True, on_epoch=True, sync_dist=True)

    def on_train_epoch_start(self):
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)
        if self.current_epoch % 5 ==0:
            print("lr is",self.trainer.optimizers[0].param_groups[0]['lr'])

from nni.nas.evaluator.pytorch import Lightning, Trainer, Classification
from model_space import *
from nni.nas.strategy import DARTS
from nni.nas.experiment import NasExperiment
from nni.nas.hub.pytorch.nasnet import NDSStageDifferentiable
import json
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

max_epochs = 200
torch.set_float32_matmul_precision('high')
# Define a checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    dirpath='checkpoints/',  # Directory where checkpoints will be saved
    filename='best-checkpoint',  # Checkpoint file name
    save_top_k=1,  # Save only the best checkpoint
    mode='min'  # Minimize validation loss
)

# Define an early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,  # Number of epochs to wait before stopping training
    mode='min'  # Minimize validation loss
)

evaluator = Lightning(
    DartsClassificationModule(0.02, 3e-5, 0., max_epochs, 3),
    Trainer(
        accelerator='gpu', 
        devices=4,
        max_epochs=max_epochs,
        fast_dev_run=False,
        accumulate_grad_batches = 2,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # overfit_batches=0.4,
        log_every_n_steps=50  # Add callbacks here
    ),
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

strategy = DARTS(mutation_hooks=[NDSStageDifferentiable.mutate],gradient_clip_val=5.)
with open('exported_arch.json', 'r') as f:
    exported_arch = json.load(f)
with model_context(exported_arch):
    final_model = MKNAS(
        width=36,
        num_cells=20,
        dataset='imagenet',
        # auxiliary_loss=True, 
        drop_path_prob=0.2
    )

evaluator.fit(final_model)