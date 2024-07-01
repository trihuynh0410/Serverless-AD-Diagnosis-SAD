import torch, random, gc, json, shutil, os, torchmetrics

from typing import Optional, Callable

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from skimage import exposure
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, random_split
from nni.nas.evaluator.pytorch import DataLoader, ClassificationModule

class AugmentTransform:
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.degrees = torch.rand(1).item() * 20 - 10  # -15 to 15 degrees
        self.translate = (torch.rand(1).item() * 0.1 - 0.05, torch.rand(1).item() * 0.1 - 0.05)  # -0.1 to 0.1
        self.scale = torch.rand(1).item() * 0.2 + 0.9  # 0.9 to 1.1
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
    def __init__(self, dataset, target_count):
        self.dataset = dataset
        self.target_count = target_count
        self.class_indices = self._get_class_indices()
        self.balanced_indices = self._balance_classes()

    def _get_class_indices(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _balance_classes(self):
        balanced_indices = []
        for _, indices in self.class_indices.items():
            original_count = len(indices)
            if original_count <= self.target_count:
                additional_indices = random.choices(indices, k=self.target_count - original_count)
                balanced_indices.extend(indices + additional_indices)
            else:
                selected_indices = random.sample(indices, self.target_count)
                balanced_indices.extend(selected_indices)
        return balanced_indices

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        original_idx = self.balanced_indices[idx]
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
                percentiles = [0.43, 0.44, 0.45, 0.55, 0.56, 0.57]
            elif axis_idx == 1:
                percentiles = [0.52, 0.53, 0.54, 0.55, 0.56]
            else:
                percentiles = [0.56, 0.57, 0.58, 0.59, 0.6]
            slice_indices = [min_slice + int((max_slice - min_slice) * percentile) for percentile in percentiles]
      
            for slice_idx in slice_indices:
                if axis_idx == 0:
                    img_2d = img_3d[slice_idx, :, :]  # Sagittal slice
                elif axis_idx == 1:
                    img_2d = img_3d[:, slice_idx, :]  # Coronal slice
                else:
                    img_2d = img_3d[:, :, slice_idx]  # Axial slice
                
                img_2d = np.pad(img_2d, ((0, 224 - img_2d.shape[0]), (0, 224 - img_2d.shape[1])), mode='constant')
                if original_idx in self.class_indices[label]:  # Original image                    
                    img_2d = exposure.equalize_hist(img_2d)
                    img_2d = torch.tensor(img_2d, dtype=torch.float32).unsqueeze(0)  # Add channel dimension    
                else:
                    img_2d = torch.tensor(img_2d, dtype=torch.float64).unsqueeze(0)  # Add channel dimension    
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
            num_classes: int = 3,
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
        self.train_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', task="multiclass")
        self.val_f1 = torchmetrics.F1Score(num_classes=num_classes, average='macro', task="multiclass")

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
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
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

  
def dataloader_3d_to_2d(data_dir, target_count, train_ratio, val_ratio, batch_size=32, num_workers=32):
    full_dataset = DatasetFolder(
        root=data_dir,
        loader=lambda x: x,
        extensions='.nii.gz'
    )

    balanced_augmented_dataset = BalancedAugmentedDataset(full_dataset, target_count)
    train_size = int(train_ratio * len(balanced_augmented_dataset))
    val_size = int(val_ratio * len(balanced_augmented_dataset))
    train_dataset, val_dataset = random_split(balanced_augmented_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True)  
    return train_loader, val_loader

def visualize_slices(batch, idx1, idx2):
    images, labels = batch
    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    for i in range(6):
        axs[0, i].imshow(images[idx1][i][0], cmap='gray')
        axs[0, i].set_title(f'Image {idx1}, Slice {i+1}, Label: {labels[idx1]}')
    for i in range(6):
        axs[1, i].imshow(images[idx2][i][0], cmap='gray')
        axs[1, i].set_title(f'Image {idx2}, Slice {i+1}, Label: {labels[idx2]}')
    
    plt.tight_layout()
    plt.show()

def count_files(directory,class_dirs):
    counts = {}
    for class_dir in class_dirs:
        class_path = os.path.join(directory, class_dir)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.endswith('.nii.gz')]
            counts[class_dir] = len(files)
        else:
            counts[class_dir] = 0
    return counts

def data_spliting(data_dir,test_dir,json_file_path):
    class_dirs = ['CN', 'MCI', 'Mild']
    test_files = {class_dir: [] for class_dir in class_dirs}

    # Create directories if they don't exist
    for class_dir in class_dirs:
        os.makedirs(os.path.join(data_dir, class_dir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)

    # Move files into respective class directories
    for filename in os.listdir(data_dir):
        if filename.endswith('.nii.gz') and os.path.isfile(os.path.join(data_dir, filename)):
            class_prefix = filename.split('_')[0]
            if class_prefix in class_dirs:
                src_path = os.path.join(data_dir, filename)
                dest_path = os.path.join(data_dir, class_prefix, filename)
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} to {dest_path}")
            else:
                print(f"Skipping {filename}, unknown class prefix")

    # Move 40 files from each class to the test directory
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        files = [f for f in os.listdir(class_path) if f.endswith('.nii.gz')]
        random.shuffle(files)
        test_class_path = os.path.join(test_dir, class_dir)
        
        for file in files[:40]:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(test_class_path, file)
            shutil.move(src_path, dest_path)
            test_files[class_dir].append(file)
            print(f"Moved {src_path} to {dest_path}")

    # Save the names of the test files in a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(test_files, json_file, indent=4)

    print(f"Test file names saved to {json_file_path}")

    print("File counts in /workspace/data:")
    data_counts = count_files(data_dir)
    for class_dir, count in data_counts.items():
        print(f"{class_dir}: {count}")

    print("\nFile counts in /workspace/test:")
    test_counts = count_files(test_dir)
    for class_dir, count in test_counts.items():
        print(f"{class_dir}: {count}")

    print("\nTotal counts:")
    for class_dir in class_dirs:
        total = data_counts[class_dir] + test_counts[class_dir]
        print(f"{class_dir}: {total}")