import json
import warnings
warnings.filterwarnings("ignore")

from model_space import *
from torch.utils.data import random_split
from torchvision import datasets, transforms
from nni.nas.evaluator.pytorch import DataLoader, Classification
from nni.nas.strategy import Proxyless as Proxyless
from nni.nas.experiment import NasExperiment

class CustomTransform:
    def __call__(self, sample):
        sample = F.to_tensor(sample)
        return F.normalize(sample, [0.5], [0.5])

transform = CustomTransform()

full_dataset = datasets.ImageFolder(root='./dataset', transform=transform)

train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

evaluator = Classification(
    accelerator='gpu', 
    devices=1,
    learning_rate=1e-3,
    weight_decay=1e-4,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    max_epochs=20,
    fast_dev_run=False,
    num_classes=3
)

model_space = ProxylessNAS(
    num_labels=3,
    base_widths=(4, 8, 12 ,16, 20, 24, 28, 32, 36),
    dropout_rate=0.5   
)

strategy = Proxyless()

experiment = NasExperiment(model_space, evaluator, strategy)
experiment.run()
exported_arch = experiment.export_top_models(formatter='dict')[0]

file_path = 'exported_arch.json'

with open(file_path, 'w') as json_file:
    json.dump(exported_arch, json_file, indent=4)

print(f'Exported architecture saved to {file_path}')