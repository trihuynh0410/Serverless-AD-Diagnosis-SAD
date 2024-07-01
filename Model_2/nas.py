import json

from nni.nas.evaluator.pytorch import Lightning, Trainer
from nni.nas.strategy import DARTS
from nni.nas.experiment import NasExperiment
from pytorch_lightning.loggers import CSVLogger

from model_space import *
from utils import *

torch.set_float32_matmul_precision('medium')

data_dir = '/workspace/data'
train_loader, val_loader = dataloader_3d_to_2d(data_dir, 300, 0.5, 0.5, batch_size=16, num_workers=64)
csv_logger = CSVLogger('logs/', name='my_experiment')

max_epochs = 20
num_classes = 3

evaluator = Lightning(
    DartsClassificationModule(
        learning_rate = 1e-3, 
        weight_decay = 0., 
        max_epochs = max_epochs, 
        num_classes = num_classes,
        optimizer_class = torch.optim.SGD,
        optimizer_params = {'momentum': 0.9},
        scheduler_class= torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params= {'T_max': max_epochs, 'eta_min': 1e-4}
    ),
    Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=max_epochs,
        logger=csv_logger
        # fast_dev_run=True,
    ),
    train_dataloaders=train_loader, val_dataloaders=val_loader
)

strategy = DARTS(mutation_hook = [MixedAbsolutePositionEmbedding.mutate, MixedClassToken.mutate], gradient_clip_val=3.0)

model = MobileViT(
    num_labels=num_classes,
    num_slices_per_image = 16,
    base_widths = (6, 8, 10, 12, 14),
    dropout_rate = 0.1,
    width_mult = 1.5,
    embed_dim = 2,
    search_mlp_ratio = (2,1.5),
    search_num_heads=(3,4),
    search_depth= (4,5,6),    
)

experiment = NasExperiment(model, evaluator, strategy)
experiment.run()
exported_arch = experiment.export_top_models(formatter='dict')[0]
file_path = ''
with open(file_path, 'w') as json_file:
    json.dump(exported_arch, json_file, indent=4)
print(f'Exported architecture saved to {file_path}')