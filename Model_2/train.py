import json

from nni.nas.space import model_context
from nni.nas.evaluator.pytorch import Lightning, Trainer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger

from model_space import *
from utils import *

torch.set_float32_matmul_precision('medium')

data_dir = '/workspace/data'
train_loader, val_loader = dataloader_3d_to_2d(data_dir, 700, 0.8, 0.2, batch_size=32, num_workers=64)
csv_logger = CSVLogger('logs/', name='my_training')

checkpoint_callback = ModelCheckpoint(monitor='val_loss',  dirpath='checkpoints/', save_top_k=1,  mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15,  mode='min')
swa_callback = StochasticWeightAveraging(swa_lrs=1e-3)

max_epochs = 100
num_classes = 3

evaluator = Lightning(
    DartsClassificationModule(
        learning_rate = 1e-3, 
        weight_decay = 0., 
        max_epochs = max_epochs, 
        num_classes = num_classes,
        optimizer_class = torch.optim.SGD,
        optimizer_params = {'momentum': 0.9},
        scheduler_class= torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params= {'mode': 'min', 'factor': 0.5, 'patience': 5, 'min_lr':5}
    ),
    Trainer(
        accelerator='gpu', 
        devices=1, 
        max_epochs=max_epochs,
        logger=csv_logger,
        # fast_dev_run=True,
        accumulate_grad_batches = 4, 
        callbacks=[
            checkpoint_callback,
            swa_callback,
            early_stopping_callback
        ]
    ),
    train_dataloaders=train_loader, val_dataloaders=val_loader
)

with open('', 'r') as f:
    exported_arch = json.load(f)
with model_context(exported_arch):
    model = MobileViT(
        num_labels=3,
        num_slices_per_image = 16,
        base_widths = (8, 12, 16, 20, 24),
        dropout_rate = 0.3,
        width_mult = 3,
        embed_dim = 4,
        search_mlp_ratio = (2,1.5,2.5),
        search_num_heads=(3,4,5),
        search_depth= (4,5,6),  
    )
evaluator.fit(model)