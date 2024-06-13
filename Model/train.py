import json
from nni.nas.space import model_context
from model_space import *
from nas import *

train_loader, val_loader = 0
exported_arch = 0
with model_context(exported_arch):
    final_model = MKNAS(
        width=36,
        num_cells=20,
        dataset='imagenet',
        auxiliary_loss=True, 
        drop_path_prob=0.2
    )
max_epochs = 600

evaluator = Lightning(
    DartsClassificationModule(0.025, 3e-4, 0.4, max_epochs),
    trainer=Trainer(
        accelerator='gpu', 
        devices=4,
        gradient_clip_val=5.,
        max_epochs=max_epochs,
    ),
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

evaluator.fit(final_model)