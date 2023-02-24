
from share import *

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from training_datasets import HEDDataset
from training_datasets import DepthDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from sys import argv

if __name__ == '__main__':
    # Configs
    resume_path = './models/lycoris1000_control.ckpt'
    batch_size = 8
    logger_freq = 1000
    learning_rate = 1e-4 if argv[1] == 'hed' else 1e-5
    sd_locked = True
    only_mid_control = False

    torch.set_float32_matmul_precision('medium')
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    if argv[1] == 'depth':
        dataset = DepthDataset('/home/grapeot/co/Dreambooth-Anything/data/Lycoris', 'an anime in LycorisAnime style')
    elif argv[1] == 'hed':
        dataset = HEDDataset('/home/grapeot/co/Dreambooth-Anything/data/Lycoris', 'an anime in LycorisAnime style')
    else:
        raise RuntimeError("Lacks a parameter of model category.")
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(accelerator='gpu',
                         default_root_dir=('lightning_logs/depth' if argv[1] == 'depth' else 'lightning_logs/hed'),
                         devices=1,
                         precision=32,
                         callbacks=[logger],
                         accumulate_grad_batches=8,
                         max_epochs=50)


    # Train!
    trainer.fit(model, dataloader)
