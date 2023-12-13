import os
import datetime
import argparse
import logging
import torch
import random
import warnings
import numpy as np
from os.path import join as pjoin

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from wrapper import LightningWrapper

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
        
warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--checkpoint_name_or_path",
        default="Salesforce/instructblip-flan-t5-xxl",
        type=str,
    )
    parser.add_argument(
        "--backbone",
        default="Salesforce/instructblip-flan-t5-xxl",
        type=str,
    )
    parser.add_argument(
        "--annot_dir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vision_dir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--attributes_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_type", 
        choices=['vit', 'vit+qformer'], 
        default='vit'
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=200
    )
    parser.add_argument(
        "--embedding_size", 
        type=int, 
        default=768
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    parser.add_argument(
        "--gpuid", 
        nargs='+', 
        type=int, 
        default=0
    )
    today = datetime.datetime.now()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=f"{today.strftime('%m%d')}_test"
    )

    parser = LightningWrapper.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    set_seed(args)
    logging.info(args)
    if len(args.gpuid) > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_ckpt',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            prefix=f'{args.model_name}_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = LightningWrapper(args)
        model.train()
        trainer = Trainer(
                    check_val_every_n_epoch=1, 
                    checkpoint_callback=checkpoint_callback, 
                    flush_logs_every_n_steps=100, 
                    gpus=args.gpuid, 
                    gradient_clip_val=1.0, 
                    log_every_n_steps=50, 
                    logger=True, 
                    max_epochs=args.max_epochs, 
                    num_processes=1)
                
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
