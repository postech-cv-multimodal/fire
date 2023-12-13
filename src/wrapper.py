import argparse
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from os.path import join as pjoin
from pytorch_lightning.core.lightning import LightningModule
from pytorch_metric_learning import distances, losses, miners, reducers

from transformers import InstructBlipProcessor
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from dataset import CUB_200_2011
from vit import VitForImageRetrieval
from instruct_blip import InstructBlipForImageRetrieval

class LightningWrapper(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(LightningWrapper, self).__init__()
        self.hparams = hparams
        
        self.processor = InstructBlipProcessor.from_pretrained(hparams.backbone)
        if hparams.model_type == 'vit+qformer':
            self.model = InstructBlipForImageRetrieval.from_pretrained(hparams.backbone)
        else:
            self.model = VitForImageRetrieval.from_pretrained(hparams.backbone)
        
        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=hparams.low)
        self.loss_func = losses.TripletMarginLoss(margin=hparams.margin, 
                                                  distance=self.distance, 
                                                  reducer=self.reducer)
        
        self.mining_func = miners.TripletMarginMiner(margin=hparams.margin, 
                                                     distance=self.distance, 
                                                     type_of_triplets=hparams.type_of_triplets)


    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size',
                            type=int,
                            default=32)
        parser.add_argument('--lr',
                            type=float,
                            default=5e-3,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        
        parser.add_argument('--margin',
                            type=float,
                            default=0.2,
                            help='for margin triplet loss')
        parser.add_argument('--low',
                            type=float,
                            default=0,
                            help='for threshold reducer')
        parser.add_argument('--type_of_triplets',
                            type=str,
                            default='semihard')
        return parser

    def training_step(self, batch, batch_idx):
        labels = batch['label'][:, 0].squeeze()
        out= self.model(pixel_values=batch['pixel_values'][:, 0, :, :].squeeze())
        
        hard_pairs = self.mining_func(out, labels)
        train_loss = self.loss_func(out, labels, hard_pairs)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label'][:, 0].squeeze()
        out= self.model(pixel_values=batch['pixel_values'][:, 0, :, :].squeeze())
        
        hard_pairs = self.mining_func(out, labels)
        val_loss = self.loss_func(out, labels, hard_pairs)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val_loss', torch.stack(avg_losses).mean(), prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        self.train_set = CUB_200_2011(processor=self.processor, 
                              vision_dir=self.hparams.vision_dir, 
                              data_path=pjoin(self.hparams.annot_dir, "data.json"),
                              attributes_path=self.hparams.attributes_path)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=4,
            shuffle=True)
        return train_dataloader
    
    def val_dataloader(self):
        self.valid_set = CUB_200_2011(processor=self.processor, 
                                    vision_dir=self.hparams.vision_dir, 
                                    data_path=pjoin(self.hparams.annot_dir, "val.json"),
                                    attributes_path=self.hparams.attributes_path)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=4)
        return val_dataloader

