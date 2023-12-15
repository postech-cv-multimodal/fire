import argparse
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from os.path import join as pjoin
from einops import rearrange, reduce, repeat

from pytorch_lightning.core.lightning import LightningModule
from pytorch_metric_learning import distances, losses, miners, reducers

from transformers import InstructBlipProcessor

from dataset import CUB_200_2011
from vit import VitForImageRetrieval
from instruct_blip import InstructBlipForImageRetrieval

from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel


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
        self.margin_loss = losses.TripletMarginLoss(margin=hparams.margin, 
                                                  distance=self.distance, 
                                                  reducer=self.reducer)
        
        self.pa_loss = losses.ProxyAnchorLoss(num_classes=hparams.num_classes, 
                                              embedding_size=hparams.embedding_size, 
                                              margin=hparams.margin_pos, 
                                              alpha=hparams.alpha)
        
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
                            default=1.2,
                            help='for margin triplet loss')
        parser.add_argument('--margin_pos',
                            type=float,
                            default=1.8)
        parser.add_argument('--margin_neg',
                            type=float,
                            default=2.2)
        parser.add_argument('--alpha',
                            type=float,
                            default=16)
        parser.add_argument('--low',
                            type=float,
                            default=0,
                            help='for threshold reducer')
        parser.add_argument('--type_of_triplets',
                            type=str,
                            default='semihard')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.0001,)
        return parser

    def get_latent_feature_for_vitqformer(self, batch):
        # pixel_values (bs, n_attr, c, h, w) -> (bs, c, h, w)
        pixel_values = batch.pixel_values[:, 0, :, :].squeeze()
        
        latent_features = []
        for attr in range(self.num_attributes):
            qformer_input_ids = batch.qformer_input_ids[:, attr, :].squeeze()
            qformer_attention_mask = batch.qformer_attention_mask[:, attr, :].squeeze()
                
            output = self.model(pixel_values=pixel_values,
                        qformer_input_ids=qformer_input_ids, 
                        qformer_attention_mask=qformer_attention_mask)
            
            q_former_output = output.qformer_outputs
            latent_features.extend(q_former_output.pooler_output.unsqueeze(0))
        return reduce(torch.stack(latent_features), 'n b c -> b c', 'mean')


    def training_step(self, batch, batch_idx):
        labels = batch['label'][:, 0].squeeze()
        if self.hparams.model_type == 'vit':
            out= self.model(pixel_values=batch['pixel_values'][:, 0, :, :].squeeze())
        else:
            out = self.get_latent_feature_for_vitqformer(batch)
        
        hard_pairs = self.mining_func(out, labels)
        tr_margin_loss = self.margin_loss(out, labels, hard_pairs)
        tr_pa_loss = self.pa_loss(out, labels, hard_pairs)
        
        self.log('train_loss', tr_margin_loss + tr_pa_loss)
        return tr_margin_loss + tr_pa_loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label'][:, 0].squeeze()
        if self.hparams.model_type == 'vit':
            out= self.model(pixel_values=batch['pixel_values'][:, 0, :, :].squeeze())
        else:
            out = self.get_latent_feature_for_vitqformer(batch)
        
        hard_pairs = self.mining_func(out, labels)
        va_margin_loss = self.margin_loss(out, labels, hard_pairs)
        va_pa_loss = self.pa_loss(out, labels, hard_pairs)

        return va_margin_loss + va_pa_loss

    def validation_epoch_end(self, outputs):
        avg_losses = []
        for loss_avg in outputs:
            avg_losses.append(loss_avg)
        self.log('val_loss', torch.stack(avg_losses).mean(), prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.hparams.lr, 
                                      betas=(0.9, 0.999), 
                                      eps=1e-08, 
                                      weight_decay=self.hparams.weight_decay)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def train_dataloader(self):
        self.train_set = CUB_200_2011(processor=self.processor, 
                              vision_dir=self.hparams.vision_dir, 
                              data_path=pjoin(self.hparams.annot_dir, "data.json"),
                              attributes_path=self.hparams.attributes_path)
        self.num_attributes = len(self.train_set.attributes)
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

