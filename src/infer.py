import argparse
import torch 

import numpy as np
import pandas as pd
import torch.nn.functional as F

from glob import iglob
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
from transformers import InstructBlipProcessor

from data_utils import *
from eval_utils import l2_norm, calc_recall_at_k
from wrapper import LightningWrapper
from dataset import CUB_200_2011_Test, CUB_200_2011_Test_for_Vit
from vit import VitForImageRetrieval
from instruct_blip import InstructBlipForImageRetrieval

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

@torch.no_grad()
def get_latent_feature_for_vitqformer(model, batch, num_attributes, device):
    # pixel_values (bs, n_attr, c, h, w) -> (bs, c, h, w)
    pixel_values = batch.pixel_values[:, 0, :, :].squeeze().to(device)
    
    latent_features = []
    for attr in range(num_attributes):
        qformer_input_ids = batch.qformer_input_ids[:, attr, :].squeeze().to(device)
        qformer_attention_mask = batch.qformer_attention_mask[:, attr, :].squeeze().to(device)
            
        output = model(pixel_values=pixel_values,
                    qformer_input_ids=qformer_input_ids, 
                    qformer_attention_mask=qformer_attention_mask)
        
        q_former_output = output.qformer_outputs
        latent_features.extend(q_former_output.pooler_output.unsqueeze(0))
    return reduce(torch.stack(latent_features), 'n b c -> b c', 'mean')

@torch.no_grad()
def get_latent_feature_for_vit(model, batch, device):
    # pixel_values (bs, n_attr, c, h, w) -> (bs, c, h, w)
    pixel_values = batch.pixel_values[:, 0, :, :].squeeze().to(device)
    output = model(pixel_values=pixel_values)
    return output


@torch.no_grad()
def get_latent_feature(args, model, batch, num_attributes, device):
    if args.model_type == 'vit':
        return get_latent_feature_for_vit(model, batch, device)
    return get_latent_feature_for_vitqformer(model, batch, num_attributes, device)


@torch.no_grad()
def get_all_features(
        args,
        dataset,
        model,
        device
    ):
    all_features, all_labels = [], []
    for batch in tqdm(DataLoader(dataset, batch_size=64, num_workers=8)):
        features = get_latent_feature(args, model, batch, 
                                      num_attributes=args.num_attributes, 
                                      device=device)
        all_features += [features]
        if len(batch['label'].shape) > 1:
            all_labels += [batch['label'][:, 0]]
        else:
            all_labels += [batch['label']]
    return torch.cat(all_features), torch.cat(all_labels)
    

def evaluate_cos(args, model, testset, device):
    test_emb, test_labels = get_all_features(args=args, dataset=testset, model=model, device=device)
    test_emb = l2_norm(test_emb)
    test_labels = test_labels.to(device)
    print("Computing accuracy")

    K = 32
    cos_sim = F.linear(test_emb, test_emb)
    topk_indices = cos_sim.topk(1 + K)[1][:, 1:]
    topk_labels = [test_labels[i] for i in topk_indices[:, ]]
    topk_labels = torch.stack(topk_labels)
    
    recall = []
    for k in [1, 2, 4, 8]:
        r_at_k = calc_recall_at_k(test_labels, topk_labels, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        default=1,
        type=int,
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
        "--save_dir",
        default="output/",
        type=str,
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
        "--checkpoint_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--backbone",
        default="Salesforce/instructblip-flan-t5-xxl",
        type=str,
    )
    parser.add_argument(
        "--num_classes",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--embedding_size",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--attr_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--num_attributes",
        default=1,
        type=int,
    )
    parser = LightningWrapper.add_model_specific_args(parser)
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if args.checkpoint_name_or_path is not None:
        wrapper = LightningWrapper.load_from_checkpoint(checkpoint_path=args.checkpoint_name_or_path, 
                                                    hparams=args)
        model = wrapper.model
        processor = wrapper.processor
    else:
        
        if args.model_type == 'vit+qformer':
            processor = InstructBlipProcessor.from_pretrained(args.backbone)
            model = InstructBlipForImageRetrieval.from_pretrained(args.backbone)
        else:
            processor = InstructBlipProcessor.from_pretrained(args.backbone)
            model = VitForImageRetrieval.from_pretrained(args.backbone)
        
    model = model.to(args.device)
    model.eval()
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False  
    
    if args.attr_dir is not None:
        for path in iglob(pjoin(args.attr_dir, "*.txt")):
            testset = CUB_200_2011_Test(processor=processor, 
                        vision_dir=args.vision_dir, 
                        data_path=pjoin(args.annot_dir, "test.json"),
                        attributes_path=path)
            Recalls = evaluate_cos(args, model, testset, args.device)
            print(path, Recalls)
    else:
        testset = CUB_200_2011_Test(processor=processor, 
                        vision_dir=args.vision_dir, 
                        data_path=pjoin(args.annot_dir, "test.json"),
                        attributes_path=args.attributes_path)
        Recalls = evaluate_cos(args, model, testset, args.device)
        print(testset.attributes, Recalls)
    