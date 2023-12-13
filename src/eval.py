import argparse

import torch
import faiss                            
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as pjoin

from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from transformers import InstructBlipProcessor
from einops import rearrange, reduce, repeat

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from data_utils import *
from wrapper import LightningWrapper
from dataset import CUB_200_2011_Test
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

def get_all_features(
        args,
        dataset,
        model,
        device
    ):
    all_features, all_labels = [], []
    for batch in tqdm(DataLoader(dataset, batch_size=16, num_workers=8)):
        features = get_latent_feature(args, model, batch, 
                                      num_attributes=len(dataset.attributes), 
                                      device=device)
        all_features += [features]
        all_labels += [batch['label'][:, 0]]
        return torch.cat(all_features), torch.cat(all_labels)
    
def test(args, trainset, testset, model, accuracy_calculator, device):
    train_embeddings, train_labels = get_all_features(args=args, dataset=trainset, model=model, device=device)
    test_embeddings, test_labels = get_all_features(args=args, dataset=testset, model=model, device=device)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    print("Test set accuracy = {}".format(accuracies))

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
        default="Salesforce/instructblip-vicuna-7b",
        type=str,
    )
    parser.add_argument(
        "--backbone",
        default="Salesforce/instructblip-flan-t5-xxl",
        type=str,
    )
    parser = LightningWrapper.add_model_specific_args(parser)
    args = parser.parse_args()
    
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    
    wrapper = LightningWrapper.load_from_checkpoint(checkpoint_path=args.checkpoint_name_or_path, 
                                                  hparams=args)
    model = wrapper.model
    processor = wrapper.processor
    
    model = model.to(args.device)
    model.eval()
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False  

    faiss_index = faiss.IndexFlatIP(768)  
    im_indices = []
    im_labels = []
    trainset = CUB_200_2011_Test(processor=processor, 
                    vision_dir=args.vision_dir, 
                    data_path=pjoin(args.annot_dir, "data.json"),
                    attributes_path=args.attributes_path)
    
    testset = CUB_200_2011_Test(processor=processor, 
                    vision_dir=args.vision_dir, 
                    data_path=pjoin(args.annot_dir, "test.json"),
                    attributes_path=args.attributes_path)
    
    for batch in tqdm(DataLoader(trainset, batch_size=4, num_workers=8)):
        features = get_latent_feature(args, model, batch, 
                                      num_attributes=len(trainset.attributes), 
                                      device=args.device).squeeze().cpu().numpy()
        for _feat, _label, _img_path in zip(features, batch['label'].cpu().numpy(), batch['image_path']):
            vector = _feat.reshape(1, -1)
            faiss.normalize_L2(vector)
            faiss_index.add(vector)
            im_labels += [_label[0]]
            im_indices += [_img_path]
        del features
    
    data_rows = []
    for batch in tqdm(DataLoader(testset, batch_size=4, num_workers=8)):
        test_embed = get_latent_feature(args, model, batch, 
                                        num_attributes=len(trainset.attributes), 
                                        device=args.device).squeeze().cpu().numpy()
        for _feat, _label, _img_path in zip(test_embed, batch['label'].cpu().numpy(), batch['image_path']):
            query_vector = _feat.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            _, I = faiss_index.search(query_vector, args.k)
            # print(f"Retrieved Image: {im_indices[I[0][0]]} (Label: {im_labels[I[0][0]]})")
            data_rows += [{
                'image_path' : _img_path,
                'label': _label[0],
                'search_paths': im_indices[I[0][0]],
                'search_labels': im_labels[I[0][0]]
            }]

    
    
    mkdir_p(args.save_dir)
    result = pd.DataFrame(data_rows)
    result.to_csv(pjoin(args.save_dir, f"{args.checkpoint_name_or_path.split('/')[-1].split('.')[0]}_result.csv"), index=False)
    
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", ), k=1)
    test(args=args, trainset=trainset, testset=testset, model=model, accuracy_calculator=accuracy_calculator, device=args.device)