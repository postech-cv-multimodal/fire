import random
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import argparse

from einops import reduce
from tqdm import tqdm
from PIL import Image
from os.path import join as pjoin

from transformers import InstructBlipProcessor, InstructBlipQFormerModel
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from data_utils import mkdir_p, del_folder, read_json, write_json, read_text
from instruct_blip import InstructBlipForImageRetrieval
from dataset import CUB_200_2011    


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

num_attributes=3

def train_qformer(instruct_blip, loss_func, mining_func, device, train_loader, 
                  optimizer, epoch, writer, total_loss=0, total_step=0):
    instruct_blip.train()
        
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        labels = batch.label[:,0]
        pixel_values = batch.pixel_values[:, 0, :, :].squeeze().to(device)
        image_encoder_output = instruct_blip.vision_model(pixel_values=pixel_values, 
                                            output_attentions=True, 
                                            output_hidden_states=True, 
                                            return_dict=True)
        
        instructed_embeddings= []
        for attr in range(num_attributes):
            qformer_input_ids = batch.qformer_input_ids[:, attr, :].squeeze().to(device)
            qformer_attention_mask = batch.qformer_attention_mask[:, attr, :].squeeze().to(device)
            
            q_former_output = instruct_blip(image_embeds=image_encoder_output[0].to(device),
                                            qformer_input_ids=qformer_input_ids, 
                                            qformer_attention_mask=qformer_attention_mask, 
                                            only_qformer=True).qformer_outputs  
            instructed_embeddings.append(q_former_output.pooler_output)

        instructed_embeddings = reduce(torch.stack(instructed_embeddings), 'n b c -> b c', 'mean')


        loss = loss_func(instructed_embeddings, labels)
        loss.backward()
        optimizer.step()
        if loss.item() == 0:
            print("Loss is zero")
            continue
        
        total_loss += loss.item()
        total_step += 1
        
        if total_step % 20 == 0:
            print("Epoch {} Step {}: Loss = {}".format(
                epoch, total_step, total_loss/total_step))
    
            
            writer.add_scalar("train/batch_loss", loss, total_step)
            writer.add_scalar("train/avg_loss", total_loss/total_step, total_step)

    return instruct_blip, optimizer, total_loss, total_step

def main(args):
    
    torch.cuda.set_device(torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'))
    print ('Cuda device %s | %s | %s/%sGB' % (torch.cuda.current_device(), 
                                            torch.cuda.get_device_name(args.gpu),
                                            round(torch.cuda.memory_allocated(args.gpu)/1024**3,1),
                                            round(torch.cuda.memory_reserved(args.gpu)/1024**3,1)))

    device = "cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = InstructBlipProcessor.from_pretrained(args.blip_processor_file)
    instruct_blip = InstructBlipForImageRetrieval.from_pretrained(args.blip_model_file)

    cnt=0
    for network_name, parameter in instruct_blip.named_parameters():
        if "vision_model" in network_name:
            cnt += 1
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
    print(f"# of the vision encoder's networks : {cnt}")

    cub_train_data = CUB_200_2011(processor=processor, 
                                vision_dir=args.image_path, 
                                data_path=args.train_data_path,
                                attributes_path=args.attribute_path)

    num_epochs = args.epoch
    batch_size = args.batch_size # Model& Data Load 시, 약 23,032MiB 소모 (3090 기준 배치사이즈 32가 적당해보임)
    #margin = args.margin
    lr = args.lr

    optimizer = optim.AdamW(params=instruct_blip.parameters(), lr=lr, weight_decay=1e-4)
    #distance = distances.CosineSimilarity()
    #reducer = reducers.ThresholdReducer(low=0)

    loss_func = losses.MarginLoss(margin=1.2, beta=0.2)
    #mining_func = miners.DistanceWeightedMiner(margin=1.2, distance=distance, type_of_triplets="all")
    mining_func = miners.TripletMarginMiner(margin=0.9, type_of_triplets="all")

    #accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    train_loader = torch.utils.data.DataLoader(
        cub_train_data, batch_size=batch_size, shuffle=True
    )
    
    instruct_blip.to(device) # Model Load 시, GPU VRAM 약 4,916MiB 소모
    total_loss=0
    total_step=0
    for epoch in range(1, num_epochs + 1):
        instruct_blip, optimizer, total_loss, total_step = train_qformer(
            instruct_blip, loss_func, mining_func, device, train_loader, optimizer, epoch, writer, total_loss, total_step)

        checkpoint_path = args.checkpoint_name_or_path + f"instruct_blip_only_qformer_{epoch}.pt"
        torch.save(instruct_blip.qformer, checkpoint_path) # Qformer만 저장

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=3)
    parser.add_argument("--seed", "-s", type=int, default=42)
    
    parser.add_argument("--blip_processor_file", type=str)
    parser.add_argument("--blip_model_file", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--attribute_path", type=str)
    parser.add_argument("--checkpoint_name_or_path", type=str)


    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    main(args)
