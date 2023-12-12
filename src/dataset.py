import os
import torch
import random

import pandas as pd

from PIL import Image
from os.path import join as pjoin
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from transformers import AutoProcessor

from data_utils import read_json, read_text

class CUB_200_2011(Dataset):
    def __init__(self, 
                 processor: AutoProcessor,
                 data : dict=None,
                 data_path: str=None, 
                 vision_dir: str=None, 
                 attributes_path: str=None,
                 attributes: List[str]=None
                 ):
        self.vision_dir = vision_dir
        
        self.data = data if data is not None else read_json(data_path)
        self.attributes = read_text(attributes_path) if not attributes else attributes
            
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):        
        num_trial = 10
        for _ in range(num_trial):
            try:
                # Process image
                image_path = pjoin(self.vision_dir, self.data[index]["image_path"])
                image = Image.open(image_path).convert("RGB")
                
                encoding = self.processor(images=[image for _ in self.attributes],
                                        text=self.attributes, 
                                        padding=True, 
                                        return_tensors="pt")
                
                # pixel_values: (bs, n_attr, c, h, w)
                # qformer_input_ids: (bs, n_attr, l)
                # label: (bs, n_attr)
                cls_id = int(self.data[index]["image_path"].split(".")[0])
                encoding['label'] = torch.tensor(cls_id, dtype=torch.long).repeat(len(self.attributes))
                return encoding

            except Exception as e:
                print(f"Failed to load examples: {self.data[index]}. ({e})")
                index = random.choice(range(len(self.data)))
        
        return None

class CUB_200_2011_Test(Dataset):
    def __init__(self, 
                 processor: AutoProcessor,
                 data : dict=None,
                 data_path: str=None, 
                 vision_dir: str=None, 
                 attributes_path: str=None,
                 attributes: List[str]=None
                 ):
        self.vision_dir = vision_dir
        
        self.data = data if data is not None else read_json(data_path)
        self.attributes = read_text(attributes_path) if not attributes else attributes
            
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):        
        num_trial = 10
        for _ in range(num_trial):
            try:
                # Process image
                image_path = pjoin(self.vision_dir, self.data[index]["image_path"])
                image = Image.open(image_path).convert("RGB")
                
                encoding = self.processor(images=[image for _ in self.attributes],
                                        text=self.attributes, 
                                        padding=True, 
                                        return_tensors="pt")
                
                # pixel_values: (bs, n_attr, c, h, w)
                # qformer_input_ids: (bs, n_attr, l)
                # label: (bs, n_attr)
                cls_id = int(self.data[index]["image_path"].split(".")[0])
                encoding['label'] = torch.tensor(cls_id, dtype=torch.long).repeat(len(self.attributes))
                encoding['image_path'] = image_path
                return encoding

            except Exception as e:
                print(f"Failed to load examples: {self.data[index]}. ({e})")
                index = random.choice(range(len(self.data)))
        
        return None
    