import torch
from torch import nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from einops import rearrange

from transformers.utils import ModelOutput
from transformers.models.instructblip import (
        InstructBlipPreTrainedModel,
        InstructBlipConfig,
        InstructBlipVisionModel
)

from transformers.activations import GELUActivation
    
    
class VitForImageRetrieval(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        self.vision_model = InstructBlipVisionModel(config.vision_config)
        for _, parameter in self.vision_model.named_parameters():
            parameter.requires_grad = False
            
        self.head = nn.Linear(config.vision_config.hidden_size, 768)
        self.post_init()

    def forward(
        self,
        pixel_values : torch.FloatTensor, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(
           pixel_values=pixel_values,
           output_attentions=output_attentions,
           output_hidden_states=output_hidden_states,
           return_dict=True,
        )
        image_embeds = vision_outputs.pooler_output
        #flatten_embeds = rearrange(image_embeds, "b n h -> b (n h)")
        return out = self.head(image_embeds)
