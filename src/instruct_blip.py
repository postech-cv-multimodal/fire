import torch
from torch import nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union


from transformers.utils import ModelOutput
from transformers.models.instructblip import (
        InstructBlipPreTrainedModel,
        InstructBlipConfig,
        InstructBlipVisionConfig,
        InstructBlipQFormerModel, 
        InstructBlipVisionModel
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from transformers.activations import GELUActivation
    

@dataclass
# Copied from transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput with Blip2->InstructBlip
class InstructBlipForImageRetrievalModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InstructBlipForImageRetrieval`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
            
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
    """

    #loss: Optional[Tuple[torch.FloatTensor]] = None
    #logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )
     
    
class InstructBlipForImageRetrieval(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        self.vision_model = InstructBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        image_embeds : torch.FloatTensor, 
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForImageRetrievalModelOutput]:
        
        r"""
        
        Example : 
        
        To do...
        
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        #vision_outputs = self.vision_model(
        #    pixel_values=pixel_values,
        #    output_attentions=output_attentions,
        #    output_hidden_states=output_hidden_states,
        #    return_dict=return_dict,
        #)
        #image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        return InstructBlipForImageRetrievalModelOutput(
            vision_outputs=image_embeds,
            qformer_outputs=query_outputs,
        )