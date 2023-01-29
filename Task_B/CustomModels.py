from torchcrf import CRF
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import XLNetModel, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetForTokenClassificationOutput
from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput

   
class CustomRobertaForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config, custom_layers):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.num_LSTM = custom_layers['LSTM'] if 'LSTM' in custom_layers.keys() else 0
        self.CRF = 'CRF' in custom_layers.keys() and custom_layers['CRF'] == True

        if self.num_LSTM:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, num_layers = self.num_LSTM, bidirectional=True, dropout=0.5)
        
        if self.CRF:
            self.crf = CRF(config.num_labels, batch_first=True)
            self.crf.reset_parameters()
        
        self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        if self.num_LSTM:
            # Forward propagate through LSTM
            sequence_output, _ = self.lstm(sequence_output)
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None

        if self.CRF:
              if labels is not None:
                  loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask.byte(), reduction='token_mean')
                  logits = self.crf.decode(logits)
              else:
                  logits = self.crf.decode(logits)
              logits = torch.Tensor(logits)
        else:
              if labels is not None:
                  loss_fct = CrossEntropyLoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  
    
    
class CustomXLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config, custom_layers):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        
        self.num_LSTM = custom_layers['LSTM'] if 'LSTM' in custom_layers.keys() else 0
        self.CRF = 'CRF' in custom_layers.keys() and custom_layers['CRF'] == True

        if self.num_LSTM:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, num_layers = self.num_LSTM, bidirectional=True, dropout=0.5)
        
        if self.CRF:
            self.crf = CRF(config.num_labels, batch_first=True)
            self.crf.reset_parameters()
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
                
        if self.num_LSTM:
            # Forward propagate through LSTM
            sequence_output, _ = self.lstm(sequence_output) 

        logits = self.classifier(sequence_output)
        loss = None
        
        if self.CRF:
            if labels is not None:
                loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask.byte(), reduction='token_mean')
                logits = self.crf.decode(logits)
            else:
                logits = self.crf.decode(logits)
            logits = torch.Tensor(logits)
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 
