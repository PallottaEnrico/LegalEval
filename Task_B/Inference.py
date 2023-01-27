from typing import Union, List, Optional
import torch
import numpy as np
import copy
from transformers.pipelines import AggregationStrategy, TokenClassificationPipeline

class SlidingWindowNERPipeline(TokenClassificationPipeline):
    """Modified version of TokenClassificationPipeline that uses a sliding
    window approach to fit long texts into the limited position embeddings of a
    transformer.
    """

    def __init__(self, aggregation_strategy, window_length: Optional[int] = None,
                 stride: Optional[int] = None, *args, **kwargs):
        super(SlidingWindowNERPipeline, self).__init__(
            *args, **kwargs)
        self.window_length = window_length or self.tokenizer.model_max_length
        if stride is None:
            self.stride = self.window_length // 2
        elif stride == 0:
            self.stride = self.window_length
        elif 0 < stride <= self.window_length:
            self.stride = stride
        else:
            raise ValueError("`stride` must be a positive integer no greater "
                             "than `window_length`")
        if aggregation_strategy == 'simple':
            self.aggregation_strategy = AggregationStrategy.SIMPLE
        elif aggregation_strategy == 'first':
            self.aggregation_strategy = AggregationStrategy.FIRST
        elif aggregation_strategy == 'average':
            self.aggregation_strategy = AggregationStrategy.AVERAGE
        elif aggregation_strategy == 'max':
            self.aggregation_strategy = AggregationStrategy.MAX
        else:
            self.aggregation_strategy = AggregationStrategy.NONE
        self.ignore_labels=["O"]

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `aggregation_strategy` is not :obj:`"none"`.
            - **index** (:obj:`int`, only present when ``aggregation_strategy="none"``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []
        num_labels = self.model.num_labels

        for i, sentence in enumerate(_inputs):

            # Manage correct placement of the tensors
            with self.device_placement():
                tokens = self.tokenizer(
                    sentence,
                    padding=True,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    return_special_tokens_mask=True,
                    add_special_tokens=True,
                    return_offsets_mapping=self.tokenizer.is_fast
                )
                if self.tokenizer.is_fast:
                    offset_mapping = \
                        tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = \
                    tokens.pop("special_tokens_mask").cpu().numpy()[0]

                if self.framework == "tf":
                    raise ValueError("SlidingWindowNERPipeline does not "
                                     "support TensorFlow models.")
                # Forward inference pass
                with torch.no_grad():
                    #tokens = self.ensure_tensor_on_device(**tokens)
                    tokens.to(self.device)
                    # Get logits (i.e. tag scores)
                    entities = np.zeros(tokens['input_ids'].shape[1:] +
                                        (num_labels,))
                    writes = np.zeros(entities.shape)
                    for start in range(
                            0, tokens['input_ids'].shape[1] - 1,
                            self.stride):
                        end = start + self.window_length - 2

                        window_input_ids = torch.cat([
                            torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device),
                            tokens['input_ids'][:, start:end],
                            torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
                        ], dim=1)
                        window_logits = self.model(
                            input_ids=window_input_ids)[0][0].cpu().numpy()
                        entities[start:end] += window_logits[1:-1]
                        writes[start:end] += 1
                    # Old way for getting logits under PyTorch
                    # entities = self.model(**tokens)[0][0].cpu().numpy()
                    input_ids = tokens["input_ids"].cpu().numpy()[0]
                    entities = entities / writes

                    scores = np.exp(entities) / np.exp(entities).sum(
                        -1, keepdims=True)
                    pre_entities = self.gather_pre_entities(
                        sentence, input_ids, scores, offset_mapping,
                        special_tokens_mask, aggregation_strategy=self.aggregation_strategy)
                    grouped_entities = self.aggregate(
                        pre_entities, self.aggregation_strategy)
                    if self.aggregation_strategy != AggregationStrategy.NONE:
                    # Filter anything that is in self.ignore_labels
                        entities = [
                            entity
                            for entity in grouped_entities
                            if entity.get("entity", None) not in self.ignore_labels
                            and entity.get("entity_group", None) not in
                            self.ignore_labels
                        ]
                        answers.append(entities)
                    else:
                        answers.append(grouped_entities)

        if len(answers) == 1:
            return answers[0]
        return answers
      
class CrfSlidingWindowNERPipeline(SlidingWindowNERPipeline):
    """Modified version of SlidingWindowNERPipeline made for models that
    use a CRF.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `aggregation_strategy` is not :obj:`"none"`.
            - **index** (:obj:`int`, only present when ``aggregation_strategy="none"``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []
        num_labels = self.model.num_labels

        for i, sentence in enumerate(_inputs):

            # Manage correct placement of the tensors
            with self.device_placement():
                tokens = self.tokenizer(
                    sentence,
                    #padding=True,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    return_special_tokens_mask=True,
                    add_special_tokens=False,
                    return_offsets_mapping=self.tokenizer.is_fast
                )
                if self.tokenizer.is_fast:
                    offset_mapping = \
                        tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = \
                    tokens.pop("special_tokens_mask").cpu().numpy()[0]

                if self.framework == "tf":
                    raise ValueError("SlidingWindowNERPipeline does not "
                                     "support TensorFlow models.")
                # Forward inference pass
                with torch.no_grad():
                    #tokens = self.ensure_tensor_on_device(**tokens)
                    tokens.to(self.device)
                    # Get logits (i.e. tag scores)
                    entities = np.zeros(tokens['input_ids'].shape[1:])
                    writes = np.zeros(entities.shape)
                    # ROBERTA
                    for start in range(
                            0, tokens['input_ids'].shape[1] - 1,
                            self.stride):
                        end = start + self.window_length - 2
                        
                        window_input_ids = torch.cat([
                            torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device),
                            tokens['input_ids'][:, start:end],
                            torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
                        ], dim=1)
                        window_logits = self.model(
                            input_ids=window_input_ids)[0][0].cpu().numpy()
                        
                        entities[start:end] = window_logits[1:-1]
                    
                    # XLNET
                    # window_input_ids = torch.cat([
                    #         torch.tensor([[CLS_ID]]).to(self.device),
                    #         tokens['input_ids'][:,],
                    #         torch.tensor([[SEP_ID]]).to(self.device)
                    #     ], dim=1)
                    # window_logits = self.model(input_ids=window_input_ids)[0][0].cpu().numpy()
                    # entities = window_logits[1:-1]
                    
                    
                    # Old way for getting logits under PyTorch
                    # entities = self.model(**tokens)[0][0].cpu().numpy()
                    input_ids = tokens["input_ids"].cpu().numpy()[0]
                    scores = entities
                    
                    
                    pre_entities = self.gather_pre_entities(
                        sentence, input_ids, scores, offset_mapping,
                        special_tokens_mask, aggregation_strategy=self.aggregation_strategy)
                    
                    
                    entities = []
                    for pre_entity in pre_entities:
                        entity_idx = pre_entity["scores"]
                        #score = pre_entity["scores"][entity_idx]
                        entity = {
                            "entity": self.model.config.id2label[entity_idx],
                            "score": 0,
                            "index": pre_entity["index"],
                            "word": pre_entity["word"],
                            "start": pre_entity["start"],
                            "end": pre_entity["end"],
                        }
                        entities.append(entity)
                    if self.aggregation_strategy != AggregationStrategy.NONE:
                        grouped_entities = self.group_entities(entities)
                    
                    if self.aggregation_strategy != AggregationStrategy.NONE:
                    # Filter anything that is in self.ignore_labels
                        entities = [
                            entity
                            for entity in grouped_entities
                            if entity.get("entity", None) not in self.ignore_labels
                            and entity.get("entity_group", None) not in
                            self.ignore_labels
                        ]
                        answers.append(entities)
                    else:
                        answers.append(grouped_entities)

        if len(answers) == 1:
            return answers[0]
        return answers
    

def remap_predictions(df, df_clean, predictions):
    
    def escape(pattern):
    """
        Escape special characters in a string.
    """
        if isinstance(pattern, str):
            return pattern.translate(special_chars_map)
        else:
            pattern = str(pattern, 'latin1')
            return pattern.translate(special_chars_map).encode('latin1')
    
    predictions_cp = copy.copy(predictions)
    
    special_chars_map = {i: '\\' + chr(i) for i in b'()[]{}?*+-|^$\\.&~#\t\n\r\v\f'}

    for row, row_clean, i in zip(df.iloc, df_clean.iloc, range(len(predictions_cp))):
        
        pred = predictions_cp[i]
        context = row['context']
        context_clean = row_clean['context']
        
        text = [row_clean['context'][p['start']:p['end']] for p in pred]
        
        start = []
        end = []
        off = 0
        
        s_temp = [p['start'] for p in pred]
        try:
            for t,tmp in zip(text,s_temp):
                
                while True: # avoid infinite loops
                    s1 = escape(t)
                    s2 = context[off:]
                    match = re.search(r'\s*'.join(s1.split()), s2)
                    s, e = match.start(), match.end()
                    
                    string_clean = context_clean[:tmp]
                    string = context[:s+off]
                    if re.sub('\s+', '', string) == re.sub('\s+', '', string_clean):
                       break
                    off += e
                
                start.append(s+off)
                end.append(e+off)
                off = end[-1]
                    
            for j in range(len(pred)):
                predictions_cp[i][j]['start'] = start[j]
                predictions_cp[i][j]['end'] = end[j]
        except Exception as e:
            print(e)
            print("Corrispondence not found")
    return predictions_cp
