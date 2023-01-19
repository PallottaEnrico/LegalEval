import re 
from transformers import AutoTokenizer, TokenClassificationPipeline
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Optional
from transformers.pipelines import AggregationStrategy
import torch
import numpy as np

def adapt_indexes_without_spaces(row):
    """
    This function is very similar to the clean data one, it basically removes any possible space and adapt the start/end indexes consequently.
    We need this new start/end since the labeling step will work character wise, so we don't want to count spaces.
    """
    start = row['start']
    end = row['end']
    context = row['context']
    text = row['text']

    new_start, new_end, new_text = [], [], []

    # for each start-end index
    for s,e in zip(start,end):
        
        # extract the context until the start of the text
        tmp = context[:s]
        # compute the difference between the length of the context and of the cleaned context
        len_before = len(tmp)
        
        tmp_stripped = re.sub('\s', '', tmp)
        
        len_after = len(tmp_stripped)
        
        to_remove_first = len_before - len_after
        # define the new start index
        new_start.append(s-to_remove_first)
            
        # extract the context between the indices
        tmp = context[s:e]
        # compute the difference between the length of the context and of the cleaned context
        len_before = len(tmp)
        tmp_stripped = re.sub('\s', '', tmp)
        len_after = len(tmp_stripped)
        to_remove_after = len_before - len_after
        # # define the new end index
        new_end.append(e - (to_remove_first + to_remove_after))
        
        new_text.append(tmp_stripped)
    
    return new_start, new_end

def tokenize_and_label(row : dict, tokenizer : AutoTokenizer):
  """
    Tokenizes the input context and assignes a label to each token, solving the 
    misalignment between labeled words and sub-tokens.

    Params:
      row : DataFrame row to tokenize
      tokenizer : Tokenizer to use
    Returns:
      context tokenized and associated labels in B-I-O format.
  """
  # compute new start/end indexes without considering white spaces
  char_wise_start, char_wise_end = adapt_indexes_without_spaces(row)

  # standard tokenization applied
  tokens_context = tokenizer.tokenize(row['context'], truncation=True, max_length=10000)

  # our result vector with a label for each token
  labels = ['O'] * len(tokens_context)

  # keep track of labels alreay assigned to token, distinguish between "B-" and "I-" labels
  mask_label_used = [False] * len(row['label'])

  # "pointer" (in the whole context without spaces) to first character of the current token
  actual_char_index = 0

  # most transformers' tokenizers add a special character to the first sub-token of a word
  # dummy tokenization to retrieve it
  init_special_char = tokenizer.tokenize('dummy')[0][0]

  for _, token in enumerate(tokens_context):
    # remove init character if present
    clean_tok = token.replace(init_special_char, "")
    
    for lbl_index , (start, end , label) in enumerate(zip(char_wise_start, char_wise_end, row['label'])):
      # check if the pointer is inside an entity
      if actual_char_index in range(start,end):
        if mask_label_used[lbl_index] == False:
          # first time we assign the label to a token
          labels[_] = "B-" + label
          # mark it as already assigned, next time will be "I-"
          mask_label_used[lbl_index] = True
        else:
          # the label has been already assigned to a token
          labels[_] = "I-" + label
        # once we have found the label we can skip the other checks
        break     
    # update pointer
    actual_char_index += len(clean_tok)

  return list(zip(tokens_context, labels))

class NERDataMaker:
    def __init__(self, df : pd.DataFrame, tokenizer : AutoTokenizer):
        self.tokenizer = tokenizer
        self.unique_entities = []
        self.processed_texts = []

        temp_processed_texts = []

        for _, row in tqdm(df.iterrows()):
            # pass tokens with labels
            tokens_with_entities = tokenize_and_label(row, self.tokenizer)
            for _, ent in tokens_with_entities:
                if ent not in self.unique_entities:
                    self.unique_entities.append(ent)
            temp_processed_texts.append(tokens_with_entities)
        self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")
        self.unique_entities.remove("O")
        self.unique_entities.sort()
        self.unique_entities.insert(0,"O")

        for tokens_with_entities in temp_processed_texts:
            self.processed_texts.append([(t, self.unique_entities.index(ent)) for t, ent in tokens_with_entities])

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v:k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        def _process_tokens_for_one_text(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            for t, ent in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)

            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens
            }

        tokens_with_encoded_entities = self.processed_texts[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_text(idx, tokens_with_encoded_entities)
        else:
            return [_process_tokens_for_one_text(i+idx.start, tee) for i, tee in enumerate(tokens_with_encoded_entities)]

    def as_hf_dataset(self, window_length, sl_window) -> Dataset:

        def generate_input_ids_labels(examples):
            # remember we need to add the start and end token id (they will have label -100)
            tokenized_inputs = {
                'input_ids' : [[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(e)[:window_length-2]+ [self.tokenizer.eos_token_id] for e in examples["tokens"]], 
                'labels' : [[-100] + e[:window_length-2] + [-100] for e in examples['ner_tags']] 
            }
            
            return tokenized_inputs

        ner_tags, tokens = [], []
        
        if sl_window:
            for i, pt in enumerate(self.processed_texts):
                pt_tokens,pt_tags = list(zip(*pt))

                for start in range(0, len(pt_tokens) - 1, window_length // 2): # stride half window length
                  end = start + window_length - 2

                  window_tokens = pt_tokens[start:end]
                  window_ner_tags = pt_tags[start:end]

                  ner_tags.append(window_ner_tags)
                  tokens.append(window_tokens)
        else:
            for i, pt in enumerate(self.processed_texts):
                pt_tokens,pt_tags = list(zip(*pt))
                ner_tags.append(pt_tags)
                tokens.append(pt_tokens)
            
        data = {
            "ner_tags": ner_tags,
            "tokens": tokens
        }
        features = Features({
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=self.unique_entities))
        })
        ds = Dataset.from_dict(data, features)
        tokenized_ds = ds.map(generate_input_ids_labels, batched=True)
        return tokenized_ds
      
      
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
