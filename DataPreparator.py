import re 
from transformers import AutoTokenizer, TokenClassificationPipeline
from datasets import Dataset, Features, Value, ClassLabel, Sequence
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Optional
from transformers.pipelines import AggregationStrategy
import torch
import numpy as np

def clean_data(row):
    
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
        
        # clean everything before the entity start
        tmp_stripped = re.sub('\s+', ' ', tmp)
        
        len_after = len(tmp_stripped)
        
        to_remove_first = len_before - len_after
        # define the new start index
        new_start.append(s-to_remove_first)
            
        # extract the context between the indices
        tmp = context[s:e]
        # compute the difference between the length of the context and of the cleaned context
        len_before = len(tmp)

        # clean everything between start and end.
        tmp_stripped = re.sub('\s+', ' ', tmp)

        len_after = len(tmp_stripped)
        to_remove_after = len_before - len_after
        # # define the new end index
        new_end.append(e - (to_remove_first + to_remove_after))
        
        new_text.append(tmp_stripped)
        
    # define the full cleaned context   
    new_context = re.sub('\s+', ' ', context)

    # assign the new value to the row
    row['start'] = new_start
    row['end'] = new_end
    row['context'] = new_context
    row['text'] = new_text
    
    return row

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
  if init_special_char == 'd':
    # bert models do not use special char for first sub-token, they use ## for all the other sub-tokens
    init_special_char = '##'

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

    def as_hf_dataset(self, window_length, sl_window, CRF = False) -> Dataset:

        def generate_input_ids_labels(examples):
            # CRF does not allow -100 as token id, so we'll not add it.
            if CRF:
                input_ids = [tokenizer.convert_tokens_to_ids(e)[:window_length-2] for e in examples["tokens"]]
                labels = [e[:window_length-2] for e in examples['ner_tags']]   
            else:
                input_ids = [[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(e)[:window_length-2]+ [self.tokenizer.sep_token_id] for e in examples["tokens"]]
                labels = [[-100] + e[:window_length-2] + [-100] for e in examples['ner_tags']] 

            tokenized_inputs = {
                'input_ids' : input_ids, 
                'labels' : labels
            }
                
            return tokenized_inputs

        ner_tags, tokens = [], []
        
        for i, pt in enumerate(self.processed_texts):
            pt_tokens,pt_tags = list(zip(*pt))
            if sl_window and len(pt_tokens) > window_length:
                for start in range(0, len(pt_tokens) - 1, window_length // 2): # stride half window length
                    end = start + window_length - 2

                    window_tokens = pt_tokens[start:end]
                    window_ner_tags = pt_tags[start:end]

                    ner_tags.append(window_ner_tags)
                    tokens.append(window_tokens)
            else:
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

