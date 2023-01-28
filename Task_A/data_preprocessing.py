import pandas as pd
import re
from tqdm import tqdm
from transformers import (AutoTokenizer,
                          BatchEncoding)
from datasets import Dataset


def index_in_words(text: str,
                   char_index: int) -> int:
    """
      Converts the given char_index its equivalent word_index in the given text.

      Params:
        - text: text useful to convert the char_index
        - char_index: char position in the given text

      Returns:
        - word_index: equivalent word position the given text
    """

    # Find the previous and next spaces around the index
    prev_space = text[:char_index].rfind(' ')
    next_space = text[char_index:].find(' ')

    if char_index == 0:
        return 0

    # If both are found
    if prev_space != -1 and next_space != -1:

        # Count the number of words before the current word
        word_index = len(re.findall(r'\b\S+\b', text[:prev_space]))
        return word_index

    # If only the previous space is found
    elif prev_space != -1:

        # Count the number of words before the current word
        word_index = len(re.findall(r'\b\S+\b', text[:prev_space]))
        return word_index

    # If only the next space is found
    elif next_space != -1:

        # Count the number of words before the next word
        word_index = len(re.findall(r'\b\S+\b', text[:char_index]))
        return word_index - 1

    # If no space is found
    else:

        # Count the number of words before the next word
        word_index = len(re.findall(r'\b\S+\b', text[:char_index]))
        return word_index


def get_context(text: str,
                sentence: str,
                span_start: int,
                span_end: int,
                max_len: int) -> str:
    """
      Given the sentence, it extracts the context of the sentence
      from the text, that fits the transformer

      Params:
        text: text useful to extract the context
        sentence: sentence
        span_start: start position of the sentence at char-level
        span_end: end position of the sentence at char-level

      Returns:
        context: context string, including the sentence
    """

    context = []

    # Positions at word-level
    span_start = index_in_words(text, span_start)
    span_end = index_in_words(text, span_end)

    # Divide the given string into words, deleting space characters
    text = re.findall(r'\b\S+\b', text)
    sentence = re.findall(r'\b\S+\b', sentence)
    len_sentence = len(sentence)

    # Empty context
    if len_sentence > max_len:
        return " ".join(context)

    # Get length of the window
    len_window = int((max_len - (len_sentence * 2)) / 2)

    if len_window <= 0:
        return " ".join(context)

    # First sentence
    if span_start <= 0:
        context += sentence

        idx = span_end + 1
        while idx <= span_end + len_window * 2:
            context.append(text[idx])
            idx += 1

        return " ".join(context)

    # Last sentence
    if span_end >= len(text):
        idx = span_start - len_window * 2
        while idx < span_start:
            context.append(text[idx])
            idx += 1

        context += sentence

        return " ".join(context)

    # Left context smaller than window
    if span_start < len_window:
        idx = 0
        while idx < span_start:
            context.append(text[idx])
            idx += 1

        context += sentence

        idx = span_end + 1
        while idx <= span_end + len_window + (len_window - span_start):
            context.append(text[idx])
            idx += 1

        return " ".join(context)

    # Right context smaller than window
    if len_window > (len(text) - span_end):
        idx = span_start - len_window - (len_window - (len(text) - span_end))
        while idx < span_start:
            context.append(text[idx])
            idx += 1

        context += sentence

        idx = span_end + 1
        while idx <= len(text) - 1:
            context.append(text[idx])
            idx += 1

        return " ".join(context)

    # Append left context
    idx = span_start - len_window
    while idx < span_start:
        context.append(text[idx])
        idx += 1

    context += sentence

    # Append right contenxt
    idx = span_end + 1
    while idx < span_end + len_window:
        context.append(text[idx])
        idx += 1

    return " ".join(context)


def rearrange_df(data: list,
                 max_len_context: int) -> pd.DataFrame:
    """
      Given the dataset, extract useful columns from the dataset and converts it into a Pandas DataFrame

      Params:
        data: dataset

      Returns:
        new_dataset: DataFrame with the useful columns extracted from the given dataset
    """

    cols = ['doc_id', 'text', 'context', 'sentence', 'label']

    dataset = []

    for doc in tqdm(data):
        print('PIPPO')
        for doc_ann in doc['annotations']:
            for sent in doc_ann['result']:
                sentence = sent['value']['text']

                sentence_start = sent['value']['start']
                sentence_end = sent['value']['end']

                temp_list = [doc['id'], doc['data']['text'],
                             get_context(doc['data']['text'], sentence, sentence_start, sentence_end, max_len_context),
                             sentence, sent['value']['labels'][0]]

                dataset.append(temp_list)

    new_dataset = pd.DataFrame(dataset, columns=cols)

    # Drop duplicates
    new_dataset.drop_duplicates(['text', 'sentence'], inplace = True)

    return new_dataset
