import torch
import torch.nn as nn
from transformers import (BertModel, )


class BertSequenceClassifier(nn.Module):
    def __init__(self, num_classes: int, bert_name: str, id2label: dict, label2id: dict, smoothing: int = 0.2):
        super(BertSequenceClassifier, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_name,
                                              num_labels=self.num_classes,
                                              id2label=id2label,
                                              label2id=label2id,
                                              use_cache=False)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # LSTM layer
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 256, batch_first=True, num_layers=1, bidirectional=True)

        # Another dropout layer
        self.dropout2 = nn.Dropout(0.2)

        # Another LSTM layer
        self.lstm2 = nn.LSTM(512, 256, num_layers=1, batch_first=True, bidirectional=True)

        # Linear output layer
        self.output = nn.Linear(512, self.num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get the hidden state of the BERT model
        bert_hidden = self.bert(input_ids, attention_mask,
                                return_dict=False)

        # Apply dropout
        bert_hidden = self.dropout(bert_hidden[0])

        # Pass the hidden state through the LSTM layer
        lstm_out, _ = self.lstm(bert_hidden)
        # print(f'{lstm_out.shape = }')

        lstm_sentence = torch.mean(lstm_out, dim=1)

        lstm_sentence = torch.unsqueeze(lstm_sentence, dim=0)
        # print(f'{lstm_sentence.shape = }')

        # Apply dropout
        lstm_sentence_dropout = self.dropout2(lstm_sentence)

        # Pass the output through another LSTM layer
        lstm_out2, _ = self.lstm2(lstm_sentence_dropout)
        # print(f'{lstm_out2.shape = }')

        # Predict the class probabilities using the linear output layer
        logits = self.output(lstm_out2)
        logits = torch.squeeze(logits, dim=0)
        # print(f'{logits.shape = }')

        # If labels are provided, compute the cross-entropy loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.smoothing)
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits
        else:
            # Otherwise, return the class probabilities
            return (logits,)
