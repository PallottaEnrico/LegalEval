import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import random
import torch
import evaluate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def labels_frequency(df: pd.DataFrame):
    """
      Plots the frequency of the labels of the given dataset

      Params:
        df: dataset
    """

    labels = list(df['label'])
    colors = dict(zip(set(labels), plt.cm.rainbow(np.linspace(0, 1, len(set(labels))))))
    for c in set(labels):
        plt.hist([labels[i] for i in range(len(labels)) if labels[i] == c], bins=np.arange(14) - 0.5, rwidth=0.7,
                 color=colors[c])
    plt.rcParams["figure.figsize"] = (22, 5)
    plt.show()


def compute_metrics(eval_pred):
    """
      Params:
        eval_pred : predictions of the model on the validation set
      Returns:
        A dictionary containing accuracy and weighted F1 scores computed on eval_pred
    """

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy_score = evaluate.load("accuracy")
    accuracy = accuracy_score.compute(predictions=predictions, references=labels)['accuracy']
    f1_score = evaluate.load("f1")
    weighted_f1 = f1_score.compute(predictions=predictions, references=labels, average="weighted")['f1']

    return {
        'Accuracy': accuracy,
        'Weighted F1': weighted_f1,
    }


def plot_confusion_matrix(true: list,
                          predictions: list,
                          labels_name: list,
                          model_name: str = " "):
    """
      Plots three different confusion matrices:
        - Confusion matrix not normalized
        - Confusion matrix with the recall along the diagonal
        - Confusion matrix with the precision along the diagonal

      Params:
        true: true labels
        predictions: labels predicted
        labels_name: name of the labels
        model_name: name of the model
    """

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(37, 18))
    ax[0].set_title('Confusion matrix not normalized - ' + model_name)
    ax[1].set_title('Confusion matrix recall - ' + model_name)
    ax[2].set_title('Confusion matrix precision - ' + model_name)

    cm = confusion_matrix(true, predictions)
    cm_rec = confusion_matrix(true, predictions, normalize='true')
    cm_prec = confusion_matrix(true, predictions, normalize='pred')

    cmp = ConfusionMatrixDisplay(cm, display_labels=labels_name)
    cmp_rec = ConfusionMatrixDisplay(cm_rec, display_labels=labels_name)
    cmp_prec = ConfusionMatrixDisplay(cm_prec, display_labels=labels_name)

    cmp.plot(ax=ax[0], cmap=plt.cm.Blues, xticks_rotation=90, colorbar=False)
    cmp_rec.plot(ax=ax[1], cmap=plt.cm.Blues, values_format='.2f', xticks_rotation=90, colorbar=False)
    cmp_prec.plot(ax=ax[2], cmap=plt.cm.Blues, values_format='.2f', xticks_rotation=90, colorbar=False)


def print_document(test_df: pd.DataFrame,
                   y_test: list,
                   y_pred: list,
                   id2label: dict,
                   doc_id: int):
    """
    The function then prints a table with the true and predicted class labels for each sentence in the test set.
    If the true and predicted class labels are different, the text is colored red.

      Params:
        test_df: test dataset
        y_test: true labels
        predictions: labels predicted
        id2label: mapping the label to idx
        doc_id: doc id of the document to plot
    """

    x_test = test_df.copy()
    x_test.reset_index(inplace=True)
    doc = x_test[x_test['doc_id'] == doc_id]

    print('|True class\t|Predicted class|')
    print('|---------------|---------------|')
    max_len_pred = max([len(id2label[y_pred[idx]]) for idx in doc.index])
    max_len_true = max([len(id2label[y_test[idx]]) for idx in doc.index])
    for s in range(len(doc)):
        idx = doc.index[s]
        pred = id2label[y_pred[idx]]
        true = id2label[y_test[idx]]
        sentence = re.findall(r'\S+', x_test.iloc[idx]['sentence'])
        if pred != true:
            print(
                f'|\033[91m{true.ljust(max_len_true)}\t\033[0m|\033[91m{pred.ljust(max_len_pred)}\t\033[0m|\033[91m\t{" ".join(sentence)}\033[0m')
        else:
            print(f'|{true.ljust(max_len_true)}\t|{pred.ljust(max_len_pred)}\t|\t{" ".join(sentence)}')
