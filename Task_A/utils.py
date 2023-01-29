import numpy as np
import pandas as pd
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
                          normalize: str = None,
                          title: str = "Confusion Matrix"):
    """
      Plots the confusion matrix

      Params:
        true: true labels
        predictions: labels predicted
        labels_name: name of the labels
        normalize: normalization of the confusion matrix
        title: title of the plot
    """

    cm = confusion_matrix(true, predictions, normalize=normalize)
    cmp = ConfusionMatrixDisplay(cm, display_labels=labels_name)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    cmp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f', xticks_rotation=90, colorbar=False)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(cmp.im_, cax=cax)
