import numpy as np
import random
import torch
import evaluate


def set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def labels_frequency(df: pd.DataFrame):
    labels = list(df['label'])
    colors = dict(zip(set(labels), plt.cm.rainbow(np.linspace(0, 1, len(set(labels))))))
    for c in set(labels):
        plt.hist([labels[i] for i in range(len(labels)) if labels[i] == c], bins=np.arange(14) - 0.5, rwidth=0.7,
                 color=colors[c])
    plt.rcParams["figure.figsize"] = (20, 5)
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
