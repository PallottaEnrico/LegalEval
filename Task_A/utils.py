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
