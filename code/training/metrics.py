from keras import backend as K
from tensorflow.keras.metrics import Precision, Recall

"""
Macro to compute average F1-score given predictions and ground-truth labels.

Args:
- y_true: ground-truth labels
- y_pred: predicted labels
"""
"""
def avg_f1(y_true, y_pred):
    
    # For positive class
    true_positives_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision_pos = true_positives_pos / (predicted_positives_pos + K.epsilon())
    recall_pos = true_positives_pos / (possible_positives_pos + K.epsilon())

    f1_pos = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))

    # For negative class
    true_positives_neg = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_positives_neg = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    predicted_positives_neg = K.sum(K.round(K.clip(1-y_pred, 0, 1)))
    
    precision_neg = true_positives_neg / (predicted_positives_neg + K.epsilon())
    recall_neg = true_positives_neg / (possible_positives_neg + K.epsilon())

    f1_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f1_pos + f1_neg) / 2"""


def avg_f1(y_true, y_pred):
    """
    Macro to compute average F1-score given predictions and ground-truth labels.

    Args:
    - y_true: ground-truth labels
    - y_pred: predicted labels
    """

    # Calculate F1-score for the positive class (class 1)
    true_positives_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision_pos = true_positives_pos / (predicted_positives_pos + K.epsilon())
    recall_pos = true_positives_pos / (possible_positives_pos + K.epsilon())

    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon())

    # Calculate F1-score for the negative class (class 0)
    true_positives_neg = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_positives_neg = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    predicted_positives_neg = K.sum(K.round(K.clip(1 - y_pred, 0, 1)))

    precision_neg = true_positives_neg / (predicted_positives_neg + K.epsilon())
    recall_neg = true_positives_neg / (possible_positives_neg + K.epsilon())

    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon())

    # Calculate the macro average F1-score
    macro_f1 = (f1_pos + f1_neg) / 2

    return macro_f1