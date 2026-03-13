
def compute_roc(selected, true_set, num_features):

    selected = set(selected)

    TP = len(selected & true_set)
    FP = len(selected - true_set)
    FN = len(true_set - selected)
    TN = num_features - TP - FP - FN

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    accuracy = (TP + TN) / num_features

    return TPR, FPR, accuracy


