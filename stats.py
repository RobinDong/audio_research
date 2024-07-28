import math
import warnings

from sklearn import metrics

warnings.filterwarnings("ignore")


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    mAP = 0
    AUC = 0

    # Class-wise statistics
    for k in range(classes_num):
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None
        )
        if math.isnan(avg_precision):
            continue
        mAP += avg_precision

        # AUC
        try:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
        except ValueError:
            auc = 0.0
            pass
        AUC += auc


    return mAP / classes_num, AUC / classes_num
