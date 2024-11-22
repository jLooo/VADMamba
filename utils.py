import numpy as np
import torch
import torch.nn as nn
import cv2
import math
from sklearn.metrics import roc_auc_score
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu


def psnr(mse):
    return 10 * math.log10(1 / mse)


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def normalize_score_list_gel(score):  # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(anomaly_score(score[i], np.max(score), np.min(score)))
    return anomaly_score_list


def normalize_score_motion(score, max_score, min_score):
    return (1 - (score - min_score) / (max_score - min_score))


def multi_future_frames_to_scores(input):
    output = cv2.GaussianBlur(input, (5, 0), 10)
    return output


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha * list1[i] + alpha * list2[i]))
    return list_result


def score_sum2(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((list1[i] + alpha * list2[i]))
    return list_result


def score_sum3(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((list2[i] + alpha * list1[i]))
    return list_result
