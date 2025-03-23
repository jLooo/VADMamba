import numpy as np
import torch
import torch.nn as nn
import cv2
import math
import copy
from sklearn.metrics import roc_auc_score
import random
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu


def psnr(mse):
    return 10 * math.log10(1 / mse)

def psnr_llist(mse_list):
    mse = []
    for i in range(len(mse_list)):
        a = 10 * math.log10(1 / mse_list[i])
        mse.append(a)
    return mse

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def patch_max_mse(diff_map_appe, patches=3, size=16, step=4, is_multi=False):
    assert size % step == 0

    b_size = diff_map_appe.shape[0]
    max_mean = np.zeros([b_size, patches])

    # sliding window
    for i in range(0, diff_map_appe.shape[-2] - size, step):
        for j in range(0, diff_map_appe.shape[-1] - size, step):

            curr_mean = np.mean(diff_map_appe[..., i:i + size, j:j + size], axis=(1, 2, 3))
            for b in range(b_size):
                for n in range(patches):
                    if curr_mean[b] > max_mean[b, n]:
                        max_mean[b, n + 1:] = max_mean[b, n:-1]
                        max_mean[b, n] = curr_mean[b]
                        break
    return max_mean[:, 0]  #


def multi_patch_max_mse(diff_map_appe):
    mse_32 = patch_max_mse(diff_map_appe, patches=3, size=32, step=8, is_multi=False)
    mse_64 = patch_max_mse(diff_map_appe, patches=3, size=64, step=16, is_multi=False)
    mse_128 = patch_max_mse(diff_map_appe, patches=3, size=128, step=32, is_multi=False)
    return mse_32, mse_64, mse_128


def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def normalize_score_list_gel(score):  # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(anomaly_score(score[i], np.max(score), np.min(score)))
    return anomaly_score_list


def normalize_score_list_gel_recon(score):  # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(anomaly_score(score[i], np.max(score), np.min(score)))
    return anomaly_score_list

def normalize_score_motion(score, max_score, min_score):
    return (1 - (score - min_score) / (max_score - min_score))


def normalize_score_list_motion(score):  # normalize in each video and save in list form
    anomaly_score_list = list()
    for i in range(len(score)):
        anomaly_score_list.append(normalize_score_motion(score[i], np.max(score), np.min(score)))
    return anomaly_score_list


def normalize_score_clip_motion(score, max_score, min_score):
    return (1 - (score - min_score) / (max_score - min_score))


def multi_future_frames_to_scores(input):
    output = cv2.GaussianBlur(input, (5, 0), 10)
    return output


def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "auc.png"))
    plt.close()


# def score_sum(list1, list2, alpha):
#     list_result = []
#     for i in range(len(list1)):
#         # if list1[i] == 0. or list2[i] == 0.:
#         #     list_result.append(list1[i] + list2[i])
#         # else:
#         list_result.append((alpha * list1[i] + (1.1 - alpha) * list2[i]))
#     return list_result


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