import numpy as np
from utils import AUC, multi_future_frames_to_scores
labels_list = np.load('./data/scores/lables_avenue.npy')
frames_scores = np.load('./data/scores/model_0.85879_avenue.npy')
obj_scores = np.load('./data/scores/avenue_0.8308002832508526.npy')

psnr_multi_list = 0.7 * np.asarray(frames_scores) + 0.3 * np.asarray(obj_scores)
psnr_multi_list = multi_future_frames_to_scores(psnr_multi_list)
accuracy = AUC(psnr_multi_list, np.expand_dims(labels_list, 0))
print(accuracy)