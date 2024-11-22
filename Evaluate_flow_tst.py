import torch
import torch.utils.data as data
from torch.autograd import Variable
from collections import OrderedDict
from model.utils import DataLoader
from utils import *
import random
import glob
import scipy.signal as signal
import argparse
import numpy as np
# from model.loss_func import Gradient_Loss, Test_Loss
from model.vadm2 import vadm2
import torch.nn.functional as F

import torchvision.transforms as transforms
import os


def Eval(model=None, stage='re'):
    parser = argparse.ArgumentParser(description="VADMamba")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=16, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=256, help='channel dimension of the features')
    parser.add_argument('--msize', type=int, default=50, help='number of the memory items')
    parser.add_argument('--num_workers_test', type=int, default=6, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghaitech')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='directory of data')
    parser.add_argument('--model_dir1', type=str, default='./', help='directory of model')
    parser.add_argument('--model_dir2', type=str, default='./', help='directory of model')
    parser.add_argument('--seed', type=int, default=1111, help='directory of log')

    if isinstance(stage, str):
        task_num = 'P' if stage == 'Pred' else 'R'
    elif isinstance(stage, list):
        task_num = 2

    args = parser.parse_args()
    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

    if args.dataset_type == 'ped2':
        test_folder = args.dataset_path + "/" + args.dataset_type + "/testing"
    elif args.dataset_type == 'avenue':
        test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/testing_frames"
    else:
        test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, train=False, time_step=args.t_length)

    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)
    loss_func_mse = nn.MSELoss(reduction='none')
    loss_func_l1 = nn.L1Loss(reduction='none')

    if model is None and task_num == 'P':  # if not training, we give a exist model and params path
        model1 = vadm2(1, args.c - 1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=10)
        try:
            model1.load_state_dict(torch.load(args.model_dir1).state_dict())
        except:
            model1.load_state_dict(torch.load(args.model_dir1))
        model1.cuda()
    elif model is None and task_num == 'R':  # if not training, we give a exist model and params path
        model2 = vadm2(1, args.c - 1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=10)
        try:
            model2.load_state_dict(torch.load(args.model_dir2).state_dict())
        except:
            model2.load_state_dict(torch.load(args.model_dir2), strict=False)
        model2.cuda()
    elif model is None and task_num == 2:
        model1 = vadm2(args.t_length, args.c, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=50)
        model2 = vadm2(1, args.c - 1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=10)
        try:
            model1.load_state_dict(torch.load(args.model_dir1).state_dict())
            model2.load_state_dict(torch.load(args.model_dir2).state_dict())
        except:
            model1.load_state_dict(torch.load(args.model_dir1))
            model2.load_state_dict(torch.load(args.model_dir2))
        model1.cuda()
        model2.cuda()

    labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
    if labels.ndim == 1:
        labels = labels[np.newaxis, :]
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    format = '*.jpg'  # if args.dataset_type == 'ped2' else '*.jpg'
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, format))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    psnr_list1 = {}
    labels_list_recon = []
    feature_distance_list = {}
    labels_list_mix = []
    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1].split('\\')[-1]
        labels_list = np.append(labels_list,
                                labels[0][args.t_length + label_length:videos[video_name]['length'] + label_length])

        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        psnr_list1[video_name] = []
        feature_distance_list[video_name] = []

    psnr_list_recon = copy.deepcopy(psnr_list)

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1].split('\\')[-1]]['length']
    from model.flownet2.models import FlowNet2SD
    if task_num == 2:
        model1 = model1.eval()
        model2 = model2.eval()
        flow_net = FlowNet2SD()
        flow_net.load_state_dict(torch.load('./model/flownet2/FlowNet2-SD.pth')['state_dict'])
        flow_net.cuda().eval()

    elif task_num == 'P':
        model1 = model1.eval()
    elif task_num == 'R':
        model2 = model2.eval()
        flow_net = FlowNet2SD()
        flow_net.load_state_dict(torch.load('./model/flownet2/FlowNet2-SD.pth')['state_dict'])
        flow_net.cuda().eval()

    with torch.no_grad():
        for k, (imgs) in enumerate(test_batch):  # , img128_64
            imgs = Variable(imgs).cuda()

            if task_num == 'R' or task_num == 2:
                input_list = []
                for i in range(args.t_length - 2, args.t_length):
                    fore_frame = imgs[:, i * 3:(i + 1) * 3, :, :]
                    back_frame = imgs[:, (i + 1) * 3:(i + 2) * 3, :, :]
                    input_flownet = torch.cat([fore_frame.unsqueeze(2), back_frame.unsqueeze(2)], 2)
                    flow_bound = (flow_net(input_flownet * 255.) / 255.).detach()  # FlowNet2SD
                    input_list.append(flow_bound)
                optical_flow = torch.cat(input_list, dim=1)

                true_flow = optical_flow[:, 1 * 2:2 * 2, ]

            f_imgs = imgs[:, 0:3 * args.t_length, ]
            pred_imgs = imgs[:, 3 * args.t_length:, ]

            if k == label_length - args.t_length * (video_num + 1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1].split('\\')[-1]]['length']
            if task_num == 'P':
                outputs, _ = model1.forward(f_imgs)
                mse_imgs_pred = torch.mean(loss_func_mse((outputs + 1) / 2, (pred_imgs + 1) / 2)).item()  # psnr   2
            elif task_num == 'R':
                restore_flow, _ = model2.forward(true_flow)
                mse_imgs_recon = torch.mean(
                    loss_func_mse((restore_flow + 1) / 2, (true_flow + 1) / 2)).item()  # psnr   2
            elif task_num == 2:

                outputs, _ = model1.forward(f_imgs)
                restore_flow, _ = model2.forward(true_flow)

                mse_imgs_pred = torch.mean(loss_func_mse((outputs + 1) / 2, (pred_imgs + 1) / 2)).item()  # psnr   2
                mse_imgs_recon = torch.mean(loss_func_mse((restore_flow + 1) / 2, (true_flow + 1) / 2)).item()

            if task_num == 'P':
                psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_pred))  # psnr
            elif task_num == 'R':
                psnr_list_recon[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_recon))  # psnr
            else:
                psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_pred))  # psnr
                psnr_list_recon[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_recon))  # psnr
        psnr_multi_listR = []
        psnr_multi_listP = []
        psnr_multi_list0 = []
        psnr_multi_list1 = []
        psnr_multi_list2 = []
        psnr_multi_list3 = []
        psnr_multi_list4 = []
        psnr_multi_list5 = []
        psnr_multi_list6 = []
        psnr_multi_list7 = []
        psnr_multi_list8 = []
        psnr_multi_list9 = []

        psnr_multi_list01 = []
        psnr_multi_list02 = []
        psnr_multi_list04 = []
        psnr_multi_list06 = []
        psnr_multi_list08 = []
        psnr_multi_list10 = []
        psnr_multi_list20 = []
        psnr_multi_list40 = []
        psnr_multi_list60 = []
        psnr_multi_list80 = []

        for video in sorted(videos_list):
            video_name = video.split('/')[-1].split('\\')[-1]
            if task_num == 'P':
                psnr_multi_list0.extend(
                    multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))  # pred
            elif task_num == 'R':
                psnr_multi_list0.extend(multi_future_frames_to_scores(
                    np.array(normalize_score_list_gel(psnr_list_recon[video_name]))))  # pred
            else:
                psnr_multi_listP.extend(
                    multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))  # pred
                psnr_multi_listR.extend(multi_future_frames_to_scores(
                    np.array(normalize_score_list_gel(psnr_list_recon[video_name]))))  # pred
                psnr_multi_list0.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=1))))
                psnr_multi_list01.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.1))))
                psnr_multi_list02.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.2))))
                psnr_multi_list1.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.3))))
                psnr_multi_list04.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.4))))
                psnr_multi_list2.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.5))))
                psnr_multi_list06.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.6))))
                psnr_multi_list3.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.7))))
                psnr_multi_list08.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.8))))
                psnr_multi_list4.extend(
                    multi_future_frames_to_scores(np.array(score_sum2(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.9))))
                psnr_multi_list10.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.1))))
                psnr_multi_list20.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.2))))
                psnr_multi_list6.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.3))))
                psnr_multi_list40.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.4))))
                psnr_multi_list7.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.5))))
                psnr_multi_list60.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.6))))
                psnr_multi_list8.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.7))))
                psnr_multi_list80.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.8))))
                psnr_multi_list9.extend(
                    multi_future_frames_to_scores(np.array(score_sum3(normalize_score_list_gel(psnr_list[video_name]),
                                                                      normalize_score_list_gel(
                                                                          psnr_list_recon[video_name]), alpha=0.9))))

        if task_num == 'P' or task_num == 'R':
            accuracy = AUC(psnr_multi_list0, np.expand_dims(1 - labels_list, 0))
        else:
            accuracyP = AUC(psnr_multi_listP, np.expand_dims(1 - labels_list, 0))
            accuracyR = AUC(psnr_multi_listR, np.expand_dims(1 - labels_list, 0))

            accuracy = AUC(psnr_multi_list0, np.expand_dims(1 - labels_list, 0))
            accuracy1 = AUC(psnr_multi_list1, np.expand_dims(1 - labels_list, 0))
            accuracy2 = AUC(psnr_multi_list2, np.expand_dims(1 - labels_list, 0))
            accuracy3 = AUC(psnr_multi_list3, np.expand_dims(1 - labels_list, 0))
            accuracy4 = AUC(psnr_multi_list4, np.expand_dims(1 - labels_list, 0))
            # accuracy5 = AUC(psnr_multi_list5, np.expand_dims(1-labels_list, 0))
            accuracy6 = AUC(psnr_multi_list6, np.expand_dims(1 - labels_list, 0))
            accuracy7 = AUC(psnr_multi_list7, np.expand_dims(1 - labels_list, 0))
            accuracy8 = AUC(psnr_multi_list8, np.expand_dims(1 - labels_list, 0))
            accuracy9 = AUC(psnr_multi_list9, np.expand_dims(1 - labels_list, 0))

            accuracy01 = AUC(psnr_multi_list01, np.expand_dims(1 - labels_list, 0))
            accuracy02 = AUC(psnr_multi_list02, np.expand_dims(1 - labels_list, 0))
            accuracy04 = AUC(psnr_multi_list04, np.expand_dims(1 - labels_list, 0))
            accuracy06 = AUC(psnr_multi_list06, np.expand_dims(1 - labels_list, 0))
            accuracy08 = AUC(psnr_multi_list08, np.expand_dims(1 - labels_list, 0))
            accuracy10 = AUC(psnr_multi_list10, np.expand_dims(1 - labels_list, 0))
            accuracy20 = AUC(psnr_multi_list20, np.expand_dims(1 - labels_list, 0))
            accuracy40 = AUC(psnr_multi_list40, np.expand_dims(1 - labels_list, 0))
            accuracy60 = AUC(psnr_multi_list60, np.expand_dims(1 - labels_list, 0))
            accuracy80 = AUC(psnr_multi_list80, np.expand_dims(1 - labels_list, 0))

    print('The result of ', args.dataset_type)
    if not os.path.exists(psnr_dir):
        os.makedirs(psnr_dir)
    if task_num == 'P':
        print('AUC: ', accuracy * 100)
        np.save(psnr_dir + '/pred_{}.npy'.format(str(accuracy)), accuracy)
        return accuracy
    elif task_num == 'R':
        print('AUC: ', accuracy * 100)
        np.save(psnr_dir + '/recon_{}.npy'.format(str(accuracy)), accuracy)
        return accuracy
    else:
        print('AUC_PRED: ', accuracyP * 100, '%', 'AUC_RECON: ', accuracyR * 100, '%', 'AUC1+1: ', accuracy * 100, '%',
              'AUC1_0.1: ', accuracy01 * 100, 'AUC1_0.2: ', accuracy02 * 100, 'AUC1_0.3: ', accuracy1 * 100,
              'AUC1_0.4: ', accuracy04 * 100, '%', 'AUC1_0.5: ', accuracy2 * 100, '%', 'AUC1_0.6: ', accuracy06 * 100,
              'AUC1_0.7: ', accuracy3 * 100, '%', 'AUC1_0.8: ', accuracy08 * 100, 'AUC1_0.9: ', accuracy4 * 100,
              'AUC0.1_1: ', accuracy10 * 100, '%', 'AUC0.2_1: ', accuracy20 * 100, '%', 'AUC0.3_1: ', accuracy6 * 100,
              '%', 'AUC0.4_1: ', accuracy40 * 100, '%', 'AUC0.5_1: ', accuracy7 * 100, '%',
              'AUC0.6_1: ', accuracy60 * 100, '%', 'AUC0.7_1: ', accuracy8 * 100, '%', 'AUC0.8_1: ', accuracy80 * 100,
              '%', 'AUC0.9_1: ', accuracy9 * 100, '%')

        all_auc = [accuracy, accuracy01, accuracy02, accuracy1, accuracy04, accuracy2, accuracy06, accuracy3,
                   accuracy08, accuracy4, accuracy10, accuracy20, accuracy6, accuracy40, accuracy7, accuracy60,
                   accuracy8, accuracy80, accuracy9]
        max_accuracy = max(all_auc)
        max_index = all_auc.index(max(all_auc))
        print('MAX AUC IS : ', max_index)
        return max_accuracy


if __name__ == '__main__':
    Eval()
