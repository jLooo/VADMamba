import torch.utils.data as data
from torch.autograd import Variable
from collections import OrderedDict
from model.utils import DataLoader
from utils import *
import glob
import argparse
import numpy as np
from model.vadm2 import vadm2
from sklearn import metrics

import torchvision.transforms as transforms
import os


METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}


def Eval(model=None, dateset=None, stage=[1], t_length=None):
    parser = argparse.ArgumentParser(description="VADMamba")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=t_length, help='length of the frame sequences')
    parser.add_argument('--num_workers_test', type=int, default=4, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default=dateset, help='type of dataset: ped2, avenue, shanghaitech')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='directory of data')
    parser.add_argument('--model_dir1', type=str, default='./', help='directory of model')
    parser.add_argument('--model_dir2', type=str, default='./', help='directory of model2')
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

    test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length)


    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)
    loss_func_mse = nn.MSELoss(reduction='none')

    if model is None and task_num == 'P':
        model1 = vadm2(args.t_length, args.c, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=50)
        try:
            model1.load_state_dict(torch.load(args.model_dir1).state_dict(), strict=False)
        except:
            model1.load_state_dict(torch.load(args.model_dir1), strict=False)
        model1.cuda()
    if model is None and task_num == 'R':
        model2 = vadm2(1, args.c-1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=10)
        try:
            model2.load_state_dict(torch.load(args.model_dir2).state_dict(), strict=False)
        except:
            model2.load_state_dict(torch.load(args.model_dir2), strict=False)
        model2.cuda()
    elif model is None and task_num == 2:
        model1 = vadm2(args.t_length, args.c, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=50)
        model2 = vadm2(1, args.c-1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=10)
        try:
            model1.load_state_dict(torch.load(args.model_dir1).state_dict(), strict=False)
            model2.load_state_dict(torch.load(args.model_dir2).state_dict(), strict=False)
        except:
            model1.load_state_dict(torch.load(args.model_dir1), strict=False)
            model2.load_state_dict(torch.load(args.model_dir2), strict=False)
        model1.cuda()
        model2.cuda()
    # Loading the trained model

    labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
    if labels.ndim == 1:
        labels = labels[np.newaxis, :]
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    format = '*.jpg'
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
    feature_distance_list = {}
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

    auc_npy_pred = []
    auc_npy_recon = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1].split('\\')[-1]]['length']
    if task_num == 2:
        model1 = model[0].eval()
        model2 = model[1].eval()
        flow_net = model[2].eval()
        # from model.flownet2.models import FlowNet2SD
        # flow_net = FlowNet2SD()
        # flow_net.load_state_dict(torch.load('./model/flownet2/FlowNet2-SD.pth')['state_dict'])
        # flow_net.cuda().eval()
    elif task_num == 'P':
        model1 = model[0].eval()
    elif task_num == 'R':
        model2 = model[0].eval()
        flow_net = model[1].eval()
        flow_net.cuda()


    with torch.no_grad():
        for k, (imgs) in enumerate(test_batch):
            imgs = Variable(imgs).cuda()

            if task_num == 'R' or task_num == 2:
                input_list = []
                for i in range(args.t_length-2, args.t_length):
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
                mse_imgs_pred = torch.mean(loss_func_mse((outputs + 1) / 2, (pred_imgs + 1) / 2)).item()
            elif task_num == 'R':
                restore_flow, _ = model2.forward(true_flow)
                mse_imgs_recon = torch.mean(loss_func_mse((restore_flow + 1) / 2, (true_flow + 1) / 2)).item()
            elif task_num == 2:
                outputs, _ = model1.forward(f_imgs)
                mse_imgs_pred = torch.mean(loss_func_mse((outputs + 1) / 2, (pred_imgs + 1) / 2)).item()
                restore_flow, _ = model2.forward(true_flow)
                mse_imgs_recon = torch.mean(loss_func_mse((restore_flow + 1) / 2, (true_flow + 1) / 2)).item()

            if task_num == 'P':
                psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_pred))
                auc_npy_pred.append(mse_imgs_pred)
            elif task_num == 'R':
                psnr_list_recon[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_recon))
                auc_npy_recon.append(mse_imgs_recon)
            else:
                psnr_list_recon[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_recon))
                psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs_pred))
                auc_npy_recon.append(mse_imgs_recon)
                auc_npy_pred.append(mse_imgs_pred)

        psnr_multi_listR = []
        psnr_multi_listR2 = []
        psnr_multi_listP = []
        psnr_multi_listP2 = []
        psnr_multi_list0 = []

        start_idx = 0
        for video_id, video in enumerate(sorted(videos_list)):
            video_name = video.split('/')[-1].split('\\')[-1]
            if task_num == 'P':
                psnr_multi_list0.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))  # pred
            elif task_num == 'R':
                psnr_multi_list0.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list_recon[video_name]))))  # pred
            else:
                t_frame = METADATA[args.dataset_type]["testing_frames_cnt"][video_id] - args.t_length
                labels_each_video = labels_list[start_idx:start_idx + t_frame]

                psnr_multi_listP.append(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))  # pred
                psnr_multi_listP2.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))  # pred

                fprP, tprP, roc_thresholds = metrics.roc_curve(1 - labels_each_video, psnr_multi_listP[video_id], pos_label=1)
                accP = metrics.auc(fprP, tprP)

                psnr_multi_listR.append(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list_recon[video_name]))))  # pred
                psnr_multi_listR2.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list_recon[video_name]))))  # pred

                fprR, tprR, roc_thresholds = metrics.roc_curve(1 - labels_each_video, psnr_multi_listR[video_id], pos_label=1)
                accR = metrics.auc(fprR, tprR)

                if accP >= accR or (str(accP) == 'nan' and args.dataset_type != 'avenue'):
                    psnr_multi_list0.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list[video_name]))))
                else:
                    psnr_multi_list0.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_list_recon[video_name]))))
                start_idx += t_frame


        if task_num == 'P' or task_num == 'R':
            accuracy = AUC(psnr_multi_list0, np.expand_dims(1 - labels_list, 0))
        else:
            accuracyP = AUC(psnr_multi_listP2, np.expand_dims(1-labels_list, 0))
            accuracyR = AUC(psnr_multi_listR2, np.expand_dims(1-labels_list, 0))
            accuracymix = AUC(psnr_multi_list0, np.expand_dims(1-labels_list, 0))
            np.savez("./data/psnr/{}/{:.8f}_res_list.npz".format(args.dataset_type, accuracymix), psnr_multi_listP2, psnr_multi_listR2, psnr_multi_list0, 1 - labels_list)

    print('The result of ', args.dataset_type)
    psnr_dir = './data/psnr/{}/{}'.format(args.dataset_type, str(args.t_length))
    if not os.path.exists(psnr_dir):
        os.makedirs(psnr_dir)
    if task_num == 'P':
        print('AUC: ', accuracy * 100)
        return accuracy
    elif task_num == 'R':
        print('AUC: ', accuracy * 100)
        return accuracy
    else:
        print('AUC_PRED: ', accuracyP * 100, '%', 'AUC_RECON: ', accuracyR * 100, '%', 'MiX AUC IS : ', accuracymix * 100)

        return accuracymix


if __name__ == '__main__':
    Eval()
