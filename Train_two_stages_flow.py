import os
import sys
import time

import torch
import torch.utils.data as data
from torch.autograd import Variable
from model.utils import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from configs.config_setting import setting_config, get_optimizer, get_scheduler
from model.flownet2.models import FlowNet2SD

import argparse
from model.vadm2 import *
from model.loss_func import *
import Evaluate_flow as Evaluate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="VADMamba")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
    parser.add_argument('--t_length', type=int, default=16, help='length of the frame sequences')
    parser.add_argument('--msize', type=int, default=50, help='number of the VQ items')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for the train loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghaitech')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log_1111', help='directory of log')
    parser.add_argument('--seed', type=int, default=1111, help='directory of log')
    parser.add_argument('--model_dir1', type=str, default='./', help='directory of model')
    parser.add_argument('--model_dir2', type=str, default='./', help='directory of model')

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
        train_folder = args.dataset_path + "/" + args.dataset_type + "/training"
    elif args.dataset_type == 'avenue':
        train_folder = args.dataset_path + "/" + args.dataset_type + "/training/training_frames"
    elif args.dataset_type == 'shanghaitech':
        train_folder = args.dataset_path + "/" + args.dataset_type + "/training/frames"

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
        transforms.ToTensor(),
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length)

    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Model setting
    model1 = vadm2(args.t_length, args.c, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=50)
    model2 = vadm2(1, args.c - 1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=10)

    config = setting_config
    optimizer1 = get_optimizer(config, model1)
    scheduler1 = get_scheduler(config, optimizer1)
    model1.cuda()

    optimizer2 = get_optimizer(config, model2)
    scheduler2 = get_scheduler(config, optimizer2)
    model2.cuda()

    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('./model/flownet2/FlowNet2-SD.pth')['state_dict'])
    flow_net.cuda().eval()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, str(args.t_length), args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    sys.stdout = f

    print('batch_size: {}, lr: {}, msize: {}, dataset_type: {}, model_continue: {}'.format(
        args.batch_size, args.lr, args.msize, args.dataset_type, args.model_continue))
    loss_func_mse = nn.MSELoss(reduction='none')
    loss_func_gradient = Gradient_Loss(3)
    # Training

    Task = 2  # 0:Pred, 1:Recon, 2:Mix, 3:No_Pred Yes_Recon  4:Yes_Pred No_Recon

    if Task == 4:
        model2_dir = os.path.join(log_dir, 'best_model2_xxx.pth')
        model2.load_state_dict(torch.load(model2_dir).state_dict())
        model2.eval()

    early_stop = {'idx': 0, 'best_eval_auc': 0}
    if Task == 0 or Task == 2 or Task == 4:
        for epoch in tqdm(range(args.epochs)):
            model1.train()
            start = time.time()
            for j, (imgs) in enumerate(train_batch):
                imgs = Variable(imgs).cuda()
                f_imgs = imgs[:, 0:3 * args.t_length, ]
                pred_imgs = imgs[:, 3 * args.t_length:, ]
                outputs, z_loss = model1.forward(f_imgs)

                optimizer1.zero_grad()
                loss_pixel = torch.mean(loss_func_mse(outputs, pred_imgs))
                loss_gradient = torch.mean(loss_func_gradient(outputs, pred_imgs))
                loss = loss_pixel + z_loss + loss_gradient
                loss.backward(retain_graph=True)
                optimizer1.step()
            scheduler1.step()

            print('----------------------------------------')
            print('Pred Epoch:', epoch + 1)
            print('Loss:  {:.6f} / Pred: {:.6f} / Z_loss: {:.6f} / Grad: {:.6f}'.format(
                loss.item(), loss_pixel.item(), z_loss.item(), loss_gradient.item()), flush=True)

            print('----------------------------------------')
            if Task == 2 or Task == 4:
                score = Evaluate.Eval(model=[model1, model2, flow_net], dateset=args.dataset_type,
                                      stage=['Pred', 'Recon'], t_length=args.t_length)
            elif Task == 0:
                score = Evaluate.Eval(model=[model1], dateset=args.dataset_type, stage='Pred', t_length=args.t_length)
            if score > early_stop['best_eval_auc']:
                early_stop['best_eval_auc'] = score
                early_stop['idx'] = 0
                torch.save(model1, os.path.join(log_dir, 'best_model1_.pth'))
            else:
                early_stop['idx'] += 1
                print('Score drop! Model not saved')

            print('With {} epochs, auc score is: {}, best score is: {}, used time: {}'
                  .format(epoch + 1, score, early_stop['best_eval_auc'], time.time() - start), flush=True)
            print(flush=True)

    print('---------------------------------------------------------------------------------------------------------')
    if Task == 2:
        # if early_stop['best_eval_auc'] > 0.8535:
        #      model1_dir = os.path.join(log_dir, 'best_model1_.pth')
        # else:
        model1_dir = os.path.join(log_dir, 'best_model1_.pth')
        model1.load_state_dict(torch.load(model1_dir).state_dict())
        model1.eval()
    elif Task == 3:
        model1_dir = os.path.join(log_dir, 'best_model1_.pth')
        model1.load_state_dict(torch.load(model1_dir).state_dict())
        model1.eval()

    early_stop1 = {'idx': 0, 'best_eval_auc': 0}
    if Task == 1 or Task == 2 or Task == 3:
        for epoch in tqdm(range(args.epochs)):
            model2.train()
            start = time.time()
            for j, (imgs) in enumerate(train_batch):
                imgs = Variable(imgs).cuda()

                input_list = []
                for i in range(args.t_length - 2, args.t_length):
                    fore_frame = imgs[:, i * 3:(i + 1) * 3, :, :]
                    back_frame = imgs[:, (i + 1) * 3:(i + 2) * 3, :, :]
                    input_flownet = torch.cat([fore_frame.unsqueeze(2), back_frame.unsqueeze(2)], 2)
                    flow_bound = (flow_net(input_flownet * 255.) / 255.).detach()  # FlowNet2SD
                    input_list.append(flow_bound)

                optical_flow = torch.cat(input_list, dim=1)
                diff_flow = optical_flow[:, 0 * 2:1 * 2, ]

                true_flow = optical_flow[:, 1 * 2:2 * 2, ]
                restore_flow, z2_loss = model2.forward(true_flow)

                optimizer2.zero_grad()
                loss_pixel_recon = torch.mean(loss_func_mse(restore_flow, true_flow))
                loss_ssim = 1 - calculate_ssim(restore_flow.cpu().detach().numpy(),
                                               true_flow.cpu().detach().numpy()).cuda()
                loss_diff = Difference_loss(restore_flow - diff_flow, true_flow - diff_flow).cuda()
                loss = loss_pixel_recon + z2_loss + loss_diff * 0.01 + loss_ssim
                loss.backward(retain_graph=True)
                optimizer2.step()
            scheduler2.step()

            print('----------------------------------------')
            print('Recon Epoch:', epoch + 1)
            print('Loss:  {:.6f} / Recon: {:.6f} / Z_loss: {:.6f} / Diff: {:.6f} / SSIM: {:.6f}'.format(
                loss.item(), loss_pixel_recon.item(), z2_loss.item(), loss_diff.item(), loss_ssim.item()),
                flush=True)
            print('----------------------------------------')

            if Task == 2 or Task == 3 or Task == 4:
                score = Evaluate.Eval(model=[model1, model2, flow_net], dateset=args.dataset_type,
                                      stage=['Pred', 'Recon'], t_length=args.t_length,
                                      auc=round(early_stop['best_eval_auc'], 8))
            elif Task == 1:
                score = Evaluate.Eval(model=[model2, flow_net], dateset=args.dataset_type, stage='Recon',
                                      t_length=args.t_length)

            if score > early_stop1['best_eval_auc']:
                early_stop1['best_eval_auc'] = score
                early_stop1['idx'] = 0
                torch.save(model2, os.path.join(log_dir, 'best_model2_.pth'))
            else:
                early_stop1['idx'] += 1
                print('Score drop! Model not saved')

            print('With {} epochs, auc score is: {}, best score is: {}, used time: {}'
                  .format(epoch + 1, score, early_stop1['best_eval_auc'], time.time() - start), flush=True)

    print('Training is finished')

    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
    main()
