import numpy as np
import torch
from torch import nn
import cv2

def calculate_ssim(image1, image2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    ssim_all = []
    one = []
    for i in range(image1.shape[0]):
        if not image1[i].shape == image2[i].shape:
            raise ValueError('Input images must have the same dimensions.')
        h, w = 256, 256 # img1.shape[2:4]
        img1 = np.transpose(image1[i, border:h-border, border:w-border], (1,2,0))
        img2 = np.transpose(image2[i, border:h-border, border:w-border], (1,2,0))

        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            index_ndim = img1.shape[2]
            ssims = []
            # if img1.shape[2] == 3:
            #     ssims = []
            for j in range(index_ndim):
                ssims.append(ssim(img1[:,:,j], img2[:,:,j]))

            ssim_all.append(np.array(ssims).mean())
    return torch.tensor(np.array(ssim_all).mean())

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


import numpy as np


def Difference_loss(pred_diff, true_diff, epsilon=1e-3):

    diff_pred = torch.norm(pred_diff, p=2, dim=(1, 2, 3))
    diff_true = torch.norm(true_diff, p=2, dim=(1, 2, 3))
    loss = torch.sqrt((diff_pred - diff_true) ** 2 + epsilon ** 2).mean()

    return loss
#
#
# # Example usage:
# predicted_images = [np.random.rand(64, 64) for _ in range(10)]
# true_images = [np.random.rand(64, 64) for _ in range(10)]
#
# loss = compute_loss(predicted_images, true_images)
# print("Computed Loss:", loss)


class Gradient_Loss(nn.Module):
    def __init__(self, channels=1, alpha=1):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        filter = torch.FloatTensor([[-1., 1.]]).cuda()
        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, gt_frames):
        gen_frames_x = nn.functional.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = nn.functional.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = nn.functional.pad(gt_frames, (0, 0, 1, 0))
        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y)
        gt_dx = nn.functional.conv2d(gt_frames_x, self.filter_x)
        gt_dy = nn.functional.conv2d(gt_frames_y, self.filter_y)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return grad_diff_x ** self.alpha + grad_diff_y ** self.alpha


class Test_Loss(nn.Module):
    def __init__(self, channels=1, ks=(16, 8), alpha=1):
        super(Test_Loss, self).__init__()
        self.alpha = alpha
        self.ks = ks
        self.c = channels
        self.filter = torch.ones((1, 1, ks[0], ks[1]), dtype=torch.float32).cuda().repeat(1, channels, 1, 1) / (
                    ks[0] * ks[1])

    def forward(self, gen_frames):
        shape = gen_frames.size()
        b, w, h = shape[0], shape[-2], shape[-1]
        gen_frames = nn.functional.pad(gen_frames.abs().view(b, self.c, w, h),
                                       (self.ks[1], self.ks[1], self.ks[0], self.ks[0]))
        gen_dx = nn.functional.conv2d(gen_frames, self.filter).max()

        return gen_dx

class CharbonnierPenalty(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierPenalty, self).__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2):
        return torch.sqrt((x1 - x2)**2 + self.epsilon**2)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss