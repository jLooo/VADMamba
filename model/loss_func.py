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

def Difference_loss(pred_diff, true_diff, epsilon=1e-3):

    diff_pred = torch.norm(pred_diff, p=2, dim=(1, 2, 3))
    diff_true = torch.norm(true_diff, p=2, dim=(1, 2, 3))
    loss = torch.sqrt((diff_pred - diff_true) ** 2 + epsilon ** 2).mean()

    return loss

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
