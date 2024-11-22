from torch import nn
import torch
from .VM_Mamba import vmunet


class vadm2(nn.Module):
    def __init__(self, t_length=3, n_channel=3, depths=[2, 2, 2], depths_decoder=[2, 2, 1], num_embeddings=50,
                 drop_path_rate=0.2, load_ckpt_path=None, recon=False):
        super(vadm2, self).__init__()
        print('vadm')
        if t_length != 1:
            if len(depths) == 4:
                dim_en = [64, 128, 256, 512]
                dim_de = [512, 256, 128, 64]
            elif len(depths) == 3:
                dim_en = [64, 128, 256]
                dim_de = [256, 128, 64]
        else:
            if len(depths) == 4:
                dim_en = [64, 128, 256, 512]
                dim_de = [512, 256, 128, 64]
            elif len(depths) == 3:
                dim_en = [64, 128, 256]
                dim_de = [256, 128, 64]
            elif len(depths) == 2:
                dim_en = [64, 128]
                dim_de = [128, 64]
        self.model = vmunet.VMUNet(input_channels=t_length * n_channel, num_classes=n_channel, depths=depths,
                                   depths_decoder=depths_decoder, drop_path_rate=drop_path_rate, dims=dim_en,
                                   dims_decoder=dim_de, num_embeddings=num_embeddings, load_ckpt_path=load_ckpt_path,
                                   recon=recon)

    def forward(self, fx):
        out, z_loss = self.model(fx)
        return out, z_loss
