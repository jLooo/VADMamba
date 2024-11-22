from .vmamba import VSSM
from torch import nn


class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3, num_classes=1, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], drop_path_rate=0.2,
                 dims=None, dims_decoder=None, num_embeddings=50, load_ckpt_path=None, recon=False):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.vmunet = VSSM(in_chans=input_channels, num_classes=num_classes, depths=depths,
                           depths_decoder=depths_decoder, drop_path_rate=drop_path_rate,
                           dims=dims, dims_decoder=dims_decoder, num_embeddings=num_embeddings, recon=recon)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits, z_loss = self.vmunet(x)

        return logits, z_loss
