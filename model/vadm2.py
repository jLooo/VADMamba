from torch import nn
from .VQMaUnet import VQMaU

class vadm2(nn.Module):
    def __init__(self, t_length=3, n_channel=3, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=50,
                 drop_path_rate=0.2, recon=False):
        super(vadm2, self).__init__()
        if len(depths) == 4:
            dim_en = [64, 128, 256, 512]
            dim_de = [512, 256, 128, 64]
        elif len(depths) == 3:
            dim_en = [64, 128, 256]
            dim_de = [256, 128, 64]
        elif len(depths) == 2:
            dim_en = [64, 128]
            dim_de = [128, 64]
        self.model = VQMaU.VQMaUet(input_channels=t_length*n_channel, num_classes=n_channel, depths=depths,
            depths_decoder=depths_decoder, drop_path_rate=drop_path_rate, dims=dim_en, dims_decoder=dim_de,
                                   num_embeddings=num_embeddings, recon=recon)

    def forward(self, fx):
        out, z_loss = self.model(fx)
        return out, z_loss



