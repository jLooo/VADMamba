from torch import nn
from vadm2 import vadm2


class vadmamba(nn.Module):
    def __init__(self,):
        super(vadmamba, self).__init__()

        self.model = vadm2(16, 3, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1], num_embeddings=50)

        self.model_recon = vadm2(1, 2, depths=[3, 3, 3, 3], depths_decoder=[3, 3, 3, 1], num_embeddings=10)


    def forward(self, imgs, of):
        pred, _ = self.model(imgs)
        recon, _ = self.model_recon(of)
        return pred, recon
