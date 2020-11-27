"""
변경: 기존 코드 -> CAE
추가: AE (flatten)

팁:
    - 최근에는 ConvTranspose2d보다 Upsample + Conv2d를 더 많이 사용하는 추세
    - norm을 사용하는 경우, pixel의 범위가 [-5.181, 10.275] (0.3352, 0.0647 인 경우)로 bound
      되기 때문에 decoder의 마지막 레이어를 잘 고려해야 함
"""
import torch.nn as nn

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16, 0.8),
            nn.ReLU(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)  # encode.shape = torch.Size([BS, 64, 4, 4])
        decoded = self.decoder(encoded) # decode.shape = torch.Size([BS, 1, 32, 32])
        return encoded, decoded

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.LeakyReLU(True),
            nn.Linear(64, 128, bias=False),
            nn.LeakyReLU(True),
            nn.Linear(128, 1024),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded