"""
변경: 기존 코드 -> CAE
추가: AE (flatten)

팁:
    - 최근에는 ConvTranspose2d보다 Upsample + Conv2d를 더 많이 사용하는 추세
    - norm을 사용하는 경우, pixel의 범위가 [-5.181, 10.275] (0.3352, 0.0647 인 경우)로 bound
      되기 때문에 decoder의 마지막 레이어를 잘 고려해야 함
"""
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 10, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, 10, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, 10, stride=1, padding=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 10, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 10, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 10, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)  # encode.shape = torch.Size([32, 1, 17, 17])
        decoded = self.decoder(encoded) # decode.shape = torch.Size([32, 48, 17, 17])
        return encoded, decoded

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 1024),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded