import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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