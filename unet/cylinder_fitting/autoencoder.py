import nn


class autoencoder(nn.Module):
    def __init__(self, iput_dims, output_dims):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential
            (
            nn.Conv1d(3, 1, kernel_size=1)
             nn.Linear(iput_dims * 3, 128),
             nn.Linear(128, output_dims),
             )

    def forward(self, x):
        x = self.encoder(x)
        return x

