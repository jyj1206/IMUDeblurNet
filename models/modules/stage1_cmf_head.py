from torch import nn


class GlobalVHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_vectors,
        vector_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.num_vectors = int(num_vectors)
        self.vector_dim = int(vector_dim)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_channels, self.num_vectors * self.vector_dim),
        )

    def forward(self, x):
        v = self.body(x)
        return v.view(v.shape[0], self.num_vectors, self.vector_dim)
