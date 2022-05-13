import torch as th
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, emb_dim, width, num_layers, dropout=0.0):
        super(MLP, self).__init__()

        layers = []

        layers.append(nn.Linear(emb_dim, width))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers):
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=dropout))
    
        layers.append(nn.Linear(width, emb_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP_mixer(nn.Module):

    def __init__(self, emb_dim, width, num_layers, dropout=0.1):
        super(MLP_mixer, self).__init__()

        class PreNormResidual(nn.Module):
            def __init__(self, dim, fn):
                super().__init__()
                self.fn = fn
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                return self.fn(self.norm(x)) + x

        def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
            inner_dim = int(dim * expansion_factor)
            return nn.Sequential(
                dense(dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                dense(inner_dim, dim),
                nn.Dropout(dropout)
            )

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, width),
            *[
                PreNormResidual(width, FeedForward(width, 2, dropout)) for _ in range(num_layers)],
            nn.LayerNorm(width),
            nn.Linear(width, emb_dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x