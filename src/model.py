import torch as th
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(input_size)
        self.ff = nn.Linear(input_size, input_size)
        self.act = nn.GELU()

    def forward(self, x: th.Tensor):
        return x + self.act(self.ff(self.ln(x)))


class SinusoidalEmbedding(nn.Module):
    def __init__(self, input_size: int, scale: float = 1.0):
        super().__init__()
        self.size = input_size
        self.scale = scale
        half_size = self.size // 2
        emb = th.log(th.Tensor([10000.0])) / (half_size - 1)
        emb = th.exp(-emb * th.arange(half_size))
        self.emb = nn.Parameter(emb, requires_grad=False)

    def forward(self, x: th.Tensor):
        x = x * self.scale
        emb = x * self.emb.unsqueeze(0)
        emb = th.cat((th.sin(emb), th.cos(emb)), dim=-1)
        return emb


class ToyModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        twoD_data: bool = True,
    ):
        super().__init__()
        self.twoD_data = twoD_data
        self.time_emb = SinusoidalEmbedding(emb_size)
        if twoD_data:  # heart
            self.input_emb1 = SinusoidalEmbedding(emb_size, scale=25.0)
            self.input_emb2 = SinusoidalEmbedding(emb_size, scale=25.0)
            self.concat_size = 2 * emb_size + emb_size  # 2d concat time
            self.data_size = 2
        else:  # mnist
            self.concat_size = 28 * 28 + emb_size  # mnist is 28*28
            self.data_size = 28 * 28

        layers = [nn.Linear(self.concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(ResBlock(hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.Linear(hidden_size, self.data_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t = self.time_emb(t)
        if self.twoD_data:
            x_emb = self.input_emb1(x[:, 0].unsqueeze(-1))
            y_emb = self.input_emb2(x[:, 1].unsqueeze(-1))
            x = th.cat((x_emb, y_emb), dim=-1)
        x = th.cat((x, t), dim=-1)
        x = self.joint_mlp(x)
        return x
