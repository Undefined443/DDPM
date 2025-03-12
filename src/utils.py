import torch as th
from torch.utils.data import TensorDataset
import argparse


def cartesian_heart_dataset(n=8000):
    rng = th.Generator().manual_seed(42)
    theta = 2 * th.pi * th.rand(n, generator=rng)
    x = (1 - th.sin(theta)) * th.cos(theta)
    y = (1 - th.sin(theta)) * th.sin(theta) + 0.9
    X = th.stack((x, y), dim=1)
    X *= 3
    return TensorDataset(X.float())


def perfect_heart_dataset(n=8000):
    rng = th.Generator().manual_seed(42)
    theta = 2 * th.pi * th.rand(n, generator=rng)
    x = 16 * th.sin(theta) ** 3
    y = (
        13 * th.cos(theta)
        - 5 * th.cos(2 * theta)
        - 2 * th.cos(3 * theta)
        - th.cos(4 * theta)
    )
    X = th.stack((x, y), dim=1)
    X /= 5
    return TensorDataset(X.float())


def mnist_dataset(train=True):
    from torchvision.datasets import MNIST
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST("./data", download=True, train=train, transform=transform)
    data = train_set.data
    data = ((data / 255.0) * 2.0) - 1.0
    data = data.reshape(-1, 28 * 28)
    return TensorDataset(data.float())


def get_dataset(name, n=8000):
    if name == "heart":
        return perfect_heart_dataset(n)
    elif name == "mnist":
        return mnist_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="heart", choices=["heart", "mnist"]
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--show_image_step", type=int, default=1)
    return parser.parse_args()
