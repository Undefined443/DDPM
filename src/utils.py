import numpy as np
import torch
from torch.utils.data import TensorDataset
import argparse

def cartesian_heart_dataset(n=8000):
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    x = (1 - np.sin(theta)) * np.cos(theta)
    y = (1 - np.sin(theta)) * np.sin(theta) + 0.9
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def perfect_heart_dataset(n=8000):
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    x = 16 * np.sin(theta) ** 3
    y = (
        13 * np.cos(theta)
        - 5 * np.cos(2 * theta)
        - 2 * np.cos(3 * theta)
        - np.cos(4 * theta)
    )
    X = np.stack((x, y), axis=1)
    X /= 5
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def mnist_dataset(train=True):
    from torchvision.datasets import MNIST
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST("./data", download=True, train=train, transform=transform)
    data = train_set.data
    data = ((data / 255.0) * 2.0) - 1.0
    data = data.reshape(-1, 28 * 28)
    return data


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