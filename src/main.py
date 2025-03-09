import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from diffusion import Diffusion
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import Model
from utils import get_dataset
from utils import get_args

# animation
from celluloid import Camera
import imageio

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
SAVE_IMAGE = True

if __name__ == "__main__":
    args = get_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/{timestamp}")
    model_dir = f"models/{timestamp}"

    dataset = get_dataset(args.dataset)
    data_loader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )

    model = Model(
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        emb_size=args.embedding_size,
        twoD_data=args.dataset != "mnist",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    diffusion = Diffusion(num_timesteps=args.num_timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1 - step / (args.num_epochs * len(data_loader))
    )

    print("Training model...")
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        model.train()
        for batch in data_loader:
            x_0 = batch[0].to(device)
            noise = torch.randn_like(x_0)
            t = torch.randint(
                0, args.num_timesteps, (args.train_batch_size, 1), device=device
            )
            x_t = diffusion.diffusion(x_0, noise, t)

            pred_noise = model(x_t, t)  # 预测噪声
            loss = F.mse_loss(pred_noise, noise)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播（计算梯度）
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率

            if SAVE_IMAGE:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.scatter(x_0[:, 0].cpu(), x_0[:, 1].cpu(), alpha=0.5)
                ax1.set_title('x_0')
                ax2.scatter(x_t[:, 0].cpu(), x_t[:, 1].cpu(), alpha=0.5)
                ax2.set_title(f'x_t')
                # 将matplotlib图像转换为tensor
                writer.add_figure('scatter_plot', fig, epoch)
                plt.close(fig)
                SAVE_IMAGE = False

        writer.add_scalar("loss", loss.detach().item(), epoch)
        writer.add_scalar("learning rate", scheduler.get_last_lr()[0], epoch)

    print("Saving model...")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/model.pth")

    print("Evaluate & Save Animation")
    x_t = torch.randn(args.eval_batch_size, model.data_size, device=device)
    timesteps = range(args.num_timesteps - 1, -1, -1)
    diffusion = Diffusion(num_timesteps=args.num_timesteps, device=device)
    denoise_process = []
    denoise_process.append(x_t.to("cpu"))
    for step, t in tqdm(enumerate(timesteps), desc="Sampling"):
        t = torch.full((args.eval_batch_size, 1), t, device=device)
        with torch.no_grad():
            pred_noise = model(x_t, t)

        x_t = diffusion.denoise(pred_noise, t, x_t)
        if step % args.show_image_step == 0:
            denoise_process.append(x_t.to("cpu"))

    if args.dataset != "mnist":
        # also show forward samples
        dataset = get_dataset(args.dataset, n=args.eval_batch_size)
        x_0 = dataset.tensors[0].to(device)
        diffusion_process = []
        diffusion_process.append(x_0)
        for t in range(args.num_timesteps):
            noise = torch.randn_like(x_0, device=device)
            x_t = diffusion.diffusion(x_0, noise, t)
            if t % args.show_image_step == 0:
                diffusion_process.append(x_t.to("cpu"))

        x_min, x_max = -6, 6
        y_min, y_max = -6, 6
        fig, ax = plt.subplots()
        camera = Camera(fig)

        # for i, x in enumerate(diffusion_process + denoise_process):
        for i, x in enumerate(denoise_process):
            plt.scatter(x[:, 0], x[:, 1], alpha=0.5, s=15, color="red")
            # timesteps = i if i < len(diffusion_process) else i - len(diffusion_process)
            t = i
            ax.text(
                0.0,
                0.95,
                f"timestep {t + 1: 4} / {args.num_timesteps}",
                transform=ax.transAxes,
            )
            ax.text(
                0.0,
                1.01,
                "Denoise" if i < len(diffusion_process) else "Denoise",
                transform=ax.transAxes,
                size=15,
            )
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.axis("scaled")
            plt.axis("off")
            camera.snap()

        animation = camera.animate(blit=True, interval=200)
        animation.save(f"{model_dir}/animation_heart.gif")

    else:
        # 让动画停留 30 帧
        n_hold_final = 30
        for _ in range(n_hold_final):
            denoise_process.append(x_start)

        denoise_process = torch.stack(denoise_process, dim=0)
        denoise_process = (denoise_process.clamp(-1, 1) + 1) / 2
        denoise_process = (denoise_process * 255).type(torch.uint8)
        denoise_process = denoise_process.reshape(-1, args.eval_batch_size, 28, 28)
        denoise_process = list(torch.split(denoise_process, 1, dim=1))
        for i in range(len(denoise_process)):
            denoise_process[i] = denoise_process[i].squeeze(1)
        denoise_process = torch.cat(denoise_process, dim=-1)
        imageio.mimsave(
            f"{model_dir}/animation_mnist.gif", list(denoise_process), fps=5
        )
