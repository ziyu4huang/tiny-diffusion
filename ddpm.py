import argparse
import os
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import omegaconf
import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding

from torch.utils.tensorboard import SummaryWriter

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        device = next(self.parameters()).device
        x = self.joint_mlp(x.to(device))
        return x


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise.cpu()

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output.cpu())
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


def _parse_args():
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval_save_image", action='store_true')
    parser.add_argument("--detail_progress", action='store_true')
    config = parser.parse_args()
    return config

class Session:
    def __init__(self) -> None:
        pass
        self.summary_writer = None

def _train(config) -> Session:
    session = Session()

    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    
    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames = []
    losses = []

    model.to(config.device)
    outdir = f"exps/{config.experiment_name}"

    writer = SummaryWriter('runs/' + config.experiment_name)
    session.summary_writer = writer
    # config.detail_progress = False
    if True: # Training loop
        print("Training model...")
        if not config.detail_progress:
            master_pb = tqdm(total=config.num_epochs)
            master_pb.set_description("Training")
        for epoch in range(config.num_epochs):
            model.train()
            if config.detail_progress:
                progress_bar = tqdm(total=len(dataloader))
                progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                data_GT = batch[0]
                noise = torch.randn(data_GT.shape)
                timesteps = torch.randint(
                    0, noise_scheduler.num_timesteps, (data_GT.shape[0],)
                ).long()

                data_noisy = noise_scheduler.add_noise(data_GT, noise, timesteps)
                noise_pred = model(data_noisy, timesteps)
                loss = F.mse_loss(noise_pred.cpu(), noise)
                loss.backward(loss)

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                logs = {"loss": loss.detach().item(), "step": global_step}
                losses.append(loss.detach().item())
                if config.detail_progress:
                    progress_bar.update(1)
                    progress_bar.set_postfix(**logs)
                global_step += 1
            if config.detail_progress:
                progress_bar.close()
            else:
                master_pb.update(1)

            if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
                # generate data with the model to later visualize the learning process
                model.eval()
                sample = torch.randn(config.eval_batch_size, 2)
                timesteps = list(range(len(noise_scheduler)))[::-1]
                for i, t in enumerate(timesteps):
                    t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                    with torch.no_grad():
                        residual = model(sample, t)
                    sample = noise_scheduler.step(residual, t[0], sample)
                frames.append(sample.numpy())
        if not config.detail_progress:
            master_pb.close()
        
    if True: # save model pth
        ####################################
        os.makedirs(outdir, exist_ok=True)
        f = f"{outdir}/model.pth"
        print(f"Saving model... {f}")
        torch.save(model.state_dict(), f)

    if True: # write images from frames
        ####################################
        imgdir = f"{outdir}/images"
        print(f"Saving images... {imgdir}")
        os.makedirs(imgdir, exist_ok=True)
        # frames = np.stack(frames) # strange , seems not necessary
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6
        for i, frame in enumerate(frames):
            plt.figure(figsize=(10, 10))
            plt.scatter(frame[:, 0], frame[:, 1])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.savefig(f"{imgdir}/{i:04}.png")
            plt.close()
        
        print("Saving loss as numpy array...")
        np.save(f"{outdir}/loss.npy", np.array(losses))

        print("Saving frames...")
        np.save(f"{outdir}/frames.npy", frames)

    return session

def _eval(config) -> Session:
    session = Session()
    
    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    outdir = f"exps/{config.experiment_name}"

    writer = SummaryWriter('runs/' + config.experiment_name)
    session.summary_writer = writer

    if True:
        print("Loading model...")
        model = MLP()
        model.load_state_dict(torch.load(config.load_model))
        model.to(config.device)

        f = "eval.png"
        print(f"Evaluating figure {f}")
        model.eval()
        sample = torch.randn(config.eval_batch_size, 2)
        timesteps = list(range(len(noise_scheduler)))[::-1]

        imgdir = f"{outdir}/images_eval"
        os.makedirs(imgdir, exist_ok=True)

        xmin, xmax = -6, 6
        ymin, ymax = -6, 6
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
            with torch.no_grad():
                residual = model(sample, t)
            sample = noise_scheduler.step(residual, t[0], sample)
            if config.eval_save_image:
                frame = sample.numpy()
                plt.figure(figsize=(10, 10))
                plt.scatter(frame[:, 0], frame[:, 1])
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.savefig(f"{imgdir}/{i:04}.png")
                plt.close()
        
    return session


if __name__ == "__main__":

    config = _parse_args()
    config = omegaconf.OmegaConf.create(vars(config))

    if not config.load_model:
        session = _train(config)
    else:
        session = _eval(config)

    # trace may cause TracerWarning

    # t_steps = torch.randint(
    #     0, noise_scheduler.num_timesteps, (sample.shape[0],)).long()

    # writer.add_graph(model, (sample, t_steps), use_strict_trace=False)

    session.summary_writer.close()

