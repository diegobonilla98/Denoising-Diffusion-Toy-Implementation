from ModelOther import Unet
import forward_diffusion_process
import torch.nn
from torch.nn import functional as F
import tqdm
import cv2
from CustomDataLoader import Landscapes
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sendNotification import pushbullet_image, pushbullet_notification
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="matplotlib\..*" )


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_process.forward_diffusion_sample(x_0, t, "cuda" if USE_CUDA else "cpu")
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_plot_image(epoch):
    num_images = 5
    img = torch.randn((1, 3, *IMG_SIZE), device="cuda" if USE_CUDA else "cpu")
    stepsize = int(T / num_images)
    denoised_images = []
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device="cuda" if USE_CUDA else "cpu", dtype=torch.long)
        img = forward_diffusion_process.sample_timestep(img, t, model)
        if i % stepsize == 0:
            denoised_images.append(cv2.pyrUp(np.uint8((img[0].permute(1, 2, 0).cpu().data.numpy() + 1.) * 127.5)))
    cv2.imwrite(f"./checkpoints/landscapes128x128/epoch_{epoch}.png", np.hstack(denoised_images)[:, :, ::-1])
    return f"./checkpoints/landscapes128x128/epoch_{epoch}.png"


IMG_SIZE = (128, 128)
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 500
T = forward_diffusion_process.T

model = Unet(64)
print(model)

dataset = Landscapes(IMG_SIZE)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: 0.9 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

if USE_CUDA:
    model = model.cuda()

for p in model.parameters():
    p.requires_grad = True

pushbullet_notification("Training started!", f"Now")
for epoch in range(N_EPOCHS + 1):
    with tqdm.tqdm(total=len(data_loader)) as pbar:
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device="cuda" if USE_CUDA else "cpu").long()
            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            pbar.update()
            pbar.set_description(f"Epoch {epoch}/{N_EPOCHS} | step {step}/{len(data_loader)} Loss: {loss.item()}")
        scheduler.step(epoch)
    filepath = sample_plot_image(epoch)
    pushbullet_notification("New epoch!", f"Epoch {epoch}/{N_EPOCHS}")
    pushbullet_image(filepath)
    torch.save(model, './checkpoints/landscapes128x128/landscapes_model.pth')

