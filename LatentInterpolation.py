import cv2
import tqdm
import forward_diffusion_process
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')


def slerp_theta(z1, z2, theta):
    return torch.cos(theta) * z1 + torch.sin(theta) * z2


USE_CUDA = torch.cuda.is_available()
T = forward_diffusion_process.T
model_path = './checkpoints/landscapes128x128/landscapes_model.pth'
model = torch.load(model_path)


def normalize01(x, axis=None):
    return (x - np.min(x, axis=axis)) / (np.max(x, axis=axis) - np.min(x, axis=axis))


@torch.no_grad()
def sample_plot_image(noise):
    img = noise  # torch.randn((1, 3, *(128, 128)), device="cuda" if USE_CUDA else "cpu")
    with tqdm.tqdm(total=T) as pbar:
        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device="cuda" if USE_CUDA else "cpu", dtype=torch.long)
            img = forward_diffusion_process.sample_timestep(img, t, model)
            if i == 0:
                return np.uint8(normalize01(img[0].permute(1, 2, 0).cpu().data.numpy()) * 255.)
            pbar.update()


noiseA = torch.randn((1, 3, *(128, 128)), device="cuda" if USE_CUDA else "cpu")
noiseB = torch.randn((1, 3, *(128, 128)), device="cuda" if USE_CUDA else "cpu")
images = []
for val in torch.linspace(0, np.pi / 2, 10):
    images.append(cv2.resize(sample_plot_image(slerp_theta(noiseA, noiseB, val)), None, fx=3., fy=3., interpolation=cv2.INTER_LANCZOS4)[:, :, ::-1])

images = np.hstack(images)
cv2.imshow("Output", images)
cv2.waitKey()
cv2.imwrite("landscapes_interp.png", images)
