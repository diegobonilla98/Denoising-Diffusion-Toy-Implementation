import cv2
import tqdm
import forward_diffusion_process
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')


USE_CUDA = torch.cuda.is_available()
T = forward_diffusion_process.T
model_path = './checkpoints/landscapes128x128/landscapes_model.pth'
model = torch.load(model_path)


def normalize01(x, axis=None):
    return (x - np.min(x, axis=axis)) / (np.max(x, axis=axis) - np.min(x, axis=axis))


@torch.no_grad()
def sample_plot_image():
    num_images = 10
    img = torch.randn((1, 3, *(128, 128)), device="cuda" if USE_CUDA else "cpu")
    stepsize = int(T / num_images)
    with tqdm.tqdm(total=T) as pbar:
        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device="cuda" if USE_CUDA else "cpu", dtype=torch.long)
            img = forward_diffusion_process.sample_timestep(img, t, model)
            if i == 0:
                return np.uint8(normalize01(img[0].permute(1, 2, 0).cpu().data.numpy()) * 255.)
            pbar.update()


n_w = 5
n_h = 5
images_column = []
for w in range(n_w):
    images_row = []
    for h in range(n_h):
        images_row.append(cv2.resize(sample_plot_image(), None, fx=3., fy=3., interpolation=cv2.INTER_LANCZOS4)[:, :, ::-1])
    images_column.append(np.hstack(images_row))

images = np.vstack(images_column)
cv2.imshow("Output", images)
cv2.waitKey()
cv2.imwrite("landscapes_mat.png", images)
