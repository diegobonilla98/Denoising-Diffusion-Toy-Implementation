import forward_diffusion_process
from CustomDataLoader import CustomDataLoader
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


data_set = CustomDataLoader(im_size=(128, 128))

for idx in range(len(data_set)):
    image = data_set[idx]
    plt.figure(figsize=(15, 3))
    plt.axis('off')
    num_images = 10
    step_size = int(forward_diffusion_process.T / num_images)
    for j in range(0, forward_diffusion_process.T, step_size):
        t = torch.Tensor([j]).long()

        plt.subplot(1, num_images + 1, int((j / step_size) + 1))
        image, noise = forward_diffusion_process.forward_diffusion_sample(image, t)

        plt.imshow((image.permute(1, 2, 0).cpu().data.numpy() + 1.) / 2.)
    plt.show()
