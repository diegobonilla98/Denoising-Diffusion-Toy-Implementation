import glob
from io import StringIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from PIL import Image
import torch
import tqdm
import os
from torch.utils.data import Dataset
matplotlib.use('TkAgg')


class FaceDataset(Dataset):
    def __init__(self, size=(64, 64)):
        self.size = size
        folders = glob.glob('/media/bonilla/My Book/FFHQ/images1024x1024/images1024x1024/*')[:20]
        self.images_list = []
        for folder in folders:
            self.images_list += glob.glob(os.path.join(folder, '*.png'))
        self.transformation = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda l: (l * 2.) - 1.)
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        image = self.transformation(image)
        return image


class Landscapes(Dataset):
    def __init__(self, size=(64, 64)):
        self.size = size
        # self.images_list = []  # glob.glob('/media/bonilla/My Book/115_Paintings/hermitage/*')
        # folders = glob.glob('/media/bonilla/My Book/wikiart/*')
        # for folder in folders:
        #     self.images_list += glob.glob(os.path.join(folder, '*'))
        self.images_list = glob.glob('/media/bonilla/My Book/landscapes/archive/*')
        self.transformation = transforms.Compose([
            transforms.Resize(size),
            # transforms.RandomCrop(size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Lambda(lambda l: (l * 2.) - 1.)
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # with open(self.images_list[idx], 'rb') as f:
        #     check_chars = f.read()[-2:]
        # if check_chars != b'\xff\xd9':
        #     print(self.images_list[idx])
        #     return None
        # else:
        image = cv2.imread(self.images_list[idx])[:, :, ::-1]
        image = Image.fromarray(image)
        image = self.transformation(image)
        return image


class Art(Dataset):
    def __init__(self, size=(64, 64)):
        self.size = size
        # self.images_list = []  # glob.glob('/media/bonilla/My Book/115_Paintings/hermitage/*')
        # folders = glob.glob('/media/bonilla/My Book/wikiart/*')
        # for folder in folders:
        #     self.images_list += glob.glob(os.path.join(folder, '*'))
        self.images_list = glob.glob('/media/bonilla/My Book/wikiart/Color_Field_Painting/*')
        self.transformation = transforms.Compose([
            # transforms.RandomCrop(size),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda l: (l * 2.) - 1.)
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # with open(self.images_list[idx], 'rb') as f:
        #     check_chars = f.read()[-2:]
        # if check_chars != b'\xff\xd9':
        #     print(self.images_list[idx])
        #     return None
        # else:
        image = cv2.imread(self.images_list[idx])[:, :, ::-1]
        image = Image.fromarray(image)
        image = self.transformation(image)
        return image


class Dog(Dataset):
    def __init__(self, im_size=(64, 64)):
        self.files = glob.glob('/media/bonilla/My Book/DogsCats/data/train/dog*')
        self.transformation = transforms.Compose([
            transforms.Resize(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda l: (l * 2.) - 1.)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = self.transformation(image)
        return image


if __name__ == '__main__':
    dl = Art()
    for a in dl:
        pass
    # plt.imshow((a.permute(1, 2, 0).cpu().data.numpy() + 1.) / 2.)
    # plt.show()
