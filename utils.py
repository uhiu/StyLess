import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def renormalization(X, X_pert, epsilon, dataset="imagenet", use_Inc_model=False):
    if dataset == "cifar10":
        eps_added = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        return eps_added.clamp(-1.0, 1.0)
    elif dataset == "imagenet":
        eps_added = normalize_and_scale_imagenet(X_pert.detach() - X.clone(),
                                                 epsilon, use_Inc_model) + X.clone()
        mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for i in range(3):
            min_clamp = (0 - mean[i]) / stddev[i]
            max_clamp = (1 - mean[i]) / stddev[i]
            eps_added[:, i] = eps_added[:, i].clone().clamp(min_clamp, max_clamp)
        return eps_added


def normalize_and_scale_imagenet(delta_im, epsilon, use_Inc_model):
    if use_Inc_model:
        stddev_arr = [0.5, 0.5, 0.5]
    else:
        stddev_arr = [0.229, 0.224, 0.225]

    for ci in range(3):
        mag_in_scaled = epsilon / stddev_arr[ci]
        delta_im[:, ci] = delta_im[:, ci].clone().clamp(-mag_in_scaled, mag_in_scaled)

    return delta_im


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x * self.std.type_as(x)[None, :, None, None]) + \
            self.mean.type_as(x)[None, :, None, None]


class im_dataset(Dataset):
    def __init__(self, root, im_size=224, transform=None):
        self.data_dir = root
        self.imgpaths = self.get_imgpaths()

        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()])
        if transform is not None:
            self.transform = transform

    def get_imgpaths(self):
        paths = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir) if
                 x.endswith(('JPEG', 'jpg', 'png')) and not x.startswith('.')]
        paths.sort()
        return paths

    def __getitem__(self, idx):
        img_name = self.imgpaths[idx]
        file_name = os.path.splitext(os.path.basename(img_name))[0]
        image = Image.open(img_name).convert('RGB')
        image_t = self.transform(image)
        # filename as label
        return image_t, int(file_name)

    def __len__(self):
        return len(self.imgpaths)
