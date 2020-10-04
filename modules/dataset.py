# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms


def collate_fn(batch):
    hrs, lrs = [], []
    for hr, lr in batch:
        hrs.append(hr)
        lrs.append(lr)

    return torch.stack(hrs, dim=0), torch.stack(lrs, dim=0)


class STL10(torchvision.datasets.STL10):
    def __init__(self, *args, **kwargs):
        hr_size = kwargs.pop('hr_size', [64, 64])
        lr_size = kwargs.pop('lr_size', [16, 16])
        sr_size = kwargs.pop('n_sr', 2) ** 2
        super().__init__(*args, **kwargs)

        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == sr_size * lr_size[0]
            assert hr_size[1] == sr_size * lr_size[1]

        # High-res images are cropped and scaled to [-1, 1]
        self.hr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(hr_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Low-res images are downsampled with bicubic kernel and scaled to [0, 1]
        self.lr_transforms = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
            transforms.Resize(lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx])

        hr = self.hr_transforms(image)
        lr = self.lr_transforms(hr)
        return hr, lr

    def __len__(self):
        return len(self.data)


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, *args, **kwargs):
        hr_size = kwargs.pop('hr_size', [384, 384])
        lr_size = kwargs.pop('lr_size', [96, 96])
        sr_size = kwargs.pop('n_sr', 2) ** 2
        super().__init__(*args, **kwargs)

        if hr_size is not None and lr_size is not None:
            assert hr_size[0] == sr_size * lr_size[0]
            assert hr_size[1] == sr_size * lr_size[1]

        # High-res images are cropped and scaled to [-1, 1]
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop(hr_size),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Low-res images are downsampled with bicubic kernel and scaled to [0, 1]
        self.lr_transforms = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage(),
            transforms.Resize(lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        path, _ = self.imgs[idx]
        image = Image.open(path).convert('RGB')

        hr = self.hr_transforms(image)
        lr = self.lr_transforms(hr)
        return hr, lr

    def __len__(self):
        return len(self.imgs)
