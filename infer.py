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

import argparse

import yaml
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, show_n=1):
    ''' For visualizing images '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:show_n], nrow=show_n)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-n', '--n_show', type=int, default=5)
    return parser.parse_args()


def main():
    ''' Runs inference on validation dataset '''
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.resume_checkpoint is not None:
        checkpoint = config.resume_checkpoint
    elif config.pretrain_checkpoint is not None:
        checkpoint = config.pretrain_checkpoint
    else:
        raise ValueError(
            'Specify a checkpoint to load in config.resume_checkpoint or config.pretrain_checkpoint'
        )

    generator = instantiate(config.generator).to(device).eval()
    generator.load_state_dict(torch.load(checkpoint)['g_state_dict'])

    config.val_dataloader.batch_size = 1
    dataloader = torch.utils.data.DataLoader(
        instantiate(config.val_dataset), **config.val_dataloader,
    )

    n = 0
    for (hr, lr) in dataloader:
        if n == args.n_show:
            break

        hr = hr.to(device)
        lr = lr.to(device)

        with torch.no_grad():
            hr_fake = generator(lr)

        show_tensor_images(hr_fake.to(hr.dtype))
        show_tensor_images(hr)

        n += 1


if __name__ == '__main__':
    main()