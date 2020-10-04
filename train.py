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
import os
from datetime import datetime

import yaml
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate

from modules.dataset import Dataset
from modules.loss import SRGANLoss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-g', '--gan', action='store_true', default=False)
    return parser.parse_args()


def train_srresnet(dataloaders, srresnet, optimizer, train_config, device, start_step):
    ''' Train function for SRResNet '''
    train_dataloader, val_dataloader = dataloaders

    log_dir = os.path.join(train_config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    cur_step = start_step
    while cur_step < train_config.steps:
        # training epoch
        epoch_steps = 0
        mean_loss = 0.0
        generator.train()
        discriminator.train()
        pbar = tqdm(train_dataloader, position=0, desc='train [loss: -.-----]')
        for (hr, lr) in pbar:
            hr = hr.to(device)
            lr = lr.to(device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                hr_fake = srresnet(lr)
                loss = SRGANLoss.img_loss(hr, hr_fake)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            cur_step += 1
            epoch_steps += 1
            pbar.set_description(desc=f'train [loss: {mean_loss/epoch_steps:.5f}]')

            if cur_step % train_config.save_every == 0:
                print(f'Step {cur_step}: saving checkpoint')
                torch.save({
                    'g_state_dict': srresnet.state_dict(),
                    'g_optimizer': optimizer.state_dict(),
                    'step': cur_step,
                }, os.path.join(log_dir, f'step={cur_step}.pt'))

            # break from training loop to validate one more time
            if cur_step == train_config.steps:
                break

        # validation epoch
        epoch_steps = 0
        mean_loss = 0.0
        generator.eval()
        discriminator.eval()
        pbar = tqdm(val_dataloader, position=0, desc='val [loss: -.-----]')
        for (hr, lr) in pbar:
            hr = hr.to(device)
            lr = lr.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    hr_fake = srresnet(hr)
                    loss = SRGANLoss.img_loss(hr, hr_fake)

            mean_loss += loss.item()
            epoch_steps += 1
            pbar.set_description(desc=f'val [loss: {mean_loss/epoch_steps:.5f}]')

        # break from training loop
        if cur_step == train_config.steps:
            break


def train_srgan(dataloaders, models, optimizers, schedulers, train_config, device, start_step=0):
    ''' Train function for SRGAN '''
    # unpack modules
    train_dataloader, val_dataloader = dataloaders
    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    log_dir = os.path.join(train_config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    loss = SRGANLoss(device=device)

    cur_step = start_step
    while cur_step < train_config.steps:

        # training epoch
        epoch_steps = 0
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        pbar = tqdm(train_dataloader, position=0, desc='train [G loss: -.-----][D loss: -.-----]')
        for (hr, lr) in pbar:
            hr = hr.to(device)
            lr = lr.to(device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                g_loss, d_loss, hr_fake = loss(generator, discriminator, hr, lr)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item()
            mean_d_loss += d_loss.item()
            cur_step += 1
            epoch_steps += 1
            pbar.set_description(desc=f'train [G loss: {mean_g_loss/epoch_steps:.5f}][D loss: {mean_d_loss/epoch_steps:.5f}]')

            if cur_step % train_config.save_every == 0:
                print(f'Step {cur_step}: saving checkpoint')
                torch.save({
                    'g_state_dict': generator.state_dict(),
                    'd_state_dict': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'step': cur_step,
                }, os.path.join(log_dir, f'step={cur_step}.pt'))

            # break from training loop to validate one more time
            if cur_step == train_config.steps:
                break

            # decay learning rate by 10x
            if cur_step == train_config.decay_after:
                g_scheduler.step()
                d_scheduler.step()

        # validation epoch
        epoch_steps = 0
        mean_g_loss = 0.0
        mean_d_loss = 0.0
        pbar = tqdm(train_dataloader, position=0, desc='val [G loss: -.-----][D loss: -.-----]')
        for (hr, lr) in pbar:
            hr = hr.to(device)
            lr = lr.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, hr_fake = loss(generator, discriminator, hr, lr)

            mean_g_loss += g_loss.item()
            mean_d_loss += d_loss.item()
            epoch_steps += 1
            pbar.set_description(desc=f'val [G loss: {mean_g_loss/epoch_steps:.5f}][D loss: {mean_d_loss/epoch_steps:.5f}]')

        # break from training loop
        if cur_step == train_config.steps:
            break


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader = torch.utils.data.DataLoader(
        instantiate(config.train_dataset),
        collate_fn=Dataset.collate_fn,
        **config.train_dataloader,
    )
    val_dataloader = torch.utils.data.DataLoader(
        instantiate(config.val_dataset),
        collate_fn=Dataset.collate_fn,
        **config.val_dataloader,
    )

    start_step = 0
    if not args.gan:
        generator = instantiate(config.generator).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), **config.optim)

        if config.resume_checkpoint is not None:
            state_dict = torch.load(config.resume_checkpoint)
            generator.load_state_dict(state_dict['g_state_dict'])
            optimizer.load_state_dict(state_dict['g_optimizer'])
            start_step = state_dict['step']
            print('Starting SRResNet training from checkpoint')

        else:
            print('Starting SRResNet training from scratch')

        train_srresnet(
            [train_dataloader, val_dataloader],
            generator, optimizer, config.train_srresnet, device, start_step,
        )

    else:
        generator = instantiate(config.generator).to(device)
        discriminator = instantiate(config.discriminator).to(device)
        g_optimizer = torch.optim.Adam(generator.parameters(), **config.optim)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), **config.optim)
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lambda _: 0.1)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lambda _: 0.1)

        if config.resume_checkpoint is not None:
            state_dict = torch.load(config.resume_checkpoint)
            generator.load_state_dict(state_dict['g_state_dict'])
            discriminator.load_state_dict(state_dict['d_state_dict'])
            g_optimizer.load_state_dict(state_dict['g_optimizer'])
            d_optimizer.load_state_dict(state_dict['d_optimizer'])
            start_step = state_dict['step']
            print('Starting SRGAN training from checkpoint')

        elif config.pretrain_checkpoint is not None:
            state_dict = torch.load(config.pretrain_checkpoint)
            generator.load_state_dict(state_dict['g_state_dict'])
            print('Starting SRGAN training from pretrained generator')

        else:
            print('Starting SRGAN training from random initialization')

        train_srgan(
            [train_dataloader, val_dataloader],
            [generator, discriminator],
            [g_optimizer, d_optimizer],
            [g_scheduler, d_scheduler],
            config.train_srgan, device, start_step,
        )


if __name__ == '__main__':
    main()