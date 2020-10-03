# SRGAN
Unofficial Pytorch implementation of SRGAN, proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (Ledig et al. 2017). Implementation for [Generative Adversarial Networks (GANs) Specialization](https://www.coursera.org/specializations/generative-adversarial-networks-gans) course material.

## Usage
1. Download the [ImageNet dataset](http://www.image-net.org/), into `data` directory and change the `USING_CIFAR` flag in `modules/dataset.py`. If you choose to run on CIFAR10, you can skip this step since `torchvision` automatically downloads it.
2. All Python requirements can be found in `requirements.txt`. Support for Python>=3.7.
3. All defaults can be found in `config.yml` and are as per the configurations described in the original paper and code.

### Training
Note that there are two separate fields in `config.yml`, `train_srresnet` and `train_srgan`, which correspond to the training parameters for SRResNet and SRGAN, respectively. By default, all checkpoints will be stored in `logs/YYYY-MM-DD_hh_mm_ss`, but this can be edited via the `train_sr*.log_dir` field in the config file. If resuming from checkpoint, populate the `resume_checkpoint` field.

1. To pretrain the generator (SRResNet) with MSE loss as described in the paper, run `python train.py`.
2. To train the full SRGAN, run `python train.py --gan`. Optionally specify the pretrained SRResNet checkpoint in `pretrain_checkpoint`, otherwise training will start from random initialization.

### Inference
1. Edit the `resume_checkpoint` field in `config.yml` to reflect the desired checkpoint from training and run `python infer.py`. This is supported for both SRResNet and SRGAN.

> Specify how many images you want to view with the `--n_show` flag. Defaults to 5.
