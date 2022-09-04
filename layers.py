import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import random
from PIL import Image
try:
    import wandb

except ImportError:
    wandb = None

from model_new import Discriminator, MaskLoss
from model_new import Generator_next as Generator
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(x, theta):
    rot_mat = get_rot_mat(theta)[None, ...].to(device).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).to(device)
    x = F.grid_sample(x, grid)
    return x

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, image2, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    grad, = autograd.grad(
        outputs=((fake_img - image2.cuda()).pow(2)).sum(), inputs=latents, create_graph=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(2))

    return path_lengths



def g_path_regularize2(fake_img, image2, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    patch = nn.Upsample(size=(1024, 1024), mode='nearest')(fake_img[:,:,:16,:16])
    # grad, = autograd.grad(
    #     outputs=((fake_img - image2.detach().cuda()).pow(2)).sum(), inputs=latents, create_graph=True
    # )

    grad, = autograd.grad(
        outputs=((fake_img - patch.cuda()).pow(2)).sum(), inputs=latents, create_graph=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(-1).sum(-1))
    # path_lengths = grad
    # print(path_lengths.shape)

    return path_lengths

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None



def train(args, loader, generator, generator_bg, bg_extractor, discriminator, g_optim, d_optim, ema_bg, device, mean_latent):
    loader = sample_data(loader)

    pbar = range(args.iter)
    image_bg = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/0.jpg")
    image_bg1 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/1.jpg")
    image_bg2 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/2.jpg")
    image_bg3 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/3.jpg")
    image_bg4 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/4.jpg")
    image_bg5 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/5.jpg")
    image_bg6 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/6.jpg")
    image_bg7 = Image.open("/home/rameenr/stylegan2-pytorch/sg2large/stylegan2-pytorch/7.jpg")

    interpolated_correction2 = nn.Upsample(size=(1024, 1024), mode='nearest')

    image_bg = ((torch.tensor(np.array(image_bg7)/255.).unsqueeze(0) -0.5)/0.5).permute(0,3,1,2)
    image_bg = interpolated_correction2(image_bg)

    image_bg1 =  (torch.tensor(np.array(image_bg1) / 255.) - 0.5) / 0.5
    image_bg2 = (torch.tensor(np.array(image_bg2) / 255.) - 0.5) / 0.5
    image_bg3 =  (torch.tensor(np.array(image_bg3) / 255.) - 0.5) / 0.5
    image_bg4 = (torch.tensor(np.array(image_bg4) / 255.) - 0.5) / 0.5
    image_bg5 =  (torch.tensor(np.array(image_bg5) / 255.) - 0.5) / 0.5
    image_bg6 = (torch.tensor(np.array(image_bg6) / 255.) - 0.5) / 0.5
    image_bg7 = (torch.tensor(np.array(image_bg7) / 255.) - 0.5) / 0.5

    # image_bg = interpolated_correction2(torch.stack([image_bg for i in range(8)], axis = 0).permute(0,3,1,2)).type(torch.FloatTensor).cuda()

    # image_bg = interpolated_correction2(torch.stack([image_bg,image_bg1,image_bg2,image_bg3,image_bg4,image_bg5,image_bg6,image_bg7 ], axis = 0).permute(0,3,1,2).type(torch.FloatTensor).cuda())
    # print(image_bg.shape)


    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        g_module_bg = generator_bg.module
        d_module = discriminator.module
        bg_extractor_ema = bg_extractor.module

    else:
        g_module = generator
        g_module_bg = generator_bg
        d_module = discriminator
        bg_extractor_ema = bg_extractor

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)



    mask_loss_fn = MaskLoss(0.25, 2, 2)

    requires_grad(generator, True)
    # list_val = np.zeros((1,18))
    # for i in range(100):
    #     print(i)
    #     noise = mixing_noise(1, args.latent, args.mixing, device)
    #     fake_img, latents = generator(noise, return_latents=True)
    #     # fake_img, latents = generator(noise)
    #
    #     path_lengths = g_path_regularize(
    #         fake_img,image_bg, latents, mean_path_length)
    #     list_val = list_val + path_lengths.cpu().detach().numpy()
    #
    # print(list_val/100)

    list_val1 = np.zeros((1))
    list_val2 = np.zeros((1))
    list_val3 = np.zeros((1))
    list_val4 = np.zeros((1))
    for i in range(1):
        print(i)
        noise = mixing_noise(1, args.latent, args.mixing, device)

        fake_img, latents, scales = generator(noise,    back = False,  randomize_noise=True, truncation=1, truncation_latent=mean_latent)

        utils.save_image(
            fake_img,
            f"sample/{str(i+10).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(-1,1),
        )

        max__ = torch.max(latents[1]).cpu().detach().numpy()
        min__ = torch.min(latents[1]).cpu().detach().numpy()
        utils.save_image(
            latents[1],
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(np.asscalar(min__), np.asscalar(max__)),
                    )

        max__ = torch.max(latents[2]).cpu().detach().numpy()
        min__ = torch.min(latents[2]).cpu().detach().numpy()
        utils.save_image(
            latents[2],
            f"sample/{str(i+ 1).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(np.asscalar(min__), np.asscalar(max__)),
        )

        max__ = torch.max(latents[3]).cpu().detach().numpy()
        min__ = torch.min(latents[3]).cpu().detach().numpy()
        utils.save_image(
            latents[3],
            f"sample/{str(i+ 2).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(np.asscalar(min__), np.asscalar(max__)),
        )

        max__ = torch.max(latents[4]).cpu().detach().numpy()
        min__ = torch.min(latents[4]).cpu().detach().numpy()
        utils.save_image(
            latents[4],
            f"sample/{str(i+ 3).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(np.asscalar(min__), np.asscalar(max__)),
        )

        max__ = torch.max(latents[5]).cpu().detach().numpy()
        min__ = torch.min(latents[5]).cpu().detach().numpy()
        utils.save_image(
            latents[5],
            f"sample/{str(i+ 4).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(np.asscalar(min__), np.asscalar(max__)),
        )

        max__ = torch.max(latents[6]).cpu().detach().numpy()
        min__ = torch.min(latents[6]).cpu().detach().numpy()
        utils.save_image(
            latents[6],
            f"sample/{str(i+ 5).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(np.asscalar(min__), np.asscalar(max__)),
        )

        max__ = torch.max(latents[7]).cpu().detach().numpy()
        min__ = torch.min(latents[7]).cpu().detach().numpy()
        utils.save_image(
            latents[7],
            f"sample/{str(i+ 6).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(np.asscalar(min__), np.asscalar(max__)),
        )

        max__ = torch.max(latents[8]).cpu().detach().numpy()
        min__ = torch.min(latents[8]).cpu().detach().numpy()
        utils.save_image(
            latents[8],
            f"sample/{str(i+ 7).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(-1, 1),
        )

        from skimage.metrics import structural_similarity as ssim
        from skimage import io
        from skimage.transform import rescale, resize, downscale_local_mean

        i1 = io.imread('sample/000010.png')
        # i1 = resize(i1, (1024, 1024),
        #                        anti_aliasing=True)
        i2 =  io.imread('sample/0000019.png')

        i3 = io.imread('sample/000002.png')

        i3 = resize(i3, (1024, 1024),
                    anti_aliasing=True)


        print(ssim(i2, i1, data_range=i1.max() - i1.min(),  multichannel=True))
        print(ssim(i2, i3, data_range=i3.max() - i3.min(), multichannel=True))

        fake_img2, latents2, scales2 = generator(noise, back=True,  randomize_noise=False, truncation=1, truncation_latent=mean_latent)

        # path_lengths = g_path_regularize2(
        #     fake_img,fake_img2, scales, mean_path_length)
        # list_val1 = list_val1 + path_lengths.cpu().detach().numpy()

        # path_lengths1 = g_path_regularize2(
        #     fake_img, image_bg, latents[1], mean_path_length)
        # list_val2 = list_val2 + path_lengths1.cpu().detach().numpy()
        #
        # path_lengths2 = g_path_regularize2(
        #     fake_img, image_bg, latents[2], mean_path_length)
        # list_val3 = list_val3 + path_lengths2.cpu().detach().numpy()
        #
        # path_lengths3 = g_path_regularize2(
        #     fake_img, image_bg, latents[3], mean_path_length)
        # list_val4 = list_val4 + path_lengths3.cpu().detach().numpy()

    print((list_val1/1))
    # print(list_val2 / 100)
    # print(list_val3 / 100)
    # print(list_val4 / 100)
    # naum = (list_val1/100)>4900
    # naum = (list_val1 / 100) > 850
    # (unique, counts) = np.unique(naum, return_counts=True)
    # print( unique, counts)
    # mask =  1 - torch.from_numpy(naum).cuda().int().unsqueeze(-1).unsqueeze(-1)
    # # mask = torch.from_numpy(naum).cuda()
    # for j in range(10):
    #      print(j)
    #      noise = mixing_noise(1, args.latent, args.mixing, device)
    #      image, __ = generator(noise,back = True, truncation=0.5, truncation_latent=mean_latent,  mask = mask )
    #      image2, __ = generator(noise, back=False,  truncation=0.5, truncation_latent=mean_latent)
    #
    #      utils.save_image(
    #                image,
    #             f"sample/{str(j).zfill(6)}.png",
    #             nrow=int(args.n_sample ** 0.5),
    #             normalize=True,
    #             range=(-1, 1),
    #         )
    #
    #      utils.save_image(
    #         image2,
    #         f"sample/{str(10+ j).zfill(6)}.png",
    #         nrow=int(args.n_sample ** 0.5),
    #         normalize=True,
    #         range=(-1, 1),
    #     )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=10001, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        512, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    generator.eval()

    generator_bg = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    generator_bg.eval()



    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)


    g_ema = Generator(
        512, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    #
    g_ema_bg = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema_bg.eval()
    accumulate(g_ema_bg, generator_bg, 0)

    from model_new import bg_extractor

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_z2 = torch.randn(args.n_sample, args.latent, device=device)

    torch.manual_seed(102)
    bg_extractor_1 = bg_extractor().to(device)

    ema_bg = bg_extractor().to(device)
    ema_bg.eval()
    accumulate(ema_bg, bg_extractor_1, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    parm_ =  list(bg_extractor_1.parameters()) #+ list(generator.parameters())
    g_optim = optim.Adam(
        parm_,
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        # generator_bg.load_state_dict(ckpt["g_ema_bg"])
        # discriminator.load_state_dict(ckpt["d"])
        # g_ema_bg.load_state_dict(ckpt["g_ema_bg"])
        #
        # g_optim.load_state_dict(ckpt["g_optim"])
        # d_optim.load_state_dict(ckpt["d_optim"])
        # bg_extractor_1.load_state_dict(ckpt["bg_extractor_ema"])

    # ckpt2 = torch.load("stylegan2-ffhq-config-f.pt")
    ckpt2 = torch.load("stylegan2-car-config-f.pt")
    generator.load_state_dict(ckpt2['g_ema'])
    g_ema.load_state_dict(ckpt2['g_ema'])
    # discriminator.load_state_dict(ckpt2['d'])
    import pickle
    # with open('ffhq.pkl', 'rb') as f:
    #     discriminator = pickle.load(f)['D'].cuda()
    #     requires_grad(discriminator, True)
        # discriminator.eval()
    # generator.eval()
    #
    # generator = Build_model(g_ema)


    with torch.no_grad():
        mean_latent = g_ema.mean_latent(4096)


    # sample_z = torch.randn(args.n_sample, args.latent, device=device)
    # sample, _ = generator([sample_z])
    # utils.save_image(
    #     sample,
    #     f"sample/xxx.png",
    #     nrow=int(args.n_sample ** 0.5),
    #     normalize=True,
    #     range=(-1, 1),
    # )

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        generator_bg = nn.parallel.DistributedDataParallel(
            generator_bg,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        bg_extractor_1 = nn.parallel.DistributedDataParallel(
            bg_extractor_1,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator,generator_bg, bg_extractor_1,  discriminator, g_optim, d_optim, ema_bg, device, mean_latent)
