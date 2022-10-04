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
try:
    import wandb

except ImportError:
    wandb = None

from model_new import Generator, Discriminator, bg_extractor_repro, bg_extractor
from dataset import MultiResolutionDataset, TestDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    """Returns a data sampler for a dataloader.

    Args:
        dataset (torch.utils.data.Dataset): a dataset class as an input to a model.
        shuffle (bool): whether to shuffle indices for a dataloading.
        distributed (bool): whether to use distributed sampler (multi-gpu training)
    """
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
        """parameter를 이런 식으로 해주는 건 대체 어디서 나온 걸까? -> weight ema"""
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

# regularization 부분은 PertSeg paper에서 따온 아이디어였음.
def binarization_loss(mask):
    "Refer to PertSeg paper"
    return  torch.min(1-mask, mask).mean()

def fg_mask_coverage_loss(mask, min_mask_coverage):
    "Refer to PertSeg paper"
    return F.relu(min_mask_coverage - mask.mean(dim=(2, 3))).mean()

def bg_mask_coverage_loss(mask, max_mask_coverage):
    return F.relu(max_mask_coverage - (1- mask.mean(dim=(2, 3)))).mean()


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    """
    softplus를 loss함수로 쓴다..?? 신기하군
    real pred는 왜 -로 집어넣는 거지?
    """
    real_loss = F.softplus(-real_pred) # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    """
    이 loss는 어디서 쓰는 거지?
    """
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred): # 왜 애초에 nonsaturaing loss를 쓰는 거지?
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0) # unbind?

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets): # 특정 parameter만 grad를 none으로
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def expand_path(path):
    core_path = "/".join(path.split("/")[:-1])
    os.makedirs(core_path, exist_ok=True)
    return path

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train(args, loader, generator, bg_extractor, discriminator, g_optim, d_optim, ema_bg, device, mean_latent):
    """
    Train a model

    Args:
        ema_bg (_type_): _description_
        mean_latent (_type_): _description_
    """
    loader = sample_data(loader)

    pbar = range(args.iter + 1)
 
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
  
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        bg_extractor_ema = bg_extractor.module # ema?

    else:
        g_module = generator
        d_module = discriminator
        bg_extractor_ema = bg_extractor

    accum = 0.5 ** (32 / (10 * 1000)) # accum은 ema를 위한 accum
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0 # ada를 위한 augmentation probability setting

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)


    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter + 1:
            print("Done!")

            break
        real_img = next(loader)
        real_img = real_img.to(device)

        # gradient freeze
        requires_grad(generator, False)
        requires_grad(bg_extractor, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        

        with torch.no_grad(): # torch no grad는 freeze니까
            fake_img, _ = generator(noise, back = False, truncation=args.trunc, truncation_latent=mean_latent)
            fake_img2, x = generator(noise, back = True)

        alpha_mask = bg_extractor(_) # list 집어넣기 (근데 왜 변수할당 안하고 이렇게 하는지 잘 모르겠음)


        if args.augment: # augmentation 진행
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img
        
        fake_image =  fake_img * alpha_mask + (1 - alpha_mask ) * fake_img2
  

        fake_pred = discriminator(fake_image)

        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred) 

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 1.0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, False)
        requires_grad(bg_extractor, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
    
        with torch.no_grad():
         fake_img, _ = generator(noise, back=False, truncation=args.trunc, truncation_latent=mean_latent)
         fake_img2, x = generator(noise,  back=True)

        alpha_mask = bg_extractor(_)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_image = fake_img * alpha_mask + (1 - alpha_mask) * fake_img2


        fake_pred = discriminator(fake_image)
     
        g_loss = args.sat_weight * g_nonsaturating_loss(fake_pred)  + args.loss_multiplier* (binarization_loss(alpha_mask) +  fg_mask_coverage_loss(alpha_mask, args.fg_coverage_value) ) +  args.bg_coverage_wt * bg_mask_coverage_loss(alpha_mask, args.bg_coverage_value)

        loss_dict["g"] = g_loss

        bg_extractor.zero_grad()
        g_loss.backward()
        g_optim.step()


        accumulate(ema_bg, bg_extractor_ema, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
      
        
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                      
                    }
                )

            if (i+1) % args.model_save_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z], truncation=0.5, truncation_latent=mean_latent, back = False)
                    sample_bg, __ = g_ema([sample_z2], truncation=0.5, truncation_latent=mean_latent, back = True)
              
                    alpha_mask = bg_extractor(_)
    
                    image_new = sample * alpha_mask + (1 - alpha_mask) * sample_bg

                    if args.save_image:
                        utils.save_image(
                            image_new,
                            expand_path(f"sample/{args.run_name}/{str(i).zfill(6)}_composite.png"),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            value_range=(-1, 1),
                        )
                        utils.save_image(
                            alpha_mask,
                            expand_path(f"sample/{args.run_name}/{str(i).zfill(6)}_alpha_mask.png"),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=False,
                        )

                        utils.save_image(
                            sample,
                            expand_path(f"sample/{args.run_name}/{str(i).zfill(6)}_original.png"),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            value_range=(-1, 1),
                        )
                        utils.save_image(
                            sample_bg,
                            expand_path(f"sample/{args.run_name}/{str(i).zfill(6)}_background.png"),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            value_range=(-1, 1),
                        )
                    
                    if wandb and args.wandb:
                        wandb.log(
                            {
                                "Composite": wandb.Image(utils.make_grid(image_new, nrow=int(args.n_sample ** 0.5), normalize=True, value_range=(-1, 1)).permute(1,2,0).cpu().numpy()),
                                "Alpha Mask": wandb.Image(utils.make_grid(alpha_mask, nrow=int(args.n_sample ** 0.5), normalize=False).permute(1,2,0).cpu().numpy()),
                                "Original Sample": wandb.Image(utils.make_grid(sample, nrow=int(args.n_sample ** 0.5), normalize=True, value_range=(-1, 1)).permute(1,2,0).cpu().numpy()),
                                "Generated Background": wandb.Image(utils.make_grid(sample_bg, nrow=int(args.n_sample ** 0.5), normalize=True, value_range=(-1, 1)).permute(1,2,0).cpu().numpy())
                            }
                        )

                
                    torch.save(
                     {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "bg_extractor": bg_extractor.state_dict(),
                        "bg_extractor_ema": bg_extractor_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                     },
                    expand_path(f"checkpoint/{args.run_name}/{str(i).zfill(6)}.pt"),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 Alpha Network trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--use_lmdb", action="store_true", help="flag of whether to use lmdb dataset or not")
    parser.add_argument(
        "--iter", type=int, default=1001, help="total training iterations"
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
        "--reproduce_model", action="store_true", help="reproduce model in the paper"
    )
    parser.add_argument(
        "--use_disc", action="store_true", help="use pretrained discriminator"
    )
    parser.add_argument(
        "--pretrained_alphanet", action="store_true", help="use pretrained alpha network"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--loss_multiplier", type=float, default=1.2, help="weight of the mask regularization"
    )
    parser.add_argument(
        "--trunc", type=float, default=1., help="trucation value for the generator"
    )

    parser.add_argument(
        "--model_save_freq", type=int, default=100, help="model saving frequency"
    )
    parser.add_argument(
        "--bg_coverage_wt", type=float, default=0., help="bg coverage weight"
    )
    parser.add_argument(
        "--bg_coverage_value", type=float, default=0.75, help="bg coverage value"
    )
    parser.add_argument(
        "--fg_coverage_value", type=float, default=0.25, help="fg coverage value"
    )
    parser.add_argument(
        "--sat_weight", type=float, default=1., help="weight of saturation loss of the generator"
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
        "--mixing", type=float, default=0.0, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training (checkpoints should be in the format of pytorch)",
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
        "--run_name", type=str, default="exp1", help="run name for wandb"
    )
    parser.add_argument(
        "--save_image", action="store_true", help="whether to save images as well when saving models"
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
    parser.add_argument(
        "--random_seed",
        type=int, default=0, help="Random Seed for reproducibility"
    )

    args = parser.parse_args()

    fix_seed(args.random_seed)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_z2 = torch.randn(args.n_sample, args.latent, device=device)

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    generator.eval()

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
 
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    min_res = {1024: 32, 512: 16, 256: 8}
    if args.reproduce_model:
        torch.manual_seed(0)
        alphanet_model = bg_extractor_repro(image_size = args.size, min_res = min_res[args.size]).to(device)
        ema_bg = bg_extractor_repro(image_size = args.size, min_res = min_res[args.size]).to(device)
    else: 
        alphanet_model = bg_extractor(image_size = args.size, min_res = min_res[args.size]).to(device)
        ema_bg = bg_extractor(image_size = args.size, min_res = min_res[args.size]).to(device)

    ema_bg.eval()
    accumulate(ema_bg, alphanet_model, 0)


    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    parm_ =  list(alphanet_model.parameters()) # 왜 굳이 list로 해서 넣어줄까..?
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
        
        try: # stylegan2 state_dict
            generator.load_state_dict(ckpt['g_ema'])
            g_ema.load_state_dict(ckpt['g_ema'])
        
        except: # state dict from labels4free
            generator.load_state_dict(ckpt['g'])
            g_ema.load_state_dict(ckpt['g'])
        
        print("Generator Loaded.")

        if args.use_disc:   
            discriminator.load_state_dict(ckpt['d'])
            print("Discriminator Loaded.")

        if args.pretrained_alphanet:
            alphanet_model.load_state_dict(ckpt['bg_extractor_ema'])
            ema_bg.load_state_dict(ckpt['bg_extractor_ema'])
            print("AlphaNet Loaded.")
        
        print("Model successfully loaded!")
       

    with torch.no_grad():
        mean_latent = g_ema.mean_latent(4096)



    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        alphanet_model = nn.parallel.DistributedDataParallel(
            alphanet_model,
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
            transforms.Resize((args.size, args.size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    print("Building Dataset...")
    if args.use_lmdb:
        dataset = MultiResolutionDataset(args.path, transform, args.size)
    else:
        dataset = TestDataset(args.path, transform)
    print("Dataset Built!")

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        pin_memory=True
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="labels4free", name=args.run_name, entity="yoojlee", config=vars(args))

    train(args, loader, generator, alphanet_model, discriminator, g_optim, d_optim, ema_bg, device, mean_latent)
