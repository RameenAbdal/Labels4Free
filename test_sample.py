import argparse
import os
import torch
from torchvision import utils
from tqdm import tqdm
from model_new import Generator, bg_extractor_repro

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 Alpha Network test")

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
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--ckpt_bg_extractor",
        type=str,
        default=None,
        help="path to the checkpoints to test",
    )
    parser.add_argument(
        "--ckpt_generator",
        type=str,
        default=None,
        help="path to the checkpoints to test",
    )
    parser.add_argument(
        "--th",
        type=float,
        default=0.9,
        help="Threshold of the mask",
    )
  
  
  
    args = parser.parse_args()

    args.latent = 512 # latent dimension
    args.n_mlp = 8 # 아마 w code generate하는 거일 듯? 이 w code가 style code 같음.

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    generator.eval()
 
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0) # generator의 가중치를 애초에 ema 해버리는 듯?
    # 0 -> decay factor
    # 현재의 generator에만 의존하지 않기 위해 ema를 취하는 것 같다.

    min_res = {1024: 32, 512: 16, 256: 8} # input resolution에 따른 minimum output resolution을 매핑해주는 dictionary

    bg_extractor_ = bg_extractor_repro(image_size = args.size, min_res = min_res[args.size]).to(device)

    # checkpoint loading
    if args.ckpt_generator and args.ckpt_bg_extractor is not None:
        print("load bg extractor model:", args.ckpt_bg_extractor)
        print("load generator model:", args.ckpt_generator)

        ckpt = torch.load(args.ckpt_bg_extractor, map_location=lambda storage, loc: storage)
        ckpt_ = torch.load(args.ckpt_generator, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt_bg_extractor)

        except ValueError:
            pass

        bg_extractor_.load_state_dict(ckpt["bg_extractor_ema"])
        g_ema.load_state_dict(ckpt_['g_ema'])
        g_ema.eval()
        bg_extractor_.eval()
        
    
    with torch.no_grad():
        mean_latent = g_ema.mean_latent(4096) # truncation trick 정의하려고 mean_latent 구하는 것.
    
    with tqdm(range(args.n_sample)) as pbar:
     for i in pbar:

        sample_z = torch.randn(args.batch, args.latent, device=device)
        sample_z2 = torch.randn(args.batch, args.latent, device=device)


        sample, _ = g_ema([sample_z], truncation=0.5, truncation_latent=mean_latent, back = False) # back이 뭐지?
        sample_bg, __ = g_ema([sample_z2], truncation=0.5, truncation_latent=mean_latent, back = True)
              
        alpha_mask = bg_extractor_(_) # _: 저렇게도 들어가나 list가..?

        hard_mask = (alpha_mask > args.th).float() # binary mask로 변환

        image_new = sample * alpha_mask + (1 - alpha_mask) * sample_bg

        utils.save_image(
                        image_new,
                        f"test_sample/{str(i).zfill(6)}_composite.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        utils.save_image(
                        alpha_mask,
                        f"test_sample/{str(i).zfill(6)}_alpha_mask.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=False,
                    )

        utils.save_image(
                        sample,
                        f"test_sample/{str(i).zfill(6)}_original.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        utils.save_image(
                        sample_bg,
                        f"test_sample/{str(i).zfill(6)}_background.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

        utils.save_image(
                        hard_mask,
                        f"test_sample/{str(i).zfill(6)}_hard_mask.png",
                        nrow=int(args.batch ** 0.5),
                        normalize=False,
                        
                    )

