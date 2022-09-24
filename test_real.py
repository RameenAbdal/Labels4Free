"""
segmentation on real images

real image를 projection한 후에 학습한 latent_in을 다 load해줌. -> 어떻게 load할 거냐?
그건 model_new에 추가를 해줘야될 것 같다.
근데 애초에 latent_in은 그냥 learnable matrix로 생각하면 될 것 같은데..? 약간 vit에서 cls token마냥..

여러개 동시에 처리 어떻게 해줄지 생각해볼 것. (내일까지)
"""
import argparse
import os
import torch
from torchvision import utils
from model_new import *
from tqdm import tqdm
from PIL import Image

from dataset import PadTransform

MIN_RES = {1024:32, 512:16, 256:8}
N_MEAN_LATENT = 10000

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

# load latent_in & noise
def load_inputs(ckpt, idx):
    latent = ckpt[list(ckpt.keys())[idx]]['latent'] # 임시방편이라 수정해야됨
    noise = ckpt[list(ckpt.keys())[idx]]['noise']

    return latent, noise

def load_weights(path_g, path_bg, generator, alpha_net):
    """
    Load weights for generator and alpha networks.

    - Args
        path_g: path to a generator checkpoint
        path_bg: path to a background generator checkpoint
        generator: A generator object
        alpha_net: An alpha network object
    """
    g_ckpt = torch.load(path_g)
    bg_ckpt = torch.load(path_bg)

    generator.load_state_dict(g_ckpt['g_ema']) # in-place operation인지 확인
    alpha_net.load_state_dict(bg_ckpt['bg_extractor_ema'])

    return generator, alpha_net


def generate_mask(opt):
    """
    real test 경로가 있다고 가정했을 때, 해당 경로에서 하나씩 불러와서 evaluation하자.

    evaluation하려면 뭐가 필요하냐? 일단은 모델을 불러와야지.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transform = PadTransform(opt.size)

    # 일단 loop을 돌면서 latent와 path를 가져와야함.
    # load model
    g_ema = Generator(
        opt.size, opt.latent, opt.n_mlp, channel_multiplier=opt.channel_multiplier
    ).to(device)
    g_ema.eval()

    bg_extractor_ = bg_extractor_repro(image_size=opt.size, min_res=MIN_RES[opt.size]).to(device)
    bg_extractor_.eval()

    if opt.ckpt_generator and opt.ckpt_bg_extractor is not None:
        g_ema, bg_extractor_ = load_weights(opt.ckpt_generator, opt.ckpt_bg_extractor, g_ema, bg_extractor_)

    # latent_noise
    if opt.fuse_noise:
        with torch.no_grad():
            noise_sample = torch.randn(N_MEAN_LATENT, 512, device=device)
            latent_out = g_ema.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / N_MEAN_LATENT) ** 0.5

    # test 경로 generate
    ckpt = torch.load(opt.ckpt_path)
    n_test = len(ckpt.keys())

    pbar = tqdm(range(n_test))
    for i in pbar:
        # load latent code and noise
        with torch.no_grad():
            latent, noise = load_inputs(ckpt, i)
            latent = latent.unsqueeze(0)
            
            if opt.fuse_noise:
                noise_strength = latent_std * opt.noise * max(0,1-(i/n_test)/opt.noise_ramp) ** 2
                latent = latent_noise(latent, noise_strength.item())

            img_gen, _ = g_ema([latent], input_is_latent=True, noise=noise, back = False)

            alpha_mask = bg_extractor_(_)
            hard_mask = (alpha_mask > opt.th).float()
            
            alpha_mask = alpha_mask.detach().clone().cpu()
            image_org = transform(Image.open(list(ckpt.keys())[i])).unsqueeze(0)
            image_new = image_org * hard_mask.cpu()
            image_new = image_new.detach().clone().cpu()

            utils.save_image(
                            image_new,
                            f"{opt.save_dir}/{str(i).zfill(6)}_composite.png",
                            nrow=int(opt.batch ** 0.5),
                            normalize=True,
                            value_range=(-1,1)
                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation on Real Images")

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./test/proj_results/test_img.pt") # 어떤 식으로 input 받아오느냐에 따라 달라지긴 함.
    parser.add_argument("--save_dir", type=str, default="./test/final_results")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--ckpt_generator", type=str, default="/home/data/Labels4Free/checkpoint/stylegan2-car-config-f.pt")
    parser.add_argument("--ckpt_bg_extractor", type=str, default="/home/data/Labels4Free/checkpoint/bg_coverage_wt_15/000175.pt")
    parser.add_argument("--th", type=float, default=0.9)
    parser.add_argument("--fuse_noise", action="store_true")
    
    opt = parser.parse_args()
    sub_dir = opt.ckpt_path.split("/")[-1].replace(".pt", "")
    
    opt.save_dir = os.path.join(opt.save_dir, sub_dir)
    os.makedirs(opt.save_dir, exist_ok=True) # increment_path로 확장시켜두기
    generate_mask(opt)