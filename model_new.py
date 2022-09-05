import math
import random
import functools
import operator
from typing import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
import math


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8) # torch.rsqrt -> 1/sqrt 리턴 (element-wise operation)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32) # 이 k가 뭔데..?

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style, mod = False):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, mod = False):
        out = self.conv(input, style, mod )
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.res_idx = {1024: 5, 512 : 3 , 256 : 1}
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.actual_tensor = self.conv1.conv.modulation.weight[0, :]
        self.actual_tensor2 = self.conv1.conv.modulation.weight[148, :]

        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        back = False,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, 18, 1)

        if back and self.size == 512:
            out = self.input(latent)  * 0 
            out = self.conv1(out, latent[:, 0], noise=noise[0]) 
        elif back and self.size != 512:
            out = self.input(latent) 
            out = self.conv1(out, latent[:, 0], noise=noise[0]) * 0 
        else : 
            out = self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])
            
        output_lis = []

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):

            out = conv1(out, latent[:, i], noise=noise1)

            if i >= self.res_idx[self.size] :
            
                output_lis.append(out)

            out = conv2(out, latent[:, i + 1], noise=noise2)

            if i >= self.res_idx[self.size] :

                output_lis.append(out)

            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, output_lis, latent
        else:  
            return image, output_lis




class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out



class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out




class MyConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

      

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input):
        batch, in_channel, height, width = input.shape

     

        weight = self.scale * self.weight
        

        weight = weight.view( self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(batch, in_channel, height, width)
            weight = weight.view(
                1, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                 in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=1)
         
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(batch,  in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=1)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class bg_extractor_repro(nn.Module):
    def __init__(self,  image_size = 1024, min_res = 32):
        super().__init__()

        self.image_size = image_size
        self.min_res = min_res
        self.output_channels = 3
        assert self.image_size == 1024 or 512 or 256 , 'Resolution error'

        if self.image_size == 1024:
             self.tensor_resolutions = [ 512, 512, 512, 512, 256, 256,  128, 128, 64, 64, 32, 32]

        elif self.image_size == 512:
            self.tensor_resolutions = [ 512,  512,  512, 512,  512,  512,  256, 256, 128, 128, 64, 64 ]

        else:
             self.tensor_resolutions = [ 512, 512,  512,  512, 512, 512,  512,  512,  256, 256, 128, 128]

        self.image_resolutions = [ 16, 32, 64, 128, 256, 512, 1024]
        

        self.upsample_fns = [nn.Upsample(size=(res, res), mode='bicubic', align_corners=True) for res in self.image_resolutions if res >= min_res*2]
        self.upsample_fns = nn.ModuleList(self.upsample_fns)
    
        self.conv1 = nn.Conv2d(self.tensor_resolutions[0], self.output_channels, 1, stride=1)
        self.conv2 = nn.Conv2d(self.tensor_resolutions[1], self.output_channels, 1, stride=1)
        self.conv3 = nn.Conv2d(self.tensor_resolutions[2], self.output_channels, 1, stride=1)
        self.conv4 = nn.Conv2d(self.tensor_resolutions[3], self.output_channels, 1, stride=1)
        self.conv5 = nn.Conv2d(self.tensor_resolutions[4], self.output_channels, 1, stride=1)
        self.conv6 = nn.Conv2d(self.tensor_resolutions[5], self.output_channels, 1, stride=1)
        self.conv_next= nn.Conv2d(self.output_channels*4, self.output_channels*2, 1, stride=1)
        self.conv_next2 = nn.Conv2d(self.output_channels*4, self.output_channels*2, 1, stride=1)
        self.conv_next3 = nn.Conv2d(self.output_channels*4, self.output_channels*2, 1, stride=1)
        self.conv_next4 = nn.Conv2d(self.output_channels*4, self.output_channels*2, 1, stride=1)
        self.conv7 = nn.Conv2d(self.tensor_resolutions[6], self.output_channels, 1, stride=1)
        self.conv8 = nn.Conv2d(self.tensor_resolutions[7], self.output_channels, 1, stride=1)
        self.conv9 = nn.Conv2d(self.tensor_resolutions[8], self.output_channels, 1, stride=1)
        self.conv10 = nn.Conv2d(self.tensor_resolutions[9], self.output_channels, 1, stride=1)
        self.conv11 = nn.Conv2d(self.tensor_resolutions[10], self.output_channels, 1, stride=1)
        self.conv12 = nn.Conv2d(self.tensor_resolutions[11], self.output_channels, 1, stride=1)
        self.conv_n = nn.Conv2d(self.output_channels*2, 1, 1, stride=1)
 

        self.sigmoid =  nn.Sigmoid()
        self.leakr = nn.LeakyReLU(0.2)

    def forward(self, input):

        out1 = self.upsample_fns[0](torch.cat([self.conv1(input[0]), self.conv2(input[1])], axis = 1))
        out2 = (torch.cat([self.conv3(input[2]), self.conv4(input[3])], axis = 1))

        out3 = self.upsample_fns[1](self.leakr(self.conv_next(torch.cat([out2, out1], axis = 1))))
        out4 = (torch.cat([ self.conv5(input[4]), self.conv6(input[5])], axis=1))

        out5 = self.upsample_fns[2](self.leakr(self.conv_next2(torch.cat([ out4, out3], axis=1))))
        out6 = (torch.cat([self.conv7(input[6]), self.conv8(input[7])], axis=1))

        out7 = self.upsample_fns[3](self.leakr(self.conv_next3(torch.cat([out6, out5], axis=1))))
        out8 = (torch.cat([self.conv9(input[8]), self.conv10(input[9])], axis=1))

        out9 = self.upsample_fns[4](self.leakr(self.conv_next4(torch.cat([out8,out7], axis=1))))
        out10 = (torch.cat([self.conv11(input[10]), self.conv12(input[11])], axis=1))

        out_final =  self.sigmoid(self.conv_n( out9 + out10))


        return out_final


class bg_extractor(nn.Module):

    def __init__(self, image_size = 1024, min_res = 32):
        super().__init__()
        self.kernel_size = 1
        self.image_size = image_size
        self.min_res = min_res
        self.image_resolutions = [ 16, 32, 64, 128, 256, 512, 1024]

        assert self.image_size == 1024 or 512 or 256 , 'Resolution error'

        if self.image_size == 1024:
            self.tensor_resolutions = [ 512, 512, 256,  128,  64, 32]

        elif self.image_size == 512:
            self.tensor_resolutions = [ 512,  512,  512,  256,  128,  64]

        else:
            self.tensor_resolutions = [ 512, 512,  512,  512,  256,  128]

        self.upsample_fns = [nn.Upsample(size=(res, res), mode='bicubic', align_corners=True) for res in self.image_resolutions if res >= min_res*2]
        self.upsample_fns = nn.ModuleList(self.upsample_fns)
        
        self.conv_fns0 = [nn.Conv2d(res, 3, 1, stride=1) for res in self.tensor_resolutions]
        self.conv_fns0 = nn.ModuleList(self.conv_fns0)

        self.conv_fns1 = [nn.Conv2d(res, 3, 1, stride=1) for res in self.tensor_resolutions]
        self.conv_fns1 = nn.ModuleList(self.conv_fns1)

        self.conv_merge_fns = [nn.Conv2d(12, 6, self.kernel_size, stride=1) for res in range(4)]
        self.conv_merge_fns = nn.ModuleList(self.conv_merge_fns)

        

        self.conv_n = nn.Conv2d(6, 1, 1, stride=1)
        self.sigmoid =  nn.Sigmoid()
        self.leakr = nn.LeakyReLU(0.2)
      

    def forward(self, input):

        
        out_0 = self.upsample_fns[0](torch.cat([self.conv_fns0[0](input[0]), self.conv_fns1[0](input[1])], axis = 1))
        count = 2
        for conv1, conv2, conv_merge, res in zip(self.conv_fns0[1:], self.conv_fns1[1:], self.conv_merge_fns,  self.upsample_fns[1:]):
              
     
                out_1 = (torch.cat([conv1(input[count ]), conv2(input[count + 1])], axis = 1))
               
                out_0 = res(self.leakr(conv_merge(torch.cat([out_1, out_0], axis = 1))))
                count += 2

        out_1 = (torch.cat([self.conv_fns0[-1](input[count ]), self.conv_fns1[-1](input[count + 1])], axis = 1))
           

            

        out_final = self.sigmoid(self.conv_n( out_1 + out_0))


        return out_final