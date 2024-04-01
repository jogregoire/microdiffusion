<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

# Microdiffusion: Diffusion Models 101

Microdiffusion implements the basic of images diffusion. It implements both the training and sampling.In the example, it uses Sprites by ElvGames, [FrootsnVeggies](https://zrghr.itch.io/froots-and-veggies-culinary-pixels) and [kyrise](https://kyrise.itch.io/kyrises-free-16x16-rpg-icon-pack), that is packaged as Numpy muti-dimensional arrays.

The DDPM and DDIM sampling algorithms are based on the code and labs presented in Deeplearning AI's [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) by Sharon Zhou. The code is available from Ryota Kawamura's [Github repo](https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work). It is based from https://github.com/cloneofsimo/minDiffusion

This repository shows:

## an implementation of the first diffusion model (DDPM): [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) 

### Sampling

1. $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

2. for $t=T, \ldots, 1$ do

3. $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ if $t>1$, else $\mathbf{z}=\mathbf{0}$

4. $\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{z}_\theta\left(\mathbf{x}_t, t\right)\right)+\sigma_t \mathbf{z}$, noise added back is $\sigma_t \times z = \sqrt{\beta_t} \times z$, predicted noise is $\mathbf{z}_\theta$

5. end for

6. return $x_0$

## how to implement the faster DDIM model: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

## Test with different noise schedulers: Linear, Quadratic, Sigmoid and Cosine.

Linear scheduler

![linear noise scheduler](docs/linear_noise_scheduler.png)

Improments to the original DDPM paper include a different learning rate scheduler. Noise from the linear scheduler is added too fast, which make it harder to learn the reverse process. The cosine scheduler adds noise slower

![noise schedulers](docs/noise_schedulers.png)

noise scheduler: cosine (https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869#0caf)

## A simple example of what can be done using the awesome [einops library](https://github.com/arogozhnikov/einops) for tensor operations.

## A simple data augmentation technique for traning using [Torchvision's transforms](https://pytorch.org/vision/master/generated/torchvision.transforms.RandomHorizontalFlip.html)

## The Unet implementation is from the course, I found some nice educational material 

## how to get GPU Performance using [Pytorch API](https://pytorch.org/docs/master/generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats) and [Pynvml](https://pypi.org/project/pynvml/) libraries.

![GPU perf](docs/training_gpu_perf.png)

## how to build an web interface for the sampling using [Gradio](https://www.gradio.app).

![Web interface](docs/gradio.png)

# Next steps:

- Classifier-Free Guidance

- Re-implement the functionnalities using Hugging Face's Diffuser library for DDPM, DDIM and Unet2D.
