# MiniDiffusion

This motivation for this project is to create a diffusion model for image generation with significant compute constraints. The model will focus on images with low feature density in order to allow for smaller parameter models and meaningful results.

## Reference Literature

DDPMs: https://arxiv.org/pdf/2006.11239

Adversarial Guiding for Learning Manifolds: https://arxiv.org/pdf/2402.17563

## What is Diffusion?

I understand diffusion best when I think about it in terms of the forward and reverse processes. As an aside I will avoid using too much math in this explanation although understanding the underlying statistics that prove how diffusion models are structured is essential to truly understanding the process. In the forward process we consider a Markov chain in which each successive time step adds some Gaussian noise to our baseline image. What we are really trying to do with this process is take our initial image and iteratively transform it into pure noise. While this looks fun in and of itself the reason for doing this is so that we can learn a reverse process. In this reverse process we want to "denoise" the image and recover our original sample (this is certainly reminiscent of autoencoders). What our model will really do however is learn to predict the noise required to obtain some noisy sample from our forward Markov chain given our original sample and the time t. This will, if all goes as planned, allow us to do "image-to-image" translation and also (hopefully) build some cool image generation models.

## Authors

* **Dhruv Pendharkar**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments



