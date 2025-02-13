# Introduction to SMC

This repository accompanies the paper 'An introduction to Sequential Monte Carlo for Bayesian inference and model comparison -- with examples for psychology and behavioural science' (link will follow). The paper shows how to use Sequential Monte Carlo for inference in various models relevant for psychology and behavioural science.

All models are implement using the [bamojax](https://github.com/UncertaintyInComplexSystems/bamojax) Python library, which relies on [Distrax](https://github.com/google-deepmind/distrax) for probability distributions and [Blackjax](https://github.com/blackjax-devs/blackjax) for inference. 

## Installation
Although the tutorials should be self-explanatory (especially together with the paper), the installation of all required libraries and the right dependencies can be a bit tricky. We aim to make this more robust in the future, but for now we recommend the following steps:

```conda create --name introduction-to-smc
conda activate introduction-to-smc
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda install python=3.10.12 pip matplotlib pandas tqdm

git clone https://github.com:UncertaintyInComplexSystems/bamojax.git
pip install jaxtyping==0.2.34 distrax==0.1.5 blackjax==1.2.4

```

Within the code examples, you will need to update the `bamojax` path to the appropriate directory.

