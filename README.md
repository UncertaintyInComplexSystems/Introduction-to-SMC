# Introduction to SMC

This repository accompanies the paper 'An introduction to Sequential Monte Carlo for Bayesian inference and model comparison -- with examples for psychology and behavioural science' (link will follow). The paper shows how to use Sequential Monte Carlo for inference in various models relevant for psychology and behavioural science.

All models are implement using the [Blackjax](https://blackjax-devs.github.io/blackjax/) Python library, as well as our [Bayesian models code](https://github.com/UncertaintyInComplexSystems/bayesianmodels) for easier use. 

## Installation
Although the tutorials should be self-explanatory (especially together with the paper), the installation of all required libraries and the right dependencies can be a bit tricky. We aim to make this more robust in the future, but for now we recommend the following steps:

```conda create --name introduction-to-smc
conda activate introduction-to-smc
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda install python=3.10.12 pip matplotlib pandas tqdm

pip install --force-reinstall chex==0.1.5 optax==0.1.7 distrax==0.1.3 jax==0.4.6 jaxlib==0.4.6+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html numpy==1.22.4 scipy==1.10.0 tensorflow-probability==0.19.0 git+https://github.com/JaxGaussianProcesses/JaxKern.git jaxopt==0.5.5 fastprogress==0.2.0 git+https://github.com/Hesterhuijsdens/blackjax.git`
mkdir bayesianmodels
cd bayesianmodels
git clone git@github.com:UncertaintyInComplexSystems/bayesianmodels.git
mkdir ../intro-to-smc
cd ../intro-to-smc
git clone https://github.com/UncertaintyInComplexSystems/Introduction-to-SMC

```

Within the code examples, you will need to update the `PATH_TO_UICSMODELS` variable to point to the appropriate directory.

