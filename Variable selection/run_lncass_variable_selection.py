r"""
This file runs the LN-CASS prior for variable selection using different inference methods. Options are:

Metropolis-Hastings, Gibbs, and window-adapted NUTS, and each of these can be used as MCMC or MCMC-within-SMC.

The data are from the Nathan Klein Institute and are not shared here, but can be requested via https://fcon_1000.projects.nitrc.org/indi/enhanced/.



"""


import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mode', type=str)
parser.add_argument('--kernel', type=str)
parser.add_argument('--diagnostics', action='store_true')
parser.add_argument('--device', type=int)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--num_burn', type=int, default=1000)
parser.add_argument('--num_particles', type=int, default=1000)
parser.add_argument('--num_mcmc_steps', type=int, default=100)
parser.add_argument('--num_chains', type=int, default=4)
parser.add_argument('--num_thin', type=int, default=1000)


args = parser.parse_args()

SELECTED_DEVICE = f'{args.device}'
print(f'Setting CUDA visible devices to [{SELECTED_DEVICE}]')
os.environ['CUDA_VISIBLE_DEVICES'] = f'{SELECTED_DEVICE}'

import jax
jax.config.update("jax_enable_x64", True)  

import jax.random as jrnd
import jax.numpy as jnp
import jax.scipy.special as jsp
import distrax as dx
import blackjax
import pandas as pd
import time

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import sys

from blackjax import normal_random_walk, nuts
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction

sys.path.append('/scratch/big/home/maxhin/Documents/Repos/bamojax/')

import bamojax
from bamojax.base import Model
from bamojax.sampling import gibbs_sampler, smc_inference_loop, mcmc_inference_loop, mcmc_sampler

print('Python version:       ', sys.version)
print('Jax version:          ', jax.__version__)
print('BlackJax version:     ', blackjax.__version__)
print('Distrax version:      ', dx.__version__)
print('BaMoJax version:      ', bamojax.__version__)
print('Jax default backend:  ', jax.default_backend())
print('Jax devices:          ', jax.devices())

DATA_DIR = '/home/maxhin/Documents/Code/SMC tutorial/BLR/data/Request all'
RESULTS_DIR = '/home/maxhin/Documents/Code/SMC tutorial/BLR'
FIGURES_DIR = 'figures'


def read_data(add_intercept=False):
    filename = f'{DATA_DIR}/selected_variables_scaled.csv'
    df = pd.read_csv(filename, sep=',', header=[0]).drop(labels='ID', axis=1)
    data = jnp.asarray(df)
    X = data[:, 1:]        
    y = data[:, 0]
    labels = list(df.columns)
    labels.remove('DBDI_22')
    ix_age = labels.index('DEM_001')
    labels[ix_age] = 'Age'
    ix_gender = labels.index('DEM_002')
    labels[ix_gender] = 'Gender'   
    ix_nicotine = labels.index('FAGERADULT_9')
    labels[ix_nicotine] = 'Nicotine'   
    filename = f'{DATA_DIR}/DS_CODEBOOK.csv'
    df = pd.read_csv(filename, sep=',', header=[0])
    if add_intercept:
        X = jnp.column_stack((jnp.ones((X.shape[0], )), X))  # add intercept! -> note that it doesn't do anything...
        if labels != None:
            labels.insert(0, 'Intercept')
    return X, y, labels

#
def get_max_psrf(samples, verbose=True):
    max_psrf = 0.0
    for key in samples.keys():    
        R_i = potential_scale_reduction(samples[key], chain_axis=0, sample_axis=1)
        if verbose:
            print(f'PSRF for {key}:')
            print(R_i)
        max_R_i = jnp.nanmax(R_i)
        if max_R_i > max_psrf:
            max_psrf = max_R_i
    return max_psrf

#
def run_mcmc_until_convergence(key, model, mcmc_kernel):
    max_psrf = 999
    n_steps = num_samples
    n_burn = num_burn
    n_thin = num_thin
    times = []
    steps = []

    while max_psrf > 1.1:
        print(f'MCMC sampling with {n_steps} samples and burn-in')
        key, subkey = jrnd.split(key)
        start_iter = time.time()
        if store_diagnostics:
            states, info = mcmc_inference_loop(subkey, model=model, kernel=mcmc_kernel, num_samples=n_steps, num_burn=n_burn, num_chains=num_chains, num_thin=n_thin)
            if store_diagnostics:
                print('Acceptance rates')
                if kernel_name == 'gibbs':
                    for var in info.keys():
                        print(f'{var}: {jnp.mean(info[var].is_accepted):0.2f}')
                else:
                    print(f'{jnp.mean(info.is_accepted):0.2f}')
        else:
            states = mcmc_inference_loop(subkey, model=model, kernel=mcmc_kernel, num_samples=n_steps, num_burn=n_burn, num_chains=num_chains, num_thin=n_thin, store_diagnostics=False)
        max_psrf = get_max_psrf(states.position)
        print(f'Max(PSRF): {max_psrf}')
        stop_iter = time.time()
        elapsed = stop_iter - start_iter
        print(f'Done in {elapsed} seconds')
        times.append(elapsed)
        steps.append(n_steps)
        n_steps *= 2
        n_burn *= 2
        n_thin *= 2

    if store_diagnostics:
        return states.position, info, steps, times
    else:
        return states.position, steps, times

#
def run_smc_until_convergence(key, model, mcmc_kernel):
    max_psrf = 999
    num_mutations = num_mcmc_steps
    times = []
    mutations = []
    num_smc_iter = []
    lmls = []

    while max_psrf > 1.1:
        print(f'SMC sampling with {num_mutations} steps')
        key, subkey = jrnd.split(key)
        start_iter = time.time()
        if store_diagnostics:
            final_state, lml, n_iter, info = smc_inference_loop(key=subkey, 
                                                                model=model, 
                                                                kernel=mcmc_kernel, 
                                                                num_particles=num_particles, 
                                                                num_mcmc_steps=num_mutations, 
                                                                num_chains=num_chains,
                                                                store_diagnostics=True)
        else:
            final_state, lml, n_iter = smc_inference_loop(key=subkey, 
                                                                model=model, 
                                                                kernel=mcmc_kernel, 
                                                                num_particles=num_particles, 
                                                                num_mcmc_steps=num_mutations, 
                                                                num_chains=num_chains,
                                                                store_diagnostics=False)
        max_psrf = get_max_psrf(final_state.particles)
        print(f'Max(PSRF): {max_psrf}')
        stop_iter = time.time()
        elapsed = stop_iter - start_iter
        print(f'Done in {elapsed} seconds')
        times.append(elapsed)
        mutations.append(num_mutations)
        lmls.append(lml)
        num_smc_iter.append(n_iter)
        num_mutations *= 2
    
    if store_diagnostics:
        return final_state.particles, info, mutations, times, num_smc_iter, lmls
    return final_state.particles, mutations, times, num_smc_iter, lmls

#
def get_mcmc_kernel(kernel_name, model, p, key=None):
    if kernel_name == 'gibbs':
        step_fns = dict(beta=normal_random_walk, 
                        sigma=normal_random_walk,
                        lam=normal_random_walk, 
                        tau=normal_random_walk)
        step_fn_params = dict(beta=dict(sigma=0.001*jnp.eye(p)), 
                            lam=dict(sigma=0.01*jnp.eye(p)), 
                            sigma=dict(sigma=0.01), 
                            tau=dict(sigma=0.2))

        gibbs = gibbs_sampler(model, step_fns=step_fns, step_fn_params=step_fn_params)
        return gibbs, dict(step_fns=step_fns, step_fn_params=step_fn_params)
    elif kernel_name == 'mh':
        step_fn_params = dict(sigma=0.001*jnp.eye(2*p + 2))
        rmh = mcmc_sampler(model, mcmc_kernel=normal_random_walk, mcmc_parameters=step_fn_params)
        return rmh, step_fn_params
    elif kernel_name == 'nuts':
        print(f'Adapting NUTS')
        adapt_start = time.time()
        key, k_init, k_warmup = jrnd.split(key, 3)
        num_burn = 500
        logdensity_fn = lambda state: model.loglikelihood_fn()(state) + model.logprior_fn()(state)
        warmup = blackjax.window_adaptation(nuts, logdensity_fn)
        (_, warm_parameters), _ = warmup.run(k_warmup, model.sample_prior(k_init), num_steps=num_burn)  # technically this isn't the burn-in, but it's usable as such
        adapt_stop = time.time()
        print(f'Adaptation done in {adapt_stop - adapt_start} seconds')
        nuts_sampler = mcmc_sampler(model, mcmc_kernel=nuts, mcmc_parameters=warm_parameters)
        return nuts_sampler, warm_parameters
    else:
        raise NotImplementedError(f'Could not find setup for kernel {kernel_name}')
    
#

sampling_mode = args.mode 
store_diagnostics = args.diagnostics

print(f'Store diagnostics: {store_diagnostics}')
print('Bainter major depressive disorder dataset')

num_particles = args.num_particles
num_mcmc_steps = args.num_mcmc_steps
num_samples = args.num_samples
num_burn = args.num_burn
num_chains = args.num_chains
num_thin = args.num_thin
seed = args.seed
key = jrnd.PRNGKey(seed)


###########################      DATA      #######################

def least_squares(X, y):
    N, p = X.shape
    beta_mle = jnp.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_mle
    sigma2_mle = jnp.sum(residuals**2) / (N - p)
    return beta_mle, sigma2_mle

#

print('Loading data')
X, y, labels = read_data(add_intercept=True)
N, p = X.shape
print('X.shape:', X.shape)
beta_mle, sigma2_mle = least_squares(X, y)
print('beta MLE:', beta_mle)
print('sigma MLE:', jnp.sqrt(sigma2_mle))

###########################      MODEL      #######################

mu_v = jsp.logit(0.2)
sigma_v = 1.0 

def lin_reg_link_fn(beta, sigma, x):
    mu = jnp.dot(x, beta)
    return dict(loc=mu, scale=sigma)

#

model = Model('LNCASS')
sigma = model.add_node('sigma', distribution=dx.Transformed(dx.Normal(loc=0., scale=1.), tfb.Exp()))
tau = model.add_node('tau', distribution=dx.Transformed(dx.Normal(loc=0., scale=1.), tfb.Exp()))
lam = model.add_node('lam', dx.Transformed(dx.Normal(loc=mu_v*jnp.ones((p, )), scale=sigma_v*jnp.ones((p, ))), tfb.Sigmoid()))
beta = model.add_node('beta', distribution=dx.Normal, parents=dict(tau=tau, lam=lam), link_fn=lambda tau, lam: dict(loc=0.0, scale=(tau*lam)**2))
x_node = model.add_node('x', observations=X)
y_node = model.add_node('y', observations=y, distribution=dx.Normal, parents=dict(beta=beta, sigma=sigma, x=x_node), link_fn=lin_reg_link_fn)

model.print_gibbs()

###########################      INFERENCE      #######################

if sampling_mode == 'smc':
    print('SMC')
else:
    print('MCMC')

kernel_name = args.kernel
if kernel_name == 'nuts':
    key, subkey = jrnd.split(key)
    mcmc_kernel, mcmc_params = get_mcmc_kernel(kernel_name, model, p, subkey)
else:
    mcmc_kernel, mcmc_params = get_mcmc_kernel(kernel_name, model, p)

print(f'Number of chains:  {num_chains}')
print(f'Thinning by: {num_thin}')
print('Starting MCMC')

if sampling_mode == 'smc':
    if store_diagnostics:
        samples, info, mutations, times, num_smc_iter, lmls = run_smc_until_convergence(key, model=model, mcmc_kernel=mcmc_kernel)
    else:
        samples, mutations, times, num_smc_iter, lmls = run_smc_until_convergence(key, model=model, mcmc_kernel=mcmc_kernel)
    print(f'MCMC-in-SMC converged in {mutations[-1]} mutation steps for {num_smc_iter[-1]} SMC cycles, taking {times[-1]} seconds')
    print(f'Cumulative time: {jnp.sum(jnp.asarray(times))}')
else:
    if store_diagnostics:
        samples, info, steps, times = run_mcmc_until_convergence(key, model=model, mcmc_kernel=mcmc_kernel)
    else:
        samples, steps, times = run_mcmc_until_convergence(key, model=model, mcmc_kernel=mcmc_kernel)

    print(f'MCMC converged in {steps[-1]} samples, taking {times[-1]} seconds')
    print(f'Cumulative time: {jnp.sum(jnp.asarray(times))} seconds')


print('beta posterior expectation:')
print(jnp.mean(samples['beta'], axis=jnp.array([0, 1])))
print('beta.shape:', samples['beta'].shape)

###########################      EVALUATION      #######################


print('Sigma:', jnp.mean(samples['sigma'], axis=jnp.array([1])))

if store_diagnostics:
    print('Acceptance rates')
    if sampling_mode == 'smc':
        for var in info.update_info.keys():
            print(f'{var}: {jnp.mean(info.update_info[var].is_accepted):0.2f}')
    else:
        for var in info.keys():
            print(f'{var}: {jnp.mean(info[var].is_accepted):0.2f}')

print('Convergence assessment')

for key in samples.keys():
    ESS = effective_sample_size(samples[key], chain_axis=0, sample_axis=1)
    print(f'{key} ESS: ')
    print(ESS)
    R = potential_scale_reduction(samples[key], chain_axis=0, sample_axis=1)
    print(f'{key} PSRF: ')
    print(R)

###########################      STORE RESULTS      #######################

if sampling_mode == 'smc':
    result = dict(states=samples, mutations=mutations, times=times, num_smc_iter=num_smc_iter, lmls=lmls, mcmc_params=mcmc_params)
    if store_diagnostics:
        result['info'] = info
    jnp.save(f'Results/{model.name}_{kernel_name}_SMC_bainter_num_particles={num_particles}_num_mcmc_steps={num_mcmc_steps}_num_chains={num_chains}_seed={seed}.npy', result)
else:   
    result = dict(states=samples, times=times, mcmc_params=mcmc_params, steps=steps) 
    if store_diagnostics:
        result['info'] = info
    jnp.save(f'Results/{model.name}_{kernel_name}_MCMC_bainter_num_samples={num_samples}_num_burn={num_burn}_num_chains={num_chains}_seed={seed}.npy', result)