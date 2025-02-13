import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int)
parser.add_argument('--kernel', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--num_chains', type=int, default=5)
parser.add_argument('--num_particles', type=int, default=1000)
parser.add_argument('--initial_num_mutations', type=int)
parser.add_argument('--num_doublings', type=int)
args = parser.parse_args()

seed = args.seed
kernel = args.kernel
num_chains = args.num_chains
num_particles = args.num_particles
initial_num_mutations = args.initial_num_mutations
num_doublings = args.num_doublings

SELECTED_DEVICE = f'{args.device}'
print(f'Setting CUDA visible devices to [{SELECTED_DEVICE}]')
os.environ['CUDA_VISIBLE_DEVICES'] = f'{SELECTED_DEVICE}'

import jax
jax.config.update("jax_enable_x64", True)  # Do we need this here? -> it seems we do for the LML computations (otherwise NaNs get introduced), but not for performance

import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import blackjax
import pandas as pd
import jax.scipy.special as jsp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import pyreadr as pr

import sys
import time
import requests

from blackjax import normal_random_walk, nuts

sys.path.append('/scratch/big/home/maxhin/Documents/Repos/bamojax/')

import bamojax
from bamojax.base import Node, Model
from bamojax.sampling import gibbs_sampler, smc_inference_loop, mcmc_sampler

print('Python version:       ', sys.version)
print('Jax version:          ', jax.__version__)
print('BlackJax version:     ', blackjax.__version__)
print('Distrax version:      ', dx.__version__)
print('BaMoJax version:      ', bamojax.__version__)
print('Jax default backend:  ', jax.default_backend())
print('Jax devices:          ', jax.devices())

def download_to_disk(url, filepath):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print('File downloaded successfully!')
    else:
        print(f'Failed to download the file. Status code: {response.status_code}')

#
data_busemeyer_url = 'https://osf.io/download/5vws6/'  # DataBusemeyerNoNA.rdata on https://osf.io/f9cq4/; contains IGT data
data_busemeyer_file = 'DataBusemeyerNoNA.rdata'

data_steingroever_url = 'https://osf.io/download/bmnsv/'  # contains Steingroever's importance sampling marginal likelihoods
data_steingroever_file = 'DataSteingroever.rdata'

lml_url = 'https://osf.io/download/txnbs/' # ind_LogMargLik.txt on https://osf.io/f9cq4/; contains Gronau's bridge sampling estmates
lml_file = 'ind_LogMargLik.txt'

download_to_disk(data_busemeyer_url, data_busemeyer_file)
download_to_disk(data_steingroever_url, data_steingroever_file)
download_to_disk(lml_url, lml_file)

data_file = pr.read_r('DataBusemeyerNoNA.rdata')
choices = jnp.asarray(data_file['choice'].to_numpy().astype(int)) - 1  # Python zero-indexing
losses = jnp.asarray(data_file['lo'].to_numpy())
wins = jnp.asarray(data_file['wi'].to_numpy())

N, T = choices.shape
K = 4

def hev_link_fn_scan(w, a, c, choices, wins, losses):
    """
    Return the full (N, T, K) array of logits,
    but do it with lax.scan instead of partial updates.

    This function was optimized from the previous implementation by ChatGPT-o1
    """
    c = 4*c - 2.0
    ev0 = jnp.zeros((N, K))
    row_inds = jnp.arange(N)
    init_logit = jnp.ones((N, K))  # shape (N, K) for time t=0

    def scan_step(carry, t):
        ev_ = carry
        current_utility = (1 - w) * wins[:, t] + w * losses[:, t]
        chosen_decks_ev = ev_.at[row_inds, choices[:, t]]
        ev_updated = ev_.at[row_inds, choices[:, t]].set(
            chosen_decks_ev.get() + a * (current_utility - chosen_decks_ev.get())
        )
        theta = (0.1 * (t + 1)) ** c
        logits_tplus1 = theta[:, None] * ev_updated
        return ev_updated, logits_tplus1
    
    #
    carry_init = ev0

    # Run the scan over t in [0..T-1]
    carry_final, logits_seq = jax.lax.scan(
        scan_step, 
        carry_init, 
        jnp.arange(T-1)
    )

    logits_full = jnp.concatenate([init_logit[None], logits_seq], axis=0)  
    logits_full = jnp.swapaxes(logits_full, 0, 1)  # shape => (N, T, K)

    return dict(logits=logits_full)

#

HEVModel = Model(f'Hierarchical expectance valence model')
mu_w_node = HEVModel.add_node('mu_w', dx.Normal(loc=0.0, scale=1.0))
mu_a_node = HEVModel.add_node('mu_a', dx.Normal(loc=0.0, scale=1.0))
mu_c_node = HEVModel.add_node('mu_c', dx.Normal(loc=0.0, scale=1.0))

sigma_w_node = HEVModel.add_node('sigma_w', dx.Uniform(low=0.0, high=1.5))
sigma_a_node = HEVModel.add_node('sigma_a', dx.Uniform(low=0.0, high=1.5))
sigma_c_node = HEVModel.add_node('sigma_c', dx.Uniform(low=0.0, high=1.5))

w_node = HEVModel.add_node('w', distribution=dx.Normal, parents=dict(loc=mu_w_node, scale=sigma_w_node), shape=(N, ), bijector=tfb.NormalCDF())
a_node = HEVModel.add_node('a', distribution=dx.Normal, parents=dict(loc=mu_a_node, scale=sigma_a_node), shape=(N, ), bijector=tfb.NormalCDF())
c_node = HEVModel.add_node('c', distribution=dx.Normal, parents=dict(loc=mu_c_node, scale=sigma_c_node), shape=(N, ), bijector=tfb.NormalCDF())

wins_node = HEVModel.add_node('wins', observations=wins)
loss_node = HEVModel.add_node('losses', observations=losses)

choice_node = HEVModel.add_node('choices', 
                                observations=choices, 
                                distribution=dx.Categorical, 
                                link_fn=hev_link_fn_scan, 
                                parents=dict(w=w_node, 
                                             a=a_node, 
                                             c=c_node, 
                                             choices=choices,
                                             wins=wins_node, 
                                             losses=loss_node))

def select_mutation_kernel(kernel):
    if kernel == 'MH':
        rmh_params = dict(sigma=0.01*jnp.eye(HEVModel.get_model_size()))
        rmh = mcmc_sampler(HEVModel, mcmc_kernel=normal_random_walk, mcmc_parameters=rmh_params)
        return rmh
    elif kernel == 'Gibbs':
        step_fns = dict(mu_w=normal_random_walk, 
                        mu_a=normal_random_walk, 
                        mu_c=normal_random_walk, 
                        w=normal_random_walk, 
                        a=normal_random_walk, 
                        c=normal_random_walk,
                        sigma_w=normal_random_walk,
                        sigma_a=normal_random_walk,
                        sigma_c=normal_random_walk)
        step_fn_params = dict(mu_w=dict(sigma=0.5), 
                            mu_a=dict(sigma=0.5), 
                            mu_c=dict(sigma=0.5), 
                            w=dict(sigma=0.02), 
                            a=dict(sigma=0.02), 
                            c=dict(sigma=0.02),
                            sigma_w=dict(sigma=0.2),
                            sigma_a=dict(sigma=0.2),
                            sigma_c=dict(sigma=0.2))

        gibbs = gibbs_sampler(HEVModel, step_fns=step_fns, step_fn_params=step_fn_params)
        return gibbs
    elif kernel == 'nuts':
        key = jrnd.PRNGKey(42)
        key, k_init, k_warmup = jrnd.split(key, 3)
        num_warmup = 500
        logdensity_fn = lambda state: HEVModel.loglikelihood_fn()(state) + HEVModel.logprior_fn()(state)
        warmup = blackjax.window_adaptation(nuts, logdensity_fn)
        (_, warm_parameters), _ = warmup.run(k_warmup, HEVModel.sample_prior(k_init), num_steps=num_warmup) 
        nuts_sampler = mcmc_sampler(HEVModel, mcmc_kernel=nuts, mcmc_parameters=warm_parameters)
        return nuts_sampler
    
#

mcmc_kernel = select_mutation_kernel(kernel)


key = jrnd.PRNGKey(seed)
stepsize = 0.01
num_mcmc_steps = 100

num_mcmc_steps = initial_num_mutations

for i in range(num_doublings):
    key, subkey = jrnd.split(key)
    start = time.time()
    print(f'HEV model; {kernel}-in-SMC, {num_particles} particles, {num_mcmc_steps} mutations')
    final_state, lml, num_adapt = smc_inference_loop(subkey, 
                                                     model=HEVModel, 
                                                     kernel=mcmc_kernel, 
                                                     num_particles=num_particles, 
                                                     num_mcmc_steps=num_mcmc_steps, 
                                                     num_chains=num_chains, 
                                                     store_diagnostics=False)
    stop = time.time()
    time_elapsed = stop - start
    time_elapsed_hours = time_elapsed / 3600
    if num_chains > 1:
        print(f'{kernel}-in-SMC, {num_particles} particles, {num_mcmc_steps} mutations, {jnp.mean(lml)} ({jnp.std(lml)}); completed in {time_elapsed_hours:0.2f} hours; {num_adapt} SMC cycles')
    else:
        print(f'{kernel}-in-SMC, {num_particles} particles, {num_mcmc_steps} mutations, {lml}; completed in {time_elapsed_hours:0.2f} hours; {num_adapt} SMC cycles')
    results = {}
    results['final_state'] = final_state
    results['lml'] = lml
    results['time'] = time_elapsed
    results['num_adapt'] = num_adapt
    jnp.save(f'hev_lml_{kernel}_particles={num_particles}_steps={num_mcmc_steps}_seed={seed}_chains={num_chains}.npy', results)

    num_mcmc_steps *= 2

