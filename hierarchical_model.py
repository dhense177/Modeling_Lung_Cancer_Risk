import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform, forestplot, traceplot, plot_posterior












if __name__=='__main__':
    #os.remove('hierarchical_model.py') if os.path.exists('hierarchical_model.py') else None
    df = pd.read_csv('lung_dataframe_overall3.csv',converters={'Combined': lambda x: str(x),'State-county recode_x': lambda x: str(x)})

    #df = pd.read_csv('final_metrics.csv')
    df_ky = df[df.State=='KY']

    print(df_ky.shape)
    # counties = df.County.unique()
    # county_lookup = dict(zip(counties, range(len(counties))))
    # county = df.County.replace(county_lookup).values


# ## All U.S. counties as group
#     with pm.Model() as hierarchical_model:
#         # Hyperpriors
#         mu_a = pm.Normal('mu_alpha', mu=0., sd=1)
#         sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
#         mu_b = pm.Normal('mu_beta', mu=0., sd=1)
#         sigma_b = pm.HalfCauchy('sigma_beta', beta=1)
#
#         # Intercept for each county, distributed around group mean mu_a
#         a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(counties))
#         # Intercept for each county, distributed around group mean mu_a
#         b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(counties))
#
#         # Model error
#         eps = pm.HalfCauchy('eps', beta=1)
#
#         # Expected value
#         radon_est = a[state] + b[state] * df['cancer_mean']
#
#         # Data likelihood
#         y_like = pm.Normal('y_like', mu=radon_est, sd=eps, observed=df['2014cancer_rate'])
#
#
#     with hierarchical_model:
#         hierarchical_trace = pm.sample(1000, n_init=50000, tune=1000)
#
#     plt.figure(figsize=(6,14))
#     pm.traceplot(hierarchical_trace)
#     plt.savefig('../hierarchical_intercept.png')
#     plt.show()




# State groups analysis

    # df['State'] = df1.State

    # states = df.State.unique()
    # county_lookup = dict(zip(states, range(len(states))))
    # state = df.State.replace(county_lookup).values


    # with Model() as partial_pooling:
    #
    #     # Priors
    #     mu_a = Normal('mu_a', mu=0., sd=25)
    #     sigma_a = HalfCauchy('sigma_a', 1)
    #
    #     # Random intercepts
    #     a = Normal('a', mu=mu_a, sd=sigma_a, shape=len(states))
    #
    #     # Model error
    #     sigma_y = HalfCauchy('sigma_y',1)
    #
    #     # Expected value
    #     y_hat = a[state]
    #
    #     # Data likelihood
    #     y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=df['2014cancer_rate'].values)
    #
    # with partial_pooling:
    #     partial_pooling_trace = sample(1000, n_init=50000, tune=1000)
    #
    # sample_trace = partial_pooling_trace['a']
    #
    # fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)
    # samples, counties = sample_trace.shape
    # jitter = np.random.normal(scale=0.1, size=counties)
    #
    # n_county = df.groupby('State')['County'].count()
    # unpooled_means = df.groupby('State')['2014cancer_rate'].mean()
    # unpooled_sd = df.groupby('State')['2014cancer_rate'].std()
    # unpooled = pd.DataFrame({'n':n_county, 'm':unpooled_means, 'sd':unpooled_sd})
    # unpooled['se'] = unpooled.sd/np.sqrt(unpooled.n)
    #
    # axes[0].plot(unpooled.n + jitter, unpooled.m, 'b.')
    # for j, row in zip(jitter, unpooled.iterrows()):4) State Health Policy Research Dataset (SHEPRD)
    #     name, dat = row
    #     axes[0].plot([dat.n+j,dat.n+j], [dat.m-dat.se, dat.m+dat.se], 'b-')
    # #axes[0].set_xscale('log')
    # axes[0].hlines(sample_trace.mean(), 0.9, 100, linestyles='--')
    #
    #
    # samples, counties = sample_trace.shape
    # means = sample_trace.mean(axis=0)
    # sd = sample_trace.std(axis=0)
    # axes[1].scatter(n_county.values + jitter, means)
    # #axes[1].set_xscale('log')
    # #axes[1].set_xlim(1,100)
    # #axes[1].set_ylim(0, 3)
    # axes[1].hlines(sample_trace.mean(), 0.9, 100, linestyles='--')
    # for j,n,m,s in zip(jitter, n_county.values, means, sd):
    #     axes[1].plot([n+j]*2, [m-s, m+s], 'b-')
    #     plt.savefig('../partial_pooling.png')
    #     plt.show()

    # with Model() as varying_intercept:
    #
    #     # Priors
    #     mu_a = Normal('mu_a', mu=0., tau=0.0001)
    #     sigma_a = HalfCauchy('sigma_a', 5)
    #
    #
    #     # Random intercepts
    #     a = Normal('a', mu=mu_a, sd=sigma_a, shape=len(states))
    #     # Common slope
    #     b = Normal('b', mu=0., sd=25)
    #
    #     # Model error
    #     sd_y = HalfCauchy('sd_y', 5)
    #
    #     # Expected value
    #     y_hat = a[state] + b * df['cancer_mean']
    #
    #     # Data likelihood
    #     y_like = Normal('y_like', mu=y_hat, sd=sd_y, observed=df['2014cancer_rate'])
    #
    #
    # with varying_intercept:
    #     varying_intercept_trace = sample(1000, n_init=50000, tune=1000)
    #
    # plt.figure(figsize=(6,14))
    # forestplot(varying_intercept_trace, varnames=['a'])
    #
    # plt.savefig('../varying_intercept.png')
    # plt.show()


## States as groups
    # with pm.Model() as hierarchical_model:
    #     # Hyperpriors
    #     mu_a = pm.Normal('mu_alpha', mu=0., sd=1)
    #     sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
    #     mu_b = pm.Normal('mu_beta', mu=0., sd=1)
    #     sigma_b = pm.HalfCauchy('sigma_beta', beta=1)
    #
    #     # Intercept for each county, distributed around group mean mu_a
    #     a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(states))
    #     # Intercept for each county, distributed around group mean mu_a
    #     b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(states))
    #
    #     # Model error
    #     eps = pm.HalfCauchy('eps', beta=1)
    #
    #     # Expected value
    #     radon_est = a[state] + b[state] * df['cancer_mean']
    #
    #     # Data likelihood
    #     y_like = pm.Normal('y_like', mu=radon_est, sd=eps, observed=df['2014cancer_rate'])
    #
    #
    # with hierarchical_model:
    #     hierarchical_trace = pm.sample(1000, n_init=50000, tune=1000)
    #
    # plt.figure(figsize=(6,14))
    # pm.traceplot(hierarchical_trace)
    # plt.savefig('../hierarchical_intercept.png')
    # plt.show()
