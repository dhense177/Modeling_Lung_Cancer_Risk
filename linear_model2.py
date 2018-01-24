import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform, forestplot
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, ElasticNet
import pickle, os, csv


















if __name__=='__main__':
    df_lung = pd.read_csv('lung_tri.csv')
    df_lung.index = df_lung['State_and_county']
    df = df_lung[['year','cancer_incidence','smoking','smoking_daily','pm25','ozone', 'radon_mean','Prob_low_radon','Prob_high_radon','Days PM2.5','Median AQI','Max AQI']]

    df = df[df.index.value_counts()>10]


    X = df[df.year==2010]
    #y = X.pop('cancer_incidence')
    y = df.cancer_incidence[df.year==2011]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 17)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #Basic Liner Regression
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    rmse_linear = np.sqrt(mean_squared_error(y_test, lm.predict(X_test)))

    #Lasso
    lasso = Lasso(alpha=0.15)
    lasso.fit(X_train, y_train)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    variables = np.array(list(zip(X.columns,lasso.coef_)))



    net = ElasticNet(alpha=0.15, l1_ratio=0.5, random_state=0)
    net.fit(X_train, y_train)

    rmse_net =  np.sqrt(mean_squared_error(y_test, net.predict(X_test)))
    #print(variables, rmse_net)

######################################

    #y = df[df.year==2010]['2011_incidence'].values


########Unpooled Model:Good below


    # df['State_and_county'] = df.index
    # grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
    # grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
    # grouped['State_and_county'] = grouped.index
    # df = pd.merge(df, grouped, how='left', on='State_and_county')
    #
    #
    # latest = pd.DataFrame(df[df.year==2011][['State_and_county','cancer_incidence_x']])
    # df['2011_incidence'] = pd.merge(df, latest, how='left', on='State_and_county')['cancer_incidence_x_y']
    #
    # df = df[df.year<2011]
    # df.index = df['State_and_county']
    # # X = df[df.year==2010]['cancer_incidence_y'].values
    # # y = df[df.year==2010]['2011_incidence'].values
    #
    # # y = df['smoking_daily'].values
    # # X = df['cancer_incidence_x']
    #
    #
    # scaler = preprocessing.StandardScaler()
    #
    # X = df[['smoking_daily','Days PM2.5']]
    # X = scaler.fit_transform(X)
    #
    # y = df['2011_incidence']
    #
    # counties = df['State_and_county'].unique()
    # county_lookup = dict(zip(counties, range(len(counties))))
    # county = df['State_and_county'].replace(county_lookup)
    #
    #
    # with Model() as unpooled_model:
    #
    #     beta0 = Normal('beta0', 0, sd=1e5, shape=len(counties))
    #     beta1 = Normal('beta1', 0, sd=1e5)
    #     beta2 = Normal('beta2', 0, sd=1e5)
    #     sigma = HalfCauchy('sigma', 5)
    #
    #     # theta = beta0[county] + beta1*X[:,0] + beta2*X[:,1]
    #     theta = beta0[county] + beta1[county]*X[:,0] + beta2[county]*X[:,1]
    #     y = Normal('y', theta, sd=sigma, observed=y)
    #
    # with unpooled_model:
    #     unpooled_trace = sample(1000, n_init=50000, tune=10000)
    #
    # unpooled_estimates = pd.Series(unpooled_trace['beta0'].mean(axis=0), index=counties)
    # unpooled_se = pd.Series(unpooled_trace['beta0'].std(axis=0), index=counties)
    #
    # plt.figure(figsize=(6,14))
    # forestplot(unpooled_trace, varnames=['beta0'], ylabels=counties)
    # plt.savefig('../unpooled_model16.png')
    #
    # plt.figure(figsize=(6,14))
    # order = unpooled_estimates.sort_values().index
    #
    # plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
    # for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    #     plt.plot([i,i], [m-se, m+se], 'b-')
    #
    # plt.ylabel('2011 Lung Cancer Estimates')
    # plt.xlabel('Ordered Counties')
    # plt.savefig('../unpooled_ordered4.png')

# ######################################
# #Partial pooling-NOT working
#
#     df['State_and_county'] = df.index
#     grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
#     grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
#     grouped['State_and_county'] = grouped.index
#     df = pd.merge(df, grouped, how='left', on='State_and_county')
#
#
#     latest = pd.DataFrame(df[df.year==2011][['State_and_county','cancer_incidence_x']])
#     df['2011_incidence'] = pd.merge(df, latest, how='left', on='State_and_county')['cancer_incidence_x_y']
#
#     counties = df['State_and_county'].unique()
#     county_lookup = dict(zip(counties, range(len(counties))))
#     county = df['State_and_county'].replace(county_lookup)
#     y = df['2011_incidence']
#     with Model() as partial_pooling:
#
#         # Priors
#         mu_a = Normal('mu_a', mu=0., sd=1e5)
#         sigma_a = HalfCauchy('sigma_a', 5)
#
#         # Random intercepts
#         a = Normal('a', mu=mu_a, sd=sigma_a, shape=len(counties))
#
#         # Model error
#         sigma_y = HalfCauchy('sigma_y',5)
#
#         # Expected value
#         y_hat = a[county]
#
#         # Data likelihood
#         y_like = Normal('y_like', mu=y_hat, sd=sigma_y, observed=y)
#
#
#     with partial_pooling:
#         partial_pooling_trace = sample(1000, n_init=50000, tune=1000)
#
#     plt.figure(figsize=(6,14))
#     forestplot(partial_pooling_trace, varnames=['a'], ylabels=counties)
#     plt.savefig('../partial_pooling.png')


#############################################################
#Varying intercept -- just smoking WORKSSSSSSS rmse: 9.25

    # df['State_and_county'] = df.index
    # grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
    # grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
    # grouped['State_and_county'] = grouped.index
    # df = pd.merge(df, grouped, how='left', on='State_and_county')
    #
    #
    # latest = pd.DataFrame(df[df.year==2011][['State_and_county','cancer_incidence_x']])
    # df['2011_incidence'] = pd.merge(df, latest, how='left', on='State_and_county')['cancer_incidence_x_y']
    #
    # counties = df['State_and_county'].unique()
    # county_lookup = dict(zip(counties, range(len(counties))))
    # county = df['State_and_county'].replace(county_lookup)
    #
    # scaler = preprocessing.StandardScaler()
    #
    # X = df[['smoking_daily','Days PM2.5']]
    # X = scaler.fit_transform(X)
    # #y = df['2011_incidence']
    # y = df['cancer_incidence_x']
    #
    # with Model() as varying_intercept:
    #
    #     # Priors
    #     mu_a = Normal('mu_a', mu=0., tau=0.0001)
    #     sigma_a = HalfCauchy('sigma_a', 5)
    #
    #
    #     # Random intercepts
    #     a = Normal('a', mu=mu_a, sd=sigma_a, shape=len(counties))
    #     # Common slope
    #     b = Normal('b', mu=0., sd=1e5)
    #
    #     # Model error
    #     sd_y = HalfCauchy('sd_y', 5)
    #
    #     # Expected value
    #     y_hat = a[county] + b * X[:,0]
    #
    #     # Data likelihood
    #     y_like = Normal('y_like', mu=y_hat, sd=sd_y, observed=y)
    #
    # with varying_intercept:
    #     varying_intercept_trace = sample(1000, n_init=50000, tune=1000)
    #
    # print(np.sqrt(mean_squared_error(y, varying_intercept_trace['a'].mean(axis=0)[county]+varying_intercept_trace['b'].mean(axis=0)*X[:,0])))
    #
    # # plt.figure(figsize=(6,14))
    # # forestplot(varying_intercept_trace, varnames=['a'], ylabels=counties)
    # # plt.savefig('../varying_intercept2.png')


############################################################

#############################################################
#Varying intercept -- smoking and air rmse: 9.25

    # df['State_and_county'] = df.index
    # grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
    # grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
    # grouped['State_and_county'] = grouped.index
    # df = pd.merge(df, grouped, how='left', on='State_and_county')
    #
    #
    # latest = pd.DataFrame(df[df.year==2011][['State_and_county','cancer_incidence_x']])
    # df['2011_incidence'] = pd.merge(df, latest, how='left', on='State_and_county')['cancer_incidence_x_y']
    #
    # counties = df['State_and_county'].unique()
    # county_lookup = dict(zip(counties, range(len(counties))))
    # county = df['State_and_county'].replace(county_lookup)
    #
    # scaler = preprocessing.StandardScaler()
    #
    # X = df[['smoking_daily','Days PM2.5']]
    # X = scaler.fit_transform(X)
    # #y = df['2011_incidence']
    # y = df['cancer_incidence_x']
    #
    # with Model() as varying_intercept:
    #
    #     # Priors
    #     mu_a = Normal('mu_a', mu=0., tau=0.0001)
    #     sigma_a = HalfCauchy('sigma_a', 5)
    #
    #
    #     # Random intercepts
    #     a = Normal('a', mu=mu_a, sd=sigma_a, shape=len(counties))
    #     # Common slope
    #     b = Normal('b', mu=0., sd=1e5)
    #
    #     c = Normal('c', mu=0., sd=1e5)
    #
    #     # Model error
    #     sd_y = HalfCauchy('sd_y', 5)
    #
    #     # Expected value
    #     y_hat = a[county] + b * X[:,0] + c * X[:,1]
    #
    #     # Data likelihood
    #     y_like = Normal('y_like', mu=y_hat, sd=sd_y, observed=y)
    #
    # with varying_intercept:
    #     varying_intercept_trace = sample(1000, n_init=50000, tune=1000)
    #
    # print(np.sqrt(mean_squared_error(y, varying_intercept_trace['a'].mean(axis=0)[county]+varying_intercept_trace['b'].mean(axis=0)*X[:,0]+varying_intercept_trace['c'].mean(axis=0)*X[:,1])))
    #
    # # plt.figure(figsize=(6,14))
    # # forestplot(varying_intercept_trace, varnames=['a'], ylabels=counties)
    # # plt.savefig('../varying_intercept2.png')


############################################################
    # #Unpooled Model - Works
    # rmse_list = []
    # for i in df.index[:5]:
    #     county_df = df.loc[i]
    #     county_df = county_df[['year','cancer_incidence_x','2011_incidence']]
    #     X = county_df['cancer_incidence_x']
    #     y = county_df['2011_incidence']
    #     lm = LinearRegression()
    #     lm.fit(X.reshape(-1,1), y)
    #     y_pred = lm.intercept_ + lm.coef_*X
    #     rmse = np.sqrt(mean_squared_error(y, y_pred))
    #     rmse_list.append(rmse)
    #
    #     plt.scatter([str(i) for i in county_df.year],county_df.pivot(columns='year',values='cancer_incidence_x') ,color='blue')
    #     X = np.array([i for i in county_df.year])
    #     Y = np.array(county_df['cancer_incidence_x'])
    #     fit = np.polyfit(X, Y, deg=1)
    #
    #     plt.plot(X, fit[0] * X + fit[1], color='red')
    #     plt.xlabel('Years', fontsize=10)
    #     plt.ylabel('Mean Lung Cancer Incidence per 100,000', fontsize=10)
    #
    #     plt.show()
    #
    # mean_rmse = np.mean(rmse_list)




# ####################
# #Hierarchical States- Very good!
# rmse=15.0
#
#     # scaler = preprocessing.StandardScaler()
#     # X = df[['mean_cancer_2009-2013','mean_smoking_2001-2005','log_radon','2014cancer_rate']]
#     # y = X.pop('2014cancer_rate')
#     # X = scaler.fit_transform(X)
#     traces_pickle = 'traces.pickle'
#     print("...loading pickle")
#     tmp = open(traces_pickle,'rb')
#     indiv_traces = pickle.load(tmp)
#     tmp.close()
#     #print(indiv_traces)
#
#     df['State_and_county'] = df.index
#     grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
#     grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
#     grouped['State_and_county'] = grouped.index
#     df = pd.merge(df, grouped, how='left', on='State_and_county')
#
#
#     latest = pd.DataFrame(df[df.year==2011][['State_and_county','cancer_incidence_x']])
#     df['2011_incidence'] = pd.merge(df, latest, how='left', on='State_and_county')['cancer_incidence_x_y']
#
#     df = df[df.year<2011]
#     df.index = df['State_and_county']
#     df['State'] = df['State_and_county'].str[-2:]
#
#     scaler = preprocessing.StandardScaler()
#
#     X = df[['cancer_incidence_x','smoking_daily', 'Days PM2.5','Max AQI']]
#     #X = df[['cancer_incidence_x','smoking_daily','Max AQI']]
#     X = scaler.fit_transform(X)
#
#     y = df['2011_incidence']
#
#
#     states = df.State.unique()
#     state_lookup = dict(zip(states, range(len(states))))
#     state = df.State.replace(state_lookup).values
#
#     counties = df['State_and_county'].unique()
#     county_lookup = dict(zip(counties, range(len(counties))))
#     county = df['State_and_county'].replace(county_lookup)
#
# ## States as groups
#
#     hier_traces = 'hier_traces.pickle'
#     if not os.path.isfile(hier_traces):
#         with pm.Model() as hierarchical_model:
#             # Hyperpriors
#             mu_a = pm.Normal('mu_alpha', mu=0., sd=1e5)
#             sigma_a = pm.HalfCauchy('sigma_alpha', beta=5)
#             mu_b = pm.Normal('mu_beta', mu=0., sd=1e5)
#             sigma_b = pm.HalfCauchy('sigma_beta', beta=5)
#             mu_c = pm.Normal('mu_beta2', mu=0., sd=1e5)
#             sigma_c = pm.HalfCauchy('sigma_beta2', beta=5)
#             # mu_d = pm.Normal('mu_beta3', mu=0., sd=1e5)
#             # sigma_d = pm.HalfCauchy('sigma_beta3', beta=5)
#
#             # Intercept for each county, distributed around group mean mu_a
#             a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(states))
#             # Intercept for each county, distributed around group mean mu_a
#             b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(states))
#             c = pm.Normal('beta2', mu=mu_c, sd=sigma_c, shape=len(states))
#             #d = pm.Normal('beta3', mu=mu_d, sd=sigma_d, shape=len(counties))
#             # Model error
#             eps = pm.HalfCauchy('eps', beta=1)
#
#             # Expected value
#             cancer_est = a[state] + b[state] * X[:,1] + c[state] * X[:,2]
#
#
#             # Data likelihood
#             y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=y)
#
#
#         with hierarchical_model:
#             hierarchical_trace = pm.sample(1000, n_init=150000, tune=100000)
#             #hierarchical_trace = pm.sample(draws=5000, tune=1000)
#         print(np.sqrt(mean_squared_error(y, hierarchical_trace['alpha'].mean(axis=0)[state]+hierarchical_trace['beta'].mean(axis=0)[state]*X[:,1]+hierarchical_trace['beta2'].mean(axis=0)[state]*X[:,2])))

#         hier_a = pd.Series(hierarchical_trace['alpha'].mean(axis=0))
#         hier_b = pd.Series(hierarchical_trace['beta'].mean(axis=0))
#         indv_a = [indiv_traces[c]['alpha'].mean() for c in counties]
#         indv_b = [indiv_traces[c]['beta'].mean() for c in counties]
#
#         # plt.figure(figsize=(6,10))
#         # forestplot(hierarchical_trace, varnames=['alpha'], ylabels='  '+states)
#         # plt.savefig('../hierarchical_forestplot2.png')
#
#         print("...saving pickle")
#         tmp = open(hier_traces,'wb')
#         pickle.dump(hierarchical_trace,tmp)
#         tmp.close()
#
#     else:
#         print("...loading pickle")
#         tmp = open(hier_traces,'rb')
#         hierarchical_trace = pickle.load(tmp)
#         tmp.close()
#
#
#         # plt.figure(figsize=(6,14))
#         # pm.traceplot(hierarchical_trace)
#         # plt.savefig('../hierarchical_traces3.png')
#         # plt.show()
#
#         plt.figure(figsize=(6,10))
#         forestplot(hierarchical_trace, varnames=['beta2'], ylabels='  '+states)
#         plt.savefig('../hierarchical_forestplot4.png')
#
#         # #Shrinkage?
#         # fig = plt.figure(figsize=(10, 10))
#         # ax = fig.add_subplot(111, xlabel='Intercept', ylabel='Cancer_2001-2010',
#         #                  title='Hierarchical vs. Non-hierarchical Bayes')
#         #
#         # ax.scatter(indv_a,indv_b, s=26, alpha=0.4, label = 'non-hierarchical')
#         # ax.scatter(hier_a,hier_b, c='red', s=26, alpha=0.4, label = 'hierarchical')
#         # for i in range(len(indv_b)):
#         #     ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i],
#         #          fc="k", ec="k", length_includes_head=True, alpha=0.4, head_width=.02)
#         # ax.legend()
#         # plt.savefig('../hierarchical_county.png')
#         #
#         #
#         #
#         # plt.figure(figsize=(6,14))
#         # pm.traceplot(hierarchical_trace)
#         # plt.savefig('../hierarchical_traces.png')
#         # plt.show()



    # print(pm.stats.waic(model=hierarchical_model, trace=hierarchical_trace))
    # print(pm.stats.loo(model=hierarchical_model, trace=hierarchical_trace))


#can I get posterior predictive check plot working?
    #
    # selection = ['CASS', 'CROW WING', 'FREEBORN']
    # fig, axis = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
    # axis = axis.ravel()
    # for i, c in enumerate(selection):
    #     c_data = data.ix[data.county == c]
    #     c_data = c_data.reset_index(drop = True)
    #     z = list(c_data['county_code'])[0]
    #
    #     xvals = np.linspace(-0.2, 1.2)
    #     for a_val, b_val in zip(indiv_traces[c]['alpha'][::10], indiv_traces[c]['beta'][::10]):
    #         axis[i].plot(xvals, a_val + b_val * xvals, 'b', alpha=.1)
    #     axis[i].plot(xvals, indiv_traces[c]['alpha'][::10].mean() + indiv_traces[c]['beta'][::10].mean() * xvals,
    #                  'b', alpha=1, lw=2., label='individual')
    #     for a_val, b_val in zip(hierarchical_trace['alpha'][::10][z], hierarchical_trace['beta'][::10][z]):
    #         axis[i].plot(xvals, a_val + b_val * xvals, 'g', alpha=.1)
    #     axis[i].plot(xvals, hierarchical_trace['alpha'][::10][z].mean() + hierarchical_trace['beta'][::10][z].mean() * xvals,
    #                  'g', alpha=1, lw=2., label='hierarchical')
    #     axis[i].scatter(c_data.floor + np.random.randn(len(c_data))*0.01, c_data.log_radon,
    #                     alpha=1, color='k', marker='.', s=80, label='original data')
    #     axis[i].set_xticks([0,1])
    #     axis[i].set_xticklabels(['basement', 'no basement'])
    #     axis[i].set_ylim(-1, 4)
    #     axis[i].set_title(c)
    #     if not i%3:
    #         axis[i].legend()
    #         axis[i].set_ylabel('log radon level')




####################################################
# ####################
# #Hierarchical Counties
#rmse: 8.9
    # scaler = preprocessing.StandardScaler()
    # X = df[['mean_cancer_2009-2013','mean_smoking_2001-2005','log_radon','2014cancer_rate']]
    # y = X.pop('2014cancer_rate')
    # X = scaler.fit_transform(X)
    traces_pickle = 'traces.pickle'
    print("...loading pickle")
    tmp = open(traces_pickle,'rb')
    indiv_traces = pickle.load(tmp)
    tmp.close()
    #print(indiv_traces)

    df['State_and_county'] = df.index
    grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
    grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
    grouped['State_and_county'] = grouped.index
    df = pd.merge(df, grouped, how='left', on='State_and_county')


    latest = pd.DataFrame(df[df.year==2011][['State_and_county','cancer_incidence_x']])
    df['2011_incidence'] = pd.merge(df, latest, how='left', on='State_and_county')['cancer_incidence_x_y']

    df = df[df.year<2011]
    df.index = df['State_and_county']
    df['State'] = df['State_and_county'].str[-2:]

    scaler = preprocessing.StandardScaler()

    X = df[['cancer_incidence_x','smoking_daily', 'Days PM2.5','Max AQI']]
    #X = df[['cancer_incidence_x','smoking_daily','Max AQI']]
    X = scaler.fit_transform(X)

    y = df['cancer_incidence_x']


    states = df.State.unique()
    state_lookup = dict(zip(states, range(len(states))))
    state = df.State.replace(state_lookup).values

    counties = df['State_and_county'].unique()
    county_lookup = dict(zip(counties, range(len(counties))))
    county = df['State_and_county'].replace(county_lookup)

## States as groups

    hier_traces2 = 'hier_traces2.pickle'
    if not os.path.isfile(hier_traces2):
        with pm.Model() as hierarchical_model:
            # Hyperpriors
            mu_a = pm.Normal('mu_alpha', mu=0., sd=1e5)
            sigma_a = pm.HalfCauchy('sigma_alpha', beta=5)
            mu_b = pm.Normal('mu_beta', mu=0., sd=1e5)
            sigma_b = pm.HalfCauchy('sigma_beta', beta=5)
            mu_c = pm.Normal('mu_beta2', mu=0., sd=1e5)
            sigma_c = pm.HalfCauchy('sigma_beta2', beta=5)
            # mu_d = pm.Normal('mu_beta3', mu=0., sd=1e5)
            # sigma_d = pm.HalfCauchy('sigma_beta3', beta=5)

            # Intercept for each county, distributed around group mean mu_a
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(counties))
            # Intercept for each county, distributed around group mean mu_a
            b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(counties))
            c = pm.Normal('beta2', mu=mu_c, sd=sigma_c, shape=len(counties))
            #d = pm.Normal('beta3', mu=mu_d, sd=sigma_d, shape=len(counties))
            # Model error
            eps = pm.HalfCauchy('eps', beta=1)

            # Expected value
            cancer_est = a[county] + b[county] * X[:,1] + c[county] * X[:,2]


            # Data likelihood
            y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=y)


        with hierarchical_model:
            hierarchical_trace = pm.sample(1000, n_init=150000, tune=50000)
            #hierarchical_trace = pm.sample(draws=5000, tune=1000)
        print(np.sqrt(mean_squared_error(y, hierarchical_trace['alpha'].mean(axis=0)[county]+hierarchical_trace['beta'].mean(axis=0)[county]*X[:,1]+hierarchical_trace['beta2'].mean(axis=0)[county]*X[:,2])))

#         hier_a = pd.Series(hierarchical_trace['alpha'].mean(axis=0))
#         hier_b = pd.Series(hierarchical_trace['beta'].mean(axis=0))
#         indv_a = [indiv_traces[c]['alpha'].mean() for c in counties]
#         indv_b = [indiv_traces[c]['beta'].mean() for c in counties]
#
        plt.figure(figsize=(6,10))
        forestplot(hierarchical_trace, varnames=['alpha'], ylabels='  '+counties)
        plt.savefig('../hierarchical_forestplot4.png')

        plt.figure(figsize=(6,14))
        pm.traceplot(hierarchical_trace)
        plt.savefig('../hierarchical_traces5.png')
        plt.show()
        print("...saving pickle")
        tmp = open(hier_traces2,'wb')
        pickle.dump(hierarchical_trace,tmp)
        tmp.close()

    else:
        print("...loading pickle")
        tmp = open(hier_traces2,'rb')
        hierarchical_trace = pickle.load(tmp)
        tmp.close()

        plt.figure(figsize=(6,10))
        forestplot(hierarchical_trace, varnames=['alpha'], ylabels='  '+counties)
        plt.savefig('../hierarchical_forestplot5.png')

        plt.figure(figsize=(6,14))
        pm.traceplot(hierarchical_trace)
        plt.savefig('../hierarchical_traces6.png')
        plt.show()
#
#         plt.figure(figsize=(6,10))
#         forestplot(hierarchical_trace, varnames=['beta2'], ylabels='  '+states)
#         plt.savefig('../hierarchical_forestplot4.png')
#
#         # #Shrinkage?
#         # fig = plt.figure(figsize=(10, 10))
#         # ax = fig.add_subplot(111, xlabel='Intercept', ylabel='Cancer_2001-2010',
#         #                  title='Hierarchical vs. Non-hierarchical Bayes')
#         #
#         # ax.scatter(indv_a,indv_b, s=26, alpha=0.4, label = 'non-hierarchical')
#         # ax.scatter(hier_a,hier_b, c='red', s=26, alpha=0.4, label = 'hierarchical')
#         # for i in range(len(indv_b)):
#         #     ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i],
#         #          fc="k", ec="k", length_includes_head=True, alpha=0.4, head_width=.02)
#         # ax.legend()
#         # plt.savefig('../hierarchical_county.png')
#         #
#         #
#         #
#         # plt.figure(figsize=(6,14))
#         # pm.traceplot(hierarchical_trace)
#         # plt.savefig('../hierarchical_traces.png')
#         # plt.show()



    # print(pm.stats.waic(model=hierarchical_model, trace=hierarchical_trace))
    # print(pm.stats.loo(model=hierarchical_model, trace=hierarchical_trace))











    # #Unpooled Model- pymc3
    # county_names = df.index.unique()
    #
    # indiv_traces = {}
    # cancer_estimates = []
    # for county_name in county_names:
    #
    #     county_df = df.loc[county_name]
    #     county_df = county_df[['year','cancer_incidence']]
    #
    #     cancer = county_df['cancer_incidence'].mean()
    #     cancer_pred = county_df[county_df.year==2011]['cancer_incidence'].values
    #
    #
    #     with pm.Model() as individual_model:
    #         # Intercept prior
    #         a = pm.Normal('alpha', mu=0, sd=1)
    #         # Slope prior
    #         b = pm.Normal('beta', mu=0, sd=1)
    #
    #         # Model error prior
    #         eps = pm.HalfCauchy('eps', beta=1)
    #
    #         # Linear model
    #         cancer_est = a + b * cancer
    #
    #         # Data likelihood
    #         y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=cancer_pred)
    #
    #         # Inference button (TM)!
    #         trace = pm.sample(progressbar=False)
    #
    #     cancer_estimates.append(cancer_est)
    #
    #     indiv_traces[county_name] = trace
    #
    #     #pm.traceplot(indiv_traces['Yolo County, CA'])
