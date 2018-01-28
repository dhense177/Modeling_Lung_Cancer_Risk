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
    df = df_lung[['cancer_incidence','smoking','smoking_daily','pm25','ozone', 'radon_mean','Prob_low_radon','Prob_high_radon','Days PM2.5','Median AQI','Max AQI','ON-SITE_RELEASE_TOTAL']]

    df = df[df.index.value_counts()>10]

    df_pooled = df.copy()

    y = df_pooled.pop('cancer_incidence')
    X = df_pooled.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 17)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #Pooled Liner Regression
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    rmse_linear = np.sqrt(mean_squared_error(y_test, lm.predict(X_test)))

    #Lasso
    lasso = Lasso()

    #Gridsearch lasso parameters

    # parameters = {
    #      'alpha':[.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1]
    # }
    #
    # lassosearch = GridSearchCV(lasso, parameters, cv=10, scoring='neg_mean_squared_error')
    #
    # model = lassosearch.fit(X_train, y_train)
    #
    # print('best params')
    # print(model.best_params_)



    lasso.fit(X_train, y_train)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    variables = np.array(list(zip(X.columns,lasso.coef_)))


    net = ElasticNet(alpha=0.1, l1_ratio=1)

    #Gridsearch ElasticNet parameters

    # parameters = {
    #      'alpha':[0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1],
    #      'l1_ratio':[0.01,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1]
    # }
    #
    # netsearch = GridSearchCV(net, parameters, cv=10, scoring='neg_mean_squared_error')
    #
    # model = netsearch.fit(X_train, y_train)
    #
    # print('best params')
    # print(model.best_params_)

    net.fit(X_train, y_train)

    rmse_net =  np.sqrt(mean_squared_error(y_test, net.predict(X_test)))

##########################################################################
### Unpooled Linear Model


    df = df_lung[['year','cancer_incidence','smoking','smoking_daily','pm25','ozone', 'radon_mean','Prob_low_radon','Prob_high_radon','Days PM2.5','Median AQI','Max AQI']]
    df['State_and_county'] = df.index


    scaler = preprocessing.StandardScaler()

    X = df[['smoking_daily','Days PM2.5','Median AQI','radon_mean']]
    X = scaler.fit_transform(X)

    y = df['cancer_incidence'].values

    counties = df['State_and_county'].unique()
    county_lookup = dict(zip(counties, range(len(counties))))
    county = df['State_and_county'].replace(county_lookup)


    with Model() as unpooled_model:

        beta0 = Normal('beta0', 0, sd=1e5, shape=len(counties))
        beta1 = Normal('beta1', 0, sd=1e5)
        beta2 = Normal('beta2', 0, sd=1e5)
        beta3 = Normal('beta3', 0, sd=1e5)
        beta4 = Normal('beta4', 0, sd=1e5)
        sigma = HalfCauchy('sigma', 5)

        theta = beta0[county] + beta1*X[:,0] + beta2*X[:,1] + beta3*X[:,2] + beta4*X[:,3]
        y_like = Normal('y', theta, sd=sigma, observed=y)

    with unpooled_model:
        unpooled_trace = sample(1000, n_init=50000, tune=10000)

    unpooled_estimates = pd.Series(unpooled_trace['beta0'].mean(axis=0)[county]+unpooled_trace['beta1'].mean(axis=0)*X[:,0]+unpooled_trace['beta2'].mean(axis=0)*X[:,1]+unpooled_trace['beta3'].mean(axis=0)*X[:,2]+unpooled_trace['beta4'].mean(axis=0)*X[:,3], index=df['State_and_county'])

    unpooled_se = pd.Series(unpooled_trace['beta0'].std(axis=0)[county]+unpooled_trace['beta1'].std(axis=0)*X[:,0]+unpooled_trace['beta2'].std(axis=0)*X[:,1]+unpooled_trace['beta3'].std(axis=0)*X[:,2]+unpooled_trace['beta4'].std(axis=0)*X[:,3], index=df['State_and_county'])

    predictions = pd.DataFrame(unpooled_estimates)
    predictions['State_and_county'] = predictions.index
    predictions = pd.merge(predictions, cancer_mean, how='left', on='State_and_county')
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(predictions[0], predictions['cancer_incidence'])
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x,'r-')
    plt.xlabel('Unpooled Estimates', fontsize=15)
    plt.ylabel('Mean Incidence', fontsize=15)
    plt.title('Predicted vs. Actual Mean Lung Cancer Incidence per 100,000', fontsize=17)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('../predictions13.png')

    unpooled_rmse = np.sqrt(mean_squared_error(y, unpooled_trace['beta0'].mean(axis=0)[county]+unpooled_trace['beta1'].mean(axis=0)*X[:,0]+unpooled_trace['beta2'].mean(axis=0)*X[:,1]+unpooled_trace['beta3'].mean(axis=0)*X[:,2]+unpooled_trace['beta4'].mean(axis=0)*X[:,3]))

    print(unpooled_rmse)

    # PLOT FORESTPLOT OF BETA0's

    # plt.figure(figsize=(6,14))
    # forestplot(unpooled_trace, varnames=['beta0'], ylabels=counties)
    # plt.savefig('../unpooled_model10.png')

    # PLOT ORDERED FORESTPLOT OF UNPOOLED ESTIMATES

    # plt.figure(figsize=(6,14))
    # order = unpooled_estimates.sort_values().index
    #
    # plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
    # for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    #     plt.plot([i,i], [m-se, m+se], 'b-')
    #
    # plt.ylabel('Cancer Incidence 95% Confidence Interval')
    # plt.xlabel('Ordered Counties')
    # plt.savefig('../unpooled_ordered10.png')



#######################################################
#Hierarchical Counties
#rmse: 5.96

    traces_pickle = 'traces.pickle'
    print("...loading pickle")
    tmp = open(traces_pickle,'rb')
    indiv_traces = pickle.load(tmp)
    tmp.close()


    df['State_and_county'] = df.index
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

    df['State'] = df['State_and_county'].str[-2:]

    scaler = preprocessing.StandardScaler()

    X = df[['cancer_incidence','smoking_daily', 'Days PM2.5','Max AQI']]
    #X = df[['cancer_incidence_x','smoking_daily','Max AQI']]
    X = scaler.fit_transform(X)

    y = df['cancer_incidence']


    states = df.State.unique()
    state_lookup = dict(zip(states, range(len(states))))
    state = df.State.replace(state_lookup).values

    counties = df['State_and_county'].unique()
    county_lookup = dict(zip(counties, range(len(counties))))
    county = df['State_and_county'].replace(county_lookup)

## Counties as groups

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
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(df))
            # Intercept for each county, distributed around group mean mu_a
            b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(counties))
            c = pm.Normal('beta2', mu=mu_c, sd=sigma_c, shape=len(counties))
            #d = pm.Normal('beta3', mu=mu_d, sd=sigma_d, shape=len(counties))
            # Model error
            eps = pm.HalfCauchy('eps', beta=1)

            # Expected value
            cancer_est = a + b[county] * X[:,1] + c[county] * X[:,2]


            # Data likelihood
            y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=y)


        with hierarchical_model:
            hierarchical_trace = pm.sample(1000, n_init=150000, tune=50000)
        rmse = np.sqrt(mean_squared_error(y, hierarchical_trace['alpha'].mean(axis=0)+hierarchical_trace['beta'].mean(axis=0)[county]*X[:,1]+hierarchical_trace['beta2'].mean(axis=0)[county]*X[:,2]))
        print(rmse)

        # unpooled_estimates = pd.Series(hierarchical_trace['alpha'].mean(axis=0)[county]+hierarchical_trace['beta'].mean(axis=0)[county]*X[:,1]+hierarchical_trace['beta2'].mean(axis=0)[county]*X[:,2], index=counties)
        #
        # unpooled_se = pd.Series(hierarchical_trace['alpha'].std(axis=0)[county]+hierarchical_trace['beta'].std(axis=0)[county]*X[:,1]+hierarchical_trace['beta2'].std(axis=0)[county]*X[:,2], index=counties)

#         hier_a = pd.Series(hierarchical_trace['alpha'].mean(axis=0))
#         hier_b = pd.Series(hierarchical_trace['beta'].mean(axis=0))
#         indv_a = [indiv_traces[c]['alpha'].mean() for c in counties]
#         indv_b = [indiv_traces[c]['beta'].mean() for c in counties]
#
        plt.figure(figsize=(30,20))
        forestplot(hierarchical_trace, varnames=['alpha'])
#         plt.savefig('../hierarchical_forestplot4.png')
#
#         plt.figure(figsize=(6,14))
#         pm.traceplot(hierarchical_trace)
#         plt.savefig('../hierarchical_traces5.png')
#         plt.show()
        print("...saving pickle")
        tmp = open(hier_traces2,'wb')
        pickle.dump(hierarchical_trace,tmp)
        tmp.close()

    else:
        print("...loading pickle")
        tmp = open(hier_traces2,'rb')
        hierarchical_trace = pickle.load(tmp)
        tmp.close()

        hierarchical_estimates = pd.Series(hierarchical_trace['alpha'].mean(axis=1)[county]+hierarchical_trace['beta'].mean(axis=1)[county]*X[:,1]+hierarchical_trace['beta2'].mean(axis=1)[county]*X[:,2])

        hierarchical_se = pd.Series(hierarchical_trace['alpha'].std(axis=1)[county]+hierarchical_trace['beta'].std(axis=1)[county]*X[:,1]+hierarchical_trace['beta2'].std(axis=1)[county]*X[:,2])


        four_counties = ['Yolo County, CA','King County, WA', 'Wayne County, MI', 'Fulton County, GA']
        fig, ax = plt.subplots(2,2,figsize=(35,20))
        counter = 1
        y_new = pd.DataFrame(y)
        y_new['count'] = [i for i in range(len(y_new))]
        y_new['estimates'] = hierarchical_estimates.values
        y_new['se'] = hierarchical_se.values
        for c in four_counties:
            #county_data = large_counties.loc[county][1:12]
            plt.subplot(2,2,counter)
            plt.grid(False)
            plt.scatter([str(i) for i in range(2001,2012)],df.loc[c].cancer_incidence.values, color='black', s=500)
            X = np.array([int(i) for i in range(2001,2012)])
            Y = np.array(df.loc[c].cancer_incidence.values)
            fit = np.polyfit(X, Y, deg=1)
            plt.plot(X, fit[0] * X + fit[1], color='red', lw=10)

            z = np.polyfit(X, y_new.loc[c].estimates.values, deg=1)
            plt.plot(X, z[0] * X + z[1], color='blue', lw=10)

            # lower = np.percentile(y_new.loc[c].se.values, 2.5, axis=0)
            # upper = np.percentile(y_new.loc[c].se.values, 97.5, axis=0)


            plt.plot(X, z[0] * X + z[1]-1*y_new.loc[c].se.values, 'b--')
            plt.plot(X, z[0] * X + z[1]+ y_new.loc[c].se.values, 'b--')
            #h_fit = np.polyfit( )
            plt.xlabel('Years', fontsize=35)
            plt.ylabel('Mean Lung Cancer Incidence per 100,000', fontsize=32)
            plt.title(c, fontsize=40)
            plt.tick_params(labelsize=35)
            plt.ylim([20,120])
            counter += 1
        plt.tight_layout()
        plt.savefig('../hier_counties2.png')


        # plt.figure(figsize=(6,10))
        # forestplot(hierarchical_trace, varnames=['alpha'])
        # plt.savefig('../hierarchical_forestplot5.png')
        #
        # plt.figure(figsize=(6,14))
        # pm.traceplot(hierarchical_trace)
        # plt.savefig('../hierarchical_traces6.png')
        # plt.show()
        #
        # plt.figure(figsize=(6,10))
        # forestplot(hierarchical_trace, varnames=['beta2'], ylabels='  '+states)
        # plt.savefig('../hierarchical_forestplot4.png')


# ####################
# #Hierarchical States- Very good!
# rmse=7.75


    traces_pickle = 'traces.pickle'
    print("...loading pickle")
    tmp = open(traces_pickle,'rb')
    indiv_traces = pickle.load(tmp)
    tmp.close()
    #print(indiv_traces)
    df = df_lung[['year','cancer_incidence','smoking','smoking_daily','pm25','ozone', 'radon_mean','Prob_low_radon','Prob_high_radon','Days PM2.5','Median AQI','Max AQI']]
    df['State_and_county'] = df.index


    df['State'] = df['State_and_county'].str[-2:]

    scaler = preprocessing.StandardScaler()

    y = df.pop('cancer_incidence')
    X = df[['smoking_daily', 'Days PM2.5','Median AQI', 'radon_mean']]
    X = scaler.fit_transform(X)




    states = df.State.unique()
    state_lookup = dict(zip(states, range(len(states))))
    state = df.State.replace(state_lookup).values

    counties = df['State_and_county'].unique()
    county_lookup = dict(zip(counties, range(len(counties))))
    county = df['State_and_county'].replace(county_lookup)

## States as groups

    hier_traces = 'hier_traces.pickle'
    if not os.path.isfile(hier_traces):
        with pm.Model() as hierarchical_model:
            # Hyperpriors
            mu_a = pm.Normal('mu_beta0', mu=0., sd=1e5)
            sigma_a = pm.HalfCauchy('sigma_beta0', beta=5)
            mu_b = pm.Normal('mu_beta1', mu=0., sd=1e5)
            sigma_b = pm.HalfCauchy('sigma_beta1', beta=5)
            mu_c = pm.Normal('mu_beta2', mu=0., sd=1e5)
            sigma_c = pm.HalfCauchy('sigma_beta2', beta=5)
            mu_d = pm.Normal('mu_beta3', mu=0., sd=1e5)
            sigma_d = pm.HalfCauchy('sigma_beta3', beta=5)
            mu_e = pm.Normal('mu_beta4', mu=0., sd=1e5)
            sigma_e = pm.HalfCauchy('sigma_beta4', beta=5)

            # Intercept and Beta distributions for each state
            a = pm.Normal('beta0', mu=mu_a, sd=sigma_a, shape=len(df))
            b = pm.Normal('beta1', mu=mu_b, sd=sigma_b, shape=len(states))
            c = pm.Normal('beta2', mu=mu_c, sd=sigma_c, shape=len(states))
            d = pm.Normal('beta3', mu=mu_d, sd=sigma_d, shape=len(states))
            e = pm.Normal('beta4', mu=mu_e, sd=sigma_e, shape=len(states))
            # Model error
            eps = pm.HalfCauchy('eps', beta=1)

            # Expected value
            cancer_est = a + b[state] * X[:,0] + c[state] * X[:,1] + d[state] * X[:,2] + e[state] * X[:,3]


            # Data likelihood
            y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=y)


        with hierarchical_model:
            hierarchical_trace = pm.sample(1000, n_init=150000, tune=100000)


        print("...saving pickle")
        tmp = open(hier_traces,'wb')
        pickle.dump(hierarchical_trace,tmp)
        tmp.close()

    else:
        print("...loading pickle")
        tmp = open(hier_traces,'rb')
        hierarchical_trace = pickle.load(tmp)
        tmp.close()


        rmse_state = np.sqrt(mean_squared_error(y, hierarchical_trace['beta0'].mean(axis=0)+hierarchical_trace['beta1'].mean(axis=0)[state]*X[:,0]+hierarchical_trace['beta2'].mean(axis=0)[state]*X[:,1]+hierarchical_trace['beta3'].mean(axis=0)[state]*X[:,2]+hierarchical_trace['beta4'].mean(axis=0)[state]*X[:,3]))

        print(rmse_state)

        hiers_estimates = pd.Series(hierarchical_trace['beta0'].mean(axis=0)+hierarchical_trace['beta1'].mean(axis=0)[state]*X[:,0]+hierarchical_trace['beta2'].mean(axis=0)[state]*X[:,1]+hierarchical_trace['beta3'].mean(axis=0)[state]*X[:,2]+hierarchical_trace['beta4'].mean(axis=0)[state]*X[:,3])

        hiers_se = pd.Series(hierarchical_trace['beta0'].std(axis=0)+hierarchical_trace['beta1'].std(axis=0)[state]*X[:,0]+hierarchical_trace['beta2'].std(axis=0)[state]*X[:,1]+hierarchical_trace['beta3'].std(axis=0)[state]*X[:,2]+hierarchical_trace['beta4'].std(axis=0)[state]*X[:,3])


        y_new = y.reset_index()
        y_new['Number'] = y_new.index

        predictions = pd.DataFrame(hiers_estimates)
        predictions_se = pd.DataFrame(hiers_se)
        predictions['Number'] = predictions.index
        predictions_se['Number'] = predictions_se.index
        predictions = pd.merge(predictions, predictions_se, how='left', on='Number')

        predictions = pd.merge(predictions, y_new, how='left', on='Number')
        predictions['State'] = predictions['State_and_county'].str[-2:]

        state_predictions = pd.DataFrame(predictions.groupby('State')['0_x','0_y'].mean())

        county_predictions = pd.DataFrame(predictions.groupby('State_and_county')['0_x','0_y'].mean())

        # Plot ordered forestplot of hierarchical estimates
        fig, ax = plt.subplots(figsize=(25,10))
        county_predictions.sort_values(by='0_x', inplace=True)


        for i, m, se in zip(range(len(county_predictions)), county_predictions['0_x'], county_predictions['0_y']):
            plt.plot([i,i], [m-se, m+se], 'b-')


        plt.scatter(range(len(county_predictions)), county_predictions['0_x'], c='red')

        plt.xticks([], [])
        ax.tick_params(labelsize=20)
        plt.ylabel('Cancer Incidence 95% Confidence Interval', fontsize=20)
        plt.xlabel('Ordered Counties', fontsize=20)
        plt.title('County Estimates from Hierarchical Regression (Group: States)', fontsize=30)
        plt.tight_layout()
        plt.savefig('../unpooled_ordereds7.png')
