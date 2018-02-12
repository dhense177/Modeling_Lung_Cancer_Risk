import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform, forestplot
from scipy.stats import boxcox, probplot, norm
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, ElasticNet
import pickle, os, csv




if __name__=='__main__':
    df_lung = pd.read_csv('lung_final.csv')

    #All relevant variables
    #X = df_lung[['cancer_incidence_x','smoking','smoking_daily', 'radon_mean','log_radon', 'Prob_low_radon', 'Prob_high_radon', 'Days PM2.5','log_pm25','pm25', 'pm25_perc','Days PM10', 'pm10_perc','Median AQI','Max AQI', 'log_max_AQI','ozone','log_ozone','ON-SITE_RELEASE_TOTAL', 'log_releases']]

    #BIC features
    X = df_lung[['cancer_incidence_x','smoking_daily', 'log_radon', 'Days PM2.5','Median AQI']]

    y = X.pop('cancer_incidence_x')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 17)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
#######################################################################    #Pooled Liner Regression - ScikitLearn

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    rmse_linear = np.sqrt(mean_squared_error(y_test, lm.predict(X_test)))

    #Lasso
    lasso = Lasso(alpha=.1)

    #Gridsearch lasso parameters
######################################
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
######################################


    lasso.fit(X_train, y_train)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    variables = np.array(list(zip(X.columns,lasso.coef_)))


    net = ElasticNet(alpha=0.1, l1_ratio=1)

    #Gridsearch ElasticNet parameters
######################################
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
######################################
    net.fit(X_train, y_train)

    rmse_net =  np.sqrt(mean_squared_error(y_test, net.predict(X_test)))




#########################################################################
## PYMC3 models below

    df_lung.index = df_lung['State_and_county']
    cancer_mean = pd.DataFrame(df_lung.groupby('State_and_county')['cancer_incidence_x'].mean()).reset_index()

    scaler = preprocessing.StandardScaler()

    X = df_lung[['smoking_daily','Days PM2.5','Median AQI','log_radon']]
    X = scaler.fit_transform(X)

    y = df_lung['cancer_incidence_x'].values

    states = df_lung.State.unique()
    state_lookup = dict(zip(states, range(len(states))))
    state = df_lung.State.replace(state_lookup).values

    counties = df_lung['State_and_county'].unique()
    county_lookup = dict(zip(counties, range(len(counties))))
    county = df_lung['State_and_county'].replace(county_lookup)

#########################################################################
### Unpooled Linear Model



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

    unpooled_estimates = pd.Series(unpooled_trace['beta0'].mean(axis=0)[county]+unpooled_trace['beta1'].mean(axis=0)*X[:,0]+unpooled_trace['beta2'].mean(axis=0)*X[:,1]+unpooled_trace['beta3'].mean(axis=0)*X[:,2]+unpooled_trace['beta4'].mean(axis=0)*X[:,3], index=df_lung['State_and_county'])

    unpooled_se = pd.Series(unpooled_trace['beta0'].std(axis=0)[county]+unpooled_trace['beta1'].std(axis=0)*X[:,0]+unpooled_trace['beta2'].std(axis=0)*X[:,1]+unpooled_trace['beta3'].std(axis=0)*X[:,2]+unpooled_trace['beta4'].std(axis=0)*X[:,3], index=df_lung['State_and_county'])

    predictions = pd.DataFrame(unpooled_estimates)
    predictions['State_and_county'] = predictions.index
    predictions = pd.merge(predictions, cancer_mean, how='left', on='State_and_county')
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(predictions[0], predictions['cancer_incidence_x'])
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

    residuals = df_lung.cancer_incidence_x.values-unpooled_estimates.values

    fig, ax = plt.subplots()
    ax = sns.distplot(residuals, fit=norm, color='blue')
    ax.set_xlabel('Distribution of Unpooled Model Residuals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.tick_params(labelsize=10)
    plt.savefig('../resids_dist2.png')

    fig = plt.subplot()
    resid_var = sns.residplot(unpooled_estimates.values, y, lowess=True, color='red', line_kws={'color':'black'})
    plt.savefig('../resid_varplot2.png')

    fig = plt.subplot()
    res = probplot(residuals, plot=plt)
    plt.savefig('../probplot2.png')

    ##UNPOOLED LINEAR PLOT
    four_counties = ['Yolo County, CA','Warren County, KY', 'Wayne County, MI', 'Grant County, NM']

    fig, ax = plt.subplots(2,2,figsize=(35,20))
    counter = 1
    for c in four_counties:

        plt.subplot(2,2,counter)
        plt.grid(False)
        plt.scatter([str(i) for i in range(2001,2012)],df_lung.loc[c].cancer_incidence_x.values, color='black', s=500)

        X = np.array([int(i) for i in range(2001,2012)])
        Y = np.array(df_lung.loc[c].cancer_incidence_x.values)
        fit = np.polyfit(X, Y, deg=1)
        plt.plot(X, fit[0] * X + fit[1], color='black', lw=10)

        plt.scatter([str(i) for i in range(2001,2012)],unpooled_estimates.loc[c].values, color='red', s=500)

        plt.xlabel('Years', fontsize=35)
        plt.ylabel('Mean Lung Cancer Incidence per 100,000', fontsize=32)
        plt.title(c, fontsize=40)
        plt.tick_params(labelsize=35)
        plt.ylim([20,120])
        counter += 1
        plt.tight_layout()
        plt.savefig('../linear_model.png')


    ## PLOT FORESTPLOT OF BETA0's

    plt.figure(figsize=(6,14))
    forestplot(unpooled_trace, varnames=['beta1'], ylabels=counties)
    plt.savefig('../unpooled_model11.png')

    ## PLOT ORDERED FORESTPLOT OF UNPOOLED ESTIMATES

    plt.figure(figsize=(6,14))
    order = unpooled_estimates.sort_values().index

    plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
    for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
        plt.plot([i,i], [m-se, m+se], 'b-')

    plt.ylabel('Cancer Incidence 95% Confidence Interval')
    plt.xlabel('Ordered Counties')
    plt.savefig('../unpooled_ordered10.png')
#######################################################
#Hierarchical states

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

            # Coefficients for each state
            a = pm.Normal('beta0', mu=mu_a, sd=sigma_a, shape=len(df_lung))
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

        ##CALCULATE ROOT MEAN SQUARE ERROR (RMSE)
        rmse_state = np.sqrt(mean_squared_error(y, hierarchical_trace['beta0'].mean(axis=0)+hierarchical_trace['beta1'].mean(axis=0)[state]*X[:,0]+hierarchical_trace['beta2'].mean(axis=0)[state]*X[:,1]+hierarchical_trace['beta3'].mean(axis=0)[state]*X[:,2]+hierarchical_trace['beta4'].mean(axis=0)[state]*X[:,3]))

        print(rmse_state)

        hiers_estimates = pd.Series(hierarchical_trace['beta0'].mean(axis=0)+hierarchical_trace['beta1'].mean(axis=0)[state]*X[:,0]+hierarchical_trace['beta2'].mean(axis=0)[state]*X[:,1]+hierarchical_trace['beta3'].mean(axis=0)[state]*X[:,2]+hierarchical_trace['beta4'].mean(axis=0)[state]*X[:,3])

        hiers_se = pd.Series(hierarchical_trace['beta0'].std(axis=0)+hierarchical_trace['beta1'].std(axis=0)[state]*X[:,0]+hierarchical_trace['beta2'].std(axis=0)[state]*X[:,1]+hierarchical_trace['beta3'].std(axis=0)[state]*X[:,2]+hierarchical_trace['beta4'].std(axis=0)[state]*X[:,3])


        #MORE PLOTS

        # indv_a = np.array([hierarchical_trace['beta0'][s].mean() for s in state])
        # indv_b = np.array([hierarchical_trace['beta1'][s].mean() for s in state])
        # indv_c = np.array([hierarchical_trace['beta2'][s].mean() for s in state])
        # indv_d = np.array([hierarchical_trace['beta3'][s].mean() for s in state])
        # indv_e = np.array([hierarchical_trace['beta4'][s].mean() for s in state])
        #
        # hiers_county_estimates = pd.Series(indv_a.mean(axis=0)+indv_b.mean(axis=0)*X[:,0]+indv_c.mean(axis=0)*X[:,1]+indv_d.mean(axis=0)*X[:,2]+indv_e.mean(axis=0)*X[:,3])
        #
        # hiers_county_se = pd.Series(indv_a.std(axis=0)+indv_b.std(axis=0)*X[:,0]+indv_c.std(axis=0)*X[:,1]+indv_d.std(axis=0)*X[:,2]+indv_e.std(axis=0)*X[:,3])

        # Plot ordered forestplot of hierarchical estimates
        # plt.figure(figsize=(50,20))
        # order = hiers_county_estimates.sort_values().index
        #
        #
        # for i, m, se in zip(range(len(hiers_county_estimates)), hiers_county_estimates[order], hiers_county_se[order]):
        #     plt.plot([i,i], [m-se, m+se], 'b-')
        #
        #
        # plt.scatter(range(len(hiers_county_estimates)), hiers_county_estimates[order], c='red')
        #
        # plt.ylabel('Cancer Incidence 95% Confidence Interval')
        # plt.xlabel('Ordered Counties')
        # plt.savefig('../unpooled_ordereds3.png')
        #
        # plt.figure(figsize=(6,14))
        # pm.traceplot(hierarchical_trace)
        # plt.savefig('../hierarchical_traces11.png')
        # plt.show()
        #
        # plt.figure(figsize=(6,10))
        # forestplot(hierarchical_trace, varnames=['beta2'], ylabels='  '+states)
        # plt.savefig('../hierarchical_forestplot4.png')
        # y_new = y.reset_index()
        # y_new['Number'] = y_new.index
        #
        # predictions = pd.DataFrame(hiers_estimates)
        # predictions_se = pd.DataFrame(hiers_se)
        # predictions['Number'] = predictions.index
        # predictions_se['Number'] = predictions_se.index
        # predictions = pd.merge(predictions, predictions_se, how='left', on='Number')
        #
        # predictions = pd.merge(predictions, y_new, how='left', on='Number')
        # predictions['State'] = predictions['State_and_county'].str[-2:]
        #
        # state_predictions = pd.DataFrame(predictions.groupby('State')['0_x','0_y'].mean())
        #
        # county_predictions = pd.DataFrame(predictions.groupby('State_and_county')['0_x','0_y'].mean())
        #
        # # Plot ordered forestplot of hierarchical estimates
        # fig, ax = plt.subplots(figsize=(25,10))
        # county_predictions.sort_values(by='0_x', inplace=True)
        #
        #
        # for i, m, se in zip(range(len(county_predictions)), county_predictions['0_x'], county_predictions['0_y']):
        #     plt.plot([i,i], [m-se, m+se], 'b-')
        #
        #
        # plt.scatter(range(len(county_predictions)), county_predictions['0_x'], c='red')
        #
        # plt.ylim(0,200)
        # plt.xticks([], [])
        # ax.tick_params(labelsize=20)
        # plt.ylabel('Cancer Incidence 95% Confidence Interval', fontsize=20)
        # plt.xlabel('Ordered Counties', fontsize=20)
        # plt.title('County Estimates from Hierarchical Regression (Group: States)', fontsize=30)
        # plt.tight_layout()
        # plt.savefig('../unpooled_ordereds8.png')



#######################################################
#Hierarchical Counties
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
            mu_d = pm.Normal('mu_beta3', mu=0., sd=1e5)
            sigma_d = pm.HalfCauchy('sigma_beta3', beta=5)
            mu_e = pm.Normal('mu_beta4', mu=0., sd=1e5)
            sigma_e = pm.HalfCauchy('sigma_beta4', beta=5)

            # Intercept for each county, distributed around group mean mu_a
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(df_lung))
            # Slope coefficients for each county, distributed around group means
            b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(counties))
            c = pm.Normal('beta2', mu=mu_c, sd=sigma_c, shape=len(counties))
            d = pm.Normal('beta3', mu=mu_d, sd=sigma_d, shape=len(counties))
            e = pm.Normal('beta4', mu=mu_e, sd=sigma_e, shape=len(counties))
            # Model error
            eps = pm.HalfCauchy('eps', beta=1)

            # Expected value
            cancer_est = a + b[county] * X[:,0] + c[county] * X[:,1] + d[county] * X[:,2] + e[county] * X[:,3]


            # Data likelihood
            y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=y)


        with hierarchical_model:
            hierarchical_trace = pm.sample(5000, n_init=150000, tune=100000)

        ##CALCULATE RMSE
        rmse = np.sqrt(mean_squared_error(y, hierarchical_trace['alpha'].mean(axis=0)+hierarchical_trace['beta'].mean(axis=0)[county]*X[:,0]+hierarchical_trace['beta2'].mean(axis=0)[county]*X[:,1]+hierarchical_trace['beta3'].mean(axis=0)[county]*X[:,2]+hierarchical_trace['beta4'].mean(axis=0)[county]*X[:,3]))
        print(rmse)

        print("...saving pickle")
        tmp = open(hier_traces2,'wb')
        pickle.dump(hierarchical_trace,tmp)
        tmp.close()

    else:
        print("...loading pickle")
        tmp = open(hier_traces2,'rb')
        hierarchical_trace = pickle.load(tmp)
        tmp.close()

        hierarchical_estimates = pd.Series(hierarchical_trace['alpha'].mean(axis=0)+hierarchical_trace['beta'].mean(axis=0)[county]*X[:,0]+hierarchical_trace['beta2'].mean(axis=0)[county]*X[:,1]+hierarchical_trace['beta3'].mean(axis=0)[county]*X[:,2]+hierarchical_trace['beta4'].mean(axis=0)[county]*X[:,3])

        hierarchical_se = pd.Series(hierarchical_trace['alpha'].std(axis=0)+hierarchical_trace['beta'].std(axis=0)[county]*X[:,0]+hierarchical_trace['beta2'].std(axis=0)[county]*X[:,1]+hierarchical_trace['beta3'].std(axis=0)[county]*X[:,2]+hierarchical_trace['beta4'].std(axis=0)[county]*X[:,3])

        ####################################################
        ##PLOTS BELOW

        #FOUR COUNTIES PLOT
        four_counties = ['Yolo County, CA','Warren County, KY', 'Wayne County, MI', 'Grant County, NM']

        fig, ax = plt.subplots(2,2,figsize=(35,20))

        counter = 1
        y_new = pd.DataFrame(df_lung['cancer_incidence_x'])
        y_new['count'] = [i for i in range(len(y_new))]
        y_new['State'] = y_new.index.str[-2:]
        y_new['estimates'] = hierarchical_estimates.values
        y_new['se'] = hierarchical_se.values
        y_new['state_estimates'] = hiers_estimates.values
        y_new['state_se'] = hiers_se.values
        for c in four_counties:

            plt.subplot(2,2,counter)
            plt.grid(False)
            plt.plot([str(i) for i in range(2001,2012)],df_lung.loc[c].cancer_incidence_x.values, '.k-', lw=5, markersize=50)
            X = np.array([int(i) for i in range(2001,2012)])
            Y = np.array(df_lung.loc[c].cancer_incidence_x.values)

            z = np.polyfit(X, y_new.loc[c].estimates.values, deg=1)
            plt.plot([str(i) for i in range(2001,2012)], y_new.loc[c].estimates.values, '.b-', lw=5, markersize=35)

            plt.fill_between(X, z[0] * X + z[1] + y_new.loc[c].se.values,z[0] * X + z[1]-y_new.loc[c].se.values, interpolate=True, color = 'blue', alpha = 0.4, label = '95%_CI')

            ## COUNTY MEAN LINE
            county_mean = df_lung['cancer_incidence_x'].mean()
            plt.axhline(y=county_mean, color='#00FFFF', ls='--', lw=3)


            s = np.polyfit(X, y_new.loc[c].state_estimates.values, deg=1)
            plt.plot([str(i) for i in range(2001,2012)], y_new.loc[c].state_estimates.values, '.g-', lw=5, markersize=35)

            plt.fill_between(X, s[0] * X + s[1]+ y_new.loc[c].state_se.values, s[0] * X + s[1]-1*y_new.loc[c].state_se.values, interpolate=True, color= 'green', alpha = 0.4, label = '95%_CI')

            ## STATE MEAN LINE
            df_state = df_lung[['cancer_incidence_x','State']].reset_index()
            df_state.index=df_state.State
            state_mean = df_state.loc[c[-2:]]['cancer_incidence_x'].mean()
            plt.axhline(y=state_mean, color='#7CFC00', ls='--', lw=3)


            plt.xlabel('Years', fontsize=35)
            plt.ylabel('Mean Lung Cancer Incidence per 100,000', fontsize=32)
            plt.title(c, fontsize=40)
            plt.tick_params(labelsize=35)
            plt.ylim([20,120])
            counter += 1
            plt.tight_layout()

        plt.tight_layout()
        plt.savefig('../hier_counties15.png')

        ###SINGLE COUNTY PLOT
        single_county = ['Warren County, KY']
        fig, ax = plt.subplots(figsize=(35,20))

        for c in single_county:

            plt.grid(False)
            plt.plot([str(i) for i in range(2001,2012)],df_lung.loc[c].cancer_incidence_x.values, '.k-', lw=5, markersize=50)
            X = np.array([int(i) for i in range(2001,2012)])
            Y = np.array(df_lung.loc[c].cancer_incidence_x.values)


            z = np.polyfit(X, y_new.loc[c].estimates.values, deg=1)
            plt.plot([str(i) for i in range(2001,2012)], y_new.loc[c].estimates.values, '.b-', lw=5, markersize=35)

            plt.fill_between(X, z[0] * X + z[1] + y_new.loc[c].se.values,z[0] * X + z[1]-y_new.loc[c].se.values, interpolate=True, color = 'blue', alpha = 0.4, label = '95%_CI')

            #COUNTY MEAN LINE
            county_mean = df_lung['cancer_incidence_x'].mean()
            plt.axhline(y=county_mean, color='#00FFFF', ls='--', lw=3)


            s = np.polyfit(X, y_new.loc[c].state_estimates.values, deg=1)
            plt.plot([str(i) for i in range(2001,2012)], y_new.loc[c].state_estimates.values, '.g-', lw=5, markersize=35)

            plt.fill_between(X, s[0] * X + s[1]+ y_new.loc[c].state_se.values, s[0] * X + s[1]-1*y_new.loc[c].state_se.values, interpolate=True, color= 'green', alpha = 0.4, label = '95%_CI')

            ## STATE MEAN LINE
            df_state = df_lung[['cancer_incidence_x','State']].reset_index()
            df_state.index=df_state.State
            state_mean = df_state.loc[c[-2:]]['cancer_incidence_x'].mean()
            plt.axhline(y=state_mean, color='#7CFC00', ls='--', lw=3)



            plt.xlabel('Years', fontsize=40)
            plt.ylabel('Lung Cancer Incidence per 100,000', fontsize=40)
            plt.title(c, fontsize=50)
            plt.tick_params(labelsize=35)
            plt.ylim([40,140])

            plt.tight_layout()

        plt.tight_layout()
        plt.savefig('../hier_county2.png')



        plt.figure(figsize=(6,14))
        pm.traceplot(hierarchical_trace)
        plt.savefig('../hierarchical_traces16.png')
        plt.show()

        plt.figure(figsize=(6,10))
        forestplot(hierarchical_trace, varnames=['beta2'], ylabels='  '+states)
        plt.savefig('../hierarchical_forestplot4.png')

        # PLOT ORDERED FORESTPLOT OF HIERARCHICAL ESTIMATES
        fig, ax = plt.subplots(figsize=(25,10))

        y_new = df_lung['cancer_incidence_x'].reset_index()
        y_new['Number'] = y_new.index


        #UNPOOLED PREDICTIONS
        predictions = pd.DataFrame(unpooled_estimates).reset_index()
        predictions_se = pd.DataFrame(unpooled_se).reset_index()
        predictions['Number'] = predictions.index
        predictions_se['Number'] = predictions_se.index
        predictions = pd.merge(predictions, predictions_se, how='left', on='Number')

        predictions = pd.merge(predictions, y_new, how='left', on='Number')
        predictions['State'] = predictions['State_and_county'].str[-2:]

        predictions = pd.DataFrame(predictions.groupby('State_and_county')['0_x','0_y'].mean())

        predictions.sort_values(by='0_x', inplace=True)

        for i, m, se in zip(range(len(predictions)), predictions['0_x'], predictions['0_y']):
            plt.plot([i,i], [m-se, m+se], 'r-')

        plt.scatter(range(len(predictions)), predictions['0_x'], c='red')

        #county-grouped predictions
        county_predictions = pd.DataFrame(hierarchical_estimates)
        county_predictions_se = pd.DataFrame(hierarchical_se)
        county_predictions['Number'] = county_predictions.index
        county_predictions_se['Number'] = county_predictions_se.index
        county_predictions = pd.merge(county_predictions, county_predictions_se, how='left', on='Number')

        county_predictions = pd.merge(county_predictions, y_new, how='left', on='Number')
        county_predictions['State'] = county_predictions['State_and_county'].str[-2:]

        county_predictions = pd.DataFrame(county_predictions.groupby('State_and_county')['0_x','0_y'].mean())

        county_predictions.sort_values(by='0_x', inplace=True)


        for i, m, se in zip(range(len(county_predictions)), county_predictions['0_x'], county_predictions['0_y']):
            plt.plot([i,i], [m-se, m+se], 'b-')


        plt.scatter(range(len(county_predictions)), county_predictions['0_x'], c='blue')

        #State-grouped predictions
        predictions_state = pd.DataFrame(hiers_estimates)
        predictions_state_se = pd.DataFrame(hiers_se)
        predictions_state['Number'] = predictions_state.index
        predictions_state_se['Number'] = predictions_state_se.index
        predictions_state = pd.merge(predictions_state, predictions_state_se, how='left', on='Number')

        predictions_state = pd.merge(predictions_state, y_new, how='left', on='Number')
        predictions_state['State'] = predictions_state['State_and_county'].str[-2:]

        county_predictions_state = pd.DataFrame(predictions_state.groupby('State_and_county')['0_x','0_y'].mean())

        county_predictions_state.sort_values(by='0_x', inplace=True)


        for i, m, se in zip(range(len(county_predictions_state)), county_predictions_state['0_x'], county_predictions_state['0_y']):
            plt.plot([i,i], [m-se, m+se], 'g-')


        plt.scatter(range(len(county_predictions_state)), county_predictions_state['0_x'], c='green')

        plt.ylim(0,160)
        plt.xticks([], [])
        ax.tick_params(labelsize=20)
        plt.ylabel('Cancer Incidence 95% Confidence Intervals', fontsize=20)
        plt.xlabel('Ordered Counties', fontsize=20)
        plt.title('County Estimates from Regressions', fontsize=30)
        plt.tight_layout()
        plt.savefig('../all_estimates2.png')
