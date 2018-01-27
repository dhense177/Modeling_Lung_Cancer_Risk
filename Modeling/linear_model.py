import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pymc3 as pm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, ElasticNet
import os, pickle


















if __name__=='__main__':
    df_lung = pd.read_csv('lung_tri.csv')
    df_lung.index = df_lung['State_and_county']
    df = df_lung[['year','cancer_incidence','smoking','smoking_daily','pm25','ozone', 'radon_mean','Prob_low_radon','Prob_high_radon','Days PM2.5','Median AQI','Max AQI','ON-SITE_RELEASE_TOTAL']]

    df = df[df.index.value_counts()>10]
##FIX This - make a column for y variable 2011 incidence
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


    #X = df[df.year==2010]
    #y = X.pop('cancer_incidence')
    #y = df.cancer_incidence[df.year==2011]

    scaler = preprocessing.StandardScaler()

    X_cols = df[['smoking','smoking_daily','pm25','ozone', 'radon_mean','Prob_low_radon','Prob_high_radon','Days PM2.5','Median AQI','Max AQI','ON-SITE_RELEASE_TOTAL']]
    #X = df[['cancer_incidence_x','smoking_daily','Max AQI']]
    X = scaler.fit_transform(X_cols)

    y = df['2011_incidence']
    #y = df['cancer_incidence_x']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 17)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #Basic Liner Regression
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    rmse_linear = np.sqrt(mean_squared_error(y_test, lm.predict(X_test)))

    #Lasso
    lasso = Lasso(alpha=0.3)
    lasso.fit(X_train, y_train)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    variables = np.array(list(zip(X_cols.columns,lasso.coef_)))



    net = ElasticNet(alpha=0.3, l1_ratio=0.5, random_state=0)
    net.fit(X_train, y_train)

    rmse_net =  np.sqrt(mean_squared_error(y_test, net.predict(X_test)))
    #print(variables, rmse_net)


    # #Unpooled Model
    # for i in df.index:
    #     rmse_list = []
    #     county_df = df.loc[i]
    #     county_df = county_df[['year','cancer_incidence']]
    #     X = county_df[county_df.year<2011]['cancer_incidence'].mean()
    #     y = county_df[county_df.year==2011]['cancer_incidence'].values
    #     lm = LinearRegression()
    #     lm.fit(X, y)
    #     rmse = np.sqrt(mean_squared_error(y, lm.predict(X)))
    #     rmse_list.append(rmse)



###########################

    # traces_pickle = 'traces.pickle'
    # if not os.path.isfile(traces_pickle):
    #     #Pooled Model
    #     county_names = df.index.unique()
    #
    #     indiv_traces = {}
    #     for county_name in county_names:
    #         # Select subset of data belonging to county
    #         # c_data = data.ix[data.county == county_name]
    #         # c_data = c_data.reset_index(drop=True)
    #         #
    #         # c_log_radon = c_data.log_radon
    #         # c_floor_measure = c_data.floor.values
    #
    #         county_df = df.loc[county_name]
    #         county_df = county_df[['year','cancer_incidence']]
    #
    #         cancer = county_df['cancer_incidence'].mean()
    #         cancer_pred = county_df[county_df.year==2011]['cancer_incidence'].values
    #
    #         with pm.Model() as individual_model:
    #             # Intercept prior
    #             a = pm.Normal('alpha', mu=0, sd=1)
    #             # Slope prior
    #             b = pm.Normal('beta', mu=0, sd=1)
    #
    #             # Model error prior
    #             eps = pm.HalfCauchy('eps', beta=1)
    #
    #             # Linear model
    #             cancer_est = a + b * cancer
    #
    #             # Data likelihood
    #             y_like = pm.Normal('y_like', mu=cancer_est, sd=eps, observed=cancer_pred)
    #
    #             # Inference button (TM)!
    #             trace = pm.sample(10,progressbar=False)
    #
    #         indiv_traces[county_name] = trace
    #
    #         print("...saving pickle")
    #         tmp = open(traces_pickle,'wb')
    #         pickle.dump(indiv_traces,tmp)
    #         tmp.close()
    #     else:
    #         print("...loading pickle")
    #         tmp = open(traces_pickle,'rb')
    #         indiv_traces = pickle.load(tmp)
    #         tmp.close()
