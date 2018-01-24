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


    # #Unpooled Model - Works
    # rmse_list = []
    # for i in df.index:
    #     county_df = df.loc[i]
    #     county_df = county_df[['year','cancer_incidence']]
    #     X = county_df[county_df.year.apply(pd.to_numeric)<2011]['year'].values
    #     y = county_df[county_df.year.apply(pd.to_numeric)<2011]['cancer_incidence'].values
    #     lm = LinearRegression()
    #     lm.fit(X.reshape(-1,1), y)
    #     y_pred = lm.intercept_ + lm.coef_*X
    #     rmse = np.sqrt(mean_squared_error(y, y_pred))
    #     rmse_list.append(rmse)
    # mean_rmse = np.mean(rmse_list)

    df['State_and_county'] = df.index
    grouped = pd.DataFrame(df[df.year.apply(pd.to_numeric)<2011]['cancer_incidence'])
    grouped = pd.DataFrame(grouped.groupby('State_and_county')['cancer_incidence'].mean())
    grouped['State_and_county'] = grouped.index
    df = pd.merge(df, grouped, how='left', on='State_and_county')

    counties = df['State_and_county'].unique()
    county_lookup = dict(zip(counties, range(len(counties))))
    county = df['State_and_county'].replace(county_lookup).values
    with Model() as unpooled_model:

        beta0 = Normal('beta0', 0, sd=1e5, shape=len(counties))
        beta1 = Normal('beta1', 0, sd=1e5)
        sigma = HalfCauchy('sigma', 5)

        theta = beta0[county] + beta1*df.mean_cancer

        y = Normal('y', theta, sd=sigma, observed=df['cancer_incidence_x'].values)



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
