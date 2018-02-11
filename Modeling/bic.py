import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pymc3 import Model, sample, Normal, HalfCauchy, Uniform
from scipy.stats import boxcox, norm, probplot
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, LassoLarsCV, ElasticNet

radon_group = ['log_radon', 'radon_mean', 'Prob_low_radon', 'Prob_high_radon']
smoking_group = ['smoking', 'smoking_daily']
pm_25_group = ['Days PM2.5','log_pm25','pm25','pm25_perc']
pm_10_group = ['Days PM10', 'pm10_perc']
ozone_group = ['ozone', 'log_ozone']
aqi_group = ['Median AQI', 'Max AQI','log_max_AQI']
tri_group = ['ON-SITE_RELEASE_TOTAL','log_releases']

def ic_calc(X):
    features = pd.DataFrame()
    feature_list = []
    bic_list = []
    aic_list = []
    for s in smoking_group:
        for a in aqi_group:
            for r in radon_group:
                for t in tri_group:
                    for p in pm_25_group:
                        for n in pm_10_group:
                            for o in ozone_group:
                                lst = [s,a,r,p]

                                X_new = X[lst]

                                model_bic = LassoLarsIC(criterion='bic')
                                t1 = time.time()
                                model_bic.fit(X_new, y)
                                t_bic = time.time() - t1

                                alpha_bic_ = model_bic.alpha_
                                feature_list.append(lst)
                                bic_list.append(model_bic.criterion_.min())


                                model_aic = LassoLarsIC(criterion='aic')
                                model_aic.fit(X_new, y)
                                alpha_aic_ = model_aic.alpha_
                                aic_list.append(model_aic.criterion_.min())

    features['features'] = feature_list
    features['BIC'] = bic_list
    features['AIC'] = aic_list
    features = features.sort_values(by='BIC', ascending=True)

    return features.sort_values(by='BIC', ascending=True).iloc[0]




if __name__=='__main__':
    df_lung = pd.read_csv('/home/davidhenslovitz/Galvanize/ZNAHealth/lung_final.csv')


    X = df_lung[['cancer_incidence_x','smoking','smoking_daily', 'radon_mean','log_radon', 'Prob_low_radon', 'Prob_high_radon', 'Days PM2.5','log_pm25','pm25', 'pm25_perc','Days PM10', 'pm10_perc','Median AQI','Max AQI', 'log_max_AQI','ozone','log_ozone','ON-SITE_RELEASE_TOTAL', 'log_releases']]
    y = X.pop('cancer_incidence_x')

    X /= np.sqrt(np.sum(X ** 2, axis=0))

    scores = ic_calc(X)


    print(scores)
