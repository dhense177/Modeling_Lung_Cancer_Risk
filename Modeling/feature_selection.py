import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle, time
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, ElasticNet


#Provides list of all different combinations of variables, selecting only 1 from each group in groups
def combos(groups, items):
    group_list = []
    for i in range(1,6):
        # num = groups[:i+1]
        group_list.append(list(itertools.combinations(groups,i)))
    group_list = list(itertools.chain(*group_list))

    combinations = []
    combinations.append([list(i) for i in list(itertools.combinations(items,1))])
    for i in group_list:
        if len(i)>1:
            combinations.append([list(i) for i in list(itertools.product(*i))])

    combinations = [i for sub in combinations for i in sub]
    return combinations


def selection_prep(df):
    X = df[[i for i in items]]
    y = df['Cancer_Rate']
    #Standardize
    X /= np.sqrt(np.sum(X ** 2, axis=0))

    return X, y


def feature_selection(X, y):
    features = pd.DataFrame()
    feature_list = []
    bic_list = []
    aic_list = []
    for c in combinations:
        X_new = X[c]

        model_bic = LassoLarsIC(criterion='bic')
        model_bic.fit(X_new, y)
        alpha_bic_ = model_bic.alpha_
        feature_list.append(c)
        bic_list.append(model_bic.criterion_.min())

        model_aic = LassoLarsIC(criterion='aic')
        model_aic.fit(X_new, y)
        alpha_aic_ = model_aic.alpha_
        aic_list.append(model_aic.criterion_.min())

    features['features'] = feature_list
    features['BIC'] = bic_list
    features['AIC'] = aic_list
    features = features.sort_values(by='BIC', ascending=True)

    return features




if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/intermediate_files/'
    engineered_pickle = 'engineered.pickle'

    print("...loading pickle")
    tmp = open(filepath+engineered_pickle,'rb')
    df = pickle.load(tmp)
    tmp.close()

    '''
    scatter_matrix(df)
    plt.show()

    corrmat = df.corr()
    sns.heatmap(corrmat, annot=True, vmax=.8, square=True)
    plt.show()
    '''

    #Breaking out into groups of correlated variables. Looking for all combinations of variables where there is not more than one from a single group
    Smoking_group = ['Smoking','Smoking_daily']
    PM25_group = ['Days_PM2.5','Log_Days_PM2.5','pm25_perc']
    PM10_group = ['Days_PM10','pm10_perc']
    AQI_group = ['Median_AQI','Max_AQI','Log_Max_AQI']
    Radon_group = ['Radon_mean','Log_Radon_mean','Prob_low_radon','Prob_high_radon']

    groups = [Smoking_group,PM25_group,PM10_group,AQI_group,Radon_group]
    items = [item for sublist in groups for item in sublist]

    combinations = combos(groups,items)

    X, y = selection_prep(df)
    features = feature_selection(X,y)
    best_vars = features.iloc[0]
