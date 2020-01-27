import numpy as np
import pandas as pd
import pickle
from sklearn.mixture import GaussianMixture


#Percent Days PM2.5 and Days PM10
def scale_perc(df_lung):
    df_lung['pm25_perc'] = (df_lung['Days_PM2.5']/(df_lung['Days_with_AQI']))
    df_lung['pm10_perc'] = (df_lung['Days_PM10']/(df_lung['Days_with_AQI']))
    return df_lung

#Adds Gaussian Components of radon data (2 distinct distributions: high and low)
def gaus(df_lung):
    gmm = GaussianMixture(n_components=2)
    gmm = gmm.fit(df_lung[['Radon_mean']])
    results = gmm.predict_proba(df_lung[['Radon_mean']])
    df_lung['Prob_low_radon'] = [i[0] for i in results]
    df_lung['Prob_high_radon'] = [i[1] for i in results]
    return df_lung

#Log transform variables with skewed distributions
def log_trans(df_lung, var_list):
    for v in var_list:
        var_name = 'Log_'+v
        df_lung[var_name]=np.log(df_lung[v])
        df_lung[var_name][np.isneginf(df_lung[var_name])] = 0
    return df_lung


if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/intermediate_files/'
    merged_pickle = 'merged.pickle'
    engineered_pickle = 'engineered.pickle'

    print("...loading pickle")
    tmp = open(filepath+merged_pickle,'rb')
    df_merged = pickle.load(tmp)
    tmp.close()

    df_merged = log_trans(df_merged, ['Radon_mean','Max_AQI','Days_PM2.5'])
    df_merged = gaus(df_merged)
    df_merged = scale_perc(df_merged)

    print("...saving pickle")
    tmp = open(filepath+engineered_pickle,'wb')
    pickle.dump(df_merged,tmp)
    tmp.close()
