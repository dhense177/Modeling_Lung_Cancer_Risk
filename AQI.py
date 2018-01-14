import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os


def process(total):
    lst = []
    for i in total.State:
        if i not in list(mapper.State.values):
            lst.append('--')
        else:
            lst.append(mapper.iloc[list(mapper.State.values).index(i)]['Postal'])
    total.State = lst
    total['State_and_county'] = total.County + " County, " + total.State.astype(str)







if __name__=='__main__':
    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/EPA_AIRDATA/'
    x = os.listdir(path=filepath)

    mapper = pd.read_csv('State_to_county.csv')

    df_dict = {}
    for textfile in x:
        df_dict['df_'+textfile[21:25]+'AQI'] = pd.read_csv(filepath+textfile)

    total = pd.concat(df_dict.values(),ignore_index=True)

    process(total)

    total_pm25 = total.pivot(index='State_and_county',columns='Year',values='Days PM2.5').reset_index()

    for yr in total_pm25.iloc[1][1:].index:
        total_pm25[str(yr)+"_AQI"] = total_pm25[yr].values
        total_pm25.drop(yr, axis=1, inplace=True)
    #Join with lung dataframes
    df_lung = pd.read_csv('lung_dataframe.csv',converters={'Combined': lambda x: str(x)})
    df_lung_overall = pd.read_csv('lung_dataframe_overall.csv',converters={'Combined': lambda x: str(x)})

    df_lung = pd.merge(df_lung, total_pm25,how='left',on='State_and_county')
    df_lung_overall = pd.merge(df_lung_overall, total_pm25,how='left',on='State_and_county')

    df_lung.to_csv('lung_dataframe2.csv', index=False)
    df_lung_overall.to_csv('lung_dataframe_overall2.csv', index=False)
