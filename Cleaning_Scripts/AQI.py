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
    total['year'] = total['Year']

    
def adjust_columns(df, name):
    for yr in df.iloc[1][1:].index:
        df[str(yr)+name] = df[yr].values
        df.drop(yr, axis=1, inplace=True)




if __name__=='__main__':
    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/Data_Files/EPA_AIRDATA/'
    x = os.listdir(path=filepath)

    mapper = pd.read_csv('/home/davidhenslovitz/Galvanize/ZNAHealth/Data_Files/State_to_county.csv')

    df_dict = {}
    for textfile in x:
        df_dict['df_'+textfile[21:25]+'AQI'] = pd.read_csv(filepath+textfile)

    total = pd.concat(df_dict.values(),ignore_index=True)

    total = total[total.Year>1995]

    process(total)

    total = total[['State_and_county', 'year', 'Days PM2.5','Days PM10','Median AQI','Max AQI', 'Days with AQI']]


    df_lung_overall = pd.read_csv('/home/davidhenslovitz/Galvanize/ZNAHealth/lung_radon.csv',converters={'Combined': lambda x: str(x)})
    df_lung_overall = pd.merge(df_lung_overall, total, how='left', on=['State_and_county','year'])
    df_lung_overall.drop(df_lung_overall[pd.isnull(df_lung_overall).any(axis=1)].index, inplace=True)


    df_lung_overall.to_csv('/home/davidhenslovitz/Galvanize/ZNAHealth/lung_aqi.csv', index=False)
