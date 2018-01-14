import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def process(df):
    start_dates = df.startdt.apply(lambda x: str(x).zfill(6))
    stop_dates = df.stopdt.apply(lambda x: str(x).zfill(6))

    df['startdt'] = pd.to_datetime(start_dates)
    df['stopdt'] = pd.to_datetime(stop_dates)

    df['year'] = pd.DatetimeIndex(df['stopdt']).year

    df.stfips = df.cntyfips.apply(lambda x: str(x).zfill(2))
    df.cntyfips = df.cntyfips.apply(lambda x: str(x).zfill(3))
    df['State-county recode'] = df.stfips+df.cntyfips

    df['county'] = df.county.str.title()
    df['State_and_county'] = df.county.str.strip() + ' County, ' + df.state.str.strip()


def index_lookup(df_lung):
    counter = 0
    lst = []
    for i in df_lung["State_and_county"]:
        if i not in list(grouped['State_and_county'].values):
            lst.append(np.nan)
        else:
            ind = list(grouped['State_and_county'].values).index(i)
            lst.append(ind)
        counter += 1
    return lst


def add_radon(lst, df_lung):
    newlist = []
    for x in lst:
        if np.isnan(x):
            newlist.append(np.nan)
        else:
            newlist.append(grouped.iloc[x]['activity'])
    df_lung['radon_mean'] = newlist



if __name__=='__main__':
    df1 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs1.dat')
    df2 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs2.dat')
    df3 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs3.dat')
    df4 =  pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs4.dat')
    df5 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs5.dat')

    df = pd.concat([df1, df2, df3, df4, df5])
    df.columns = df.columns.map(str.strip)

    process(df)

    grouped = pd.DataFrame(df.groupby(['State_and_county'])['activity'].mean()).reset_index()


    df_lung = pd.read_csv('lung_dataframe2.csv',converters={'Combined': lambda x: str(x),'State-county recode_x': lambda x: str(x)})

    df_lung_overall = pd.read_csv('lung_dataframe_overall2.csv',converters={'Combined': lambda x: str(x),'State-county recode_x': lambda x: str(x)})


    lst = index_lookup(df_lung)
    add_radon(lst, df_lung)

    lst_overall = index_lookup(df_lung_overall)
    add_radon(lst_overall, df_lung_overall)

    df_lung.to_csv('lung_dataframe3.csv', index=False)
    df_lung_overall.to_csv('lung_dataframe_overall3.csv', index=False)
