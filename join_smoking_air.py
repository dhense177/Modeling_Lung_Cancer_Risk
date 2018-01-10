import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, re



def col_changes(df_lung):
    codes = list(df_lung["State-county recode_x"])
    lst = []
    lst2 = []
    for i in codes:
        lst.append(fips[fips.FIPS==i]['Name'].values)
        lst2.append(fips[fips.FIPS==i]['State'].values)

    df_lung["County"] = lst
    df_lung["County"] = df_lung["County"].str.get(0)
    df_lung["State"] = lst2
    df_lung["State"] = df_lung["State"].str.get(0)
    df_lung["State_and_county"] = df_lung["County"]+" County, "+df_lung["State"]

def index_lookup(df_lung):
    counter = 0
    lst = []
    for i in df_lung["State_and_county"]:
        if (pd.isnull(i) or i not in list(df_both['State & County'].values)):
            lst.append(np.nan)
        else:
            ind = list(df_both['State & County'].values).index(i)
            lst.append(ind)
        counter += 1
    return lst

def index_lookup2(df_lung):
    lst = []
    for i in df_lung['State-county recode_x']:
        if (i not in list(df_pm2new.index)):
            lst.append(np.nan)
        else:
            ind = list(df_pm2new.index).index(i)
            lst.append(ind)
    return lst

def add_smoking(lst):
    arr = []
    for x in lst:
        new_list = []
        for year in range(1996,2013):
            if np.isnan(x):
                new_list.append(np.nan)
            else:
                new_list.append(df_both.iloc[x][1:18][year])
        arr.append(new_list)

    counter = 0
    for yr in df_both.iloc[1][1:18].index:
        df_lung[str(yr)+"_smoking"] = [i[counter] for i in arr]
        counter += 1


def add_air(lst):
    arr = []
    for x in lst:
        new_list = []
        for col in df_pm2new.columns:
            if np.isnan(x):
                new_list.append(np.nan)
            else:
                new_list.append(df_pm2new.iloc[x][col])
        arr.append(new_list)

    counter = 0
    for col in df_pm2new.columns:
        df_lung[str(col)+"_air"] = [i[counter] for i in arr]
        counter += 1



if __name__=='__main__':
    print("...loading pickle")
    tmp = open('rates.pickle','rb')
    df_lung = pickle.load(tmp)
    tmp.close()

    df = pd.read_excel('Smoking/smoking_estimates_means.xlsx')
    df_ci = pd.read_excel('Smoking/smoking_estimates_ci.xlsx')

    df_both = df[df.Sex=='Both']

    fips = pd.read_excel('FIPS.xlsx')
    fips.FIPS = fips.FIPS.apply(lambda x: str(x).zfill(5))

    col_changes(df_lung)

    lst = index_lookup(df_lung)
    add_smoking(lst)

    df_air = pd.read_csv('Air_Quality_Measures_on_the_National_Environmental_Health_Tracking_Network.csv')

    df_pm2 = df_air[df_air.MeasureName=='Annual average ambient concentrations of PM 2.5 in micrograms per cubic meter, based on seasonal averages and daily measurement (monitor and modeled data)']

    df_pm2['CountyFips'] = df_pm2['CountyFips'].apply(lambda x: str(x).zfill(5))

    df_pm2 = df_pm2[['CountyFips', 'ReportYear','Value','UnitName']]
    df_pm2new = df_pm2.pivot(index='CountyFips', columns='ReportYear', values='Value')


    lst = index_lookup2(df_lung)
    add_air(lst)

    df_lung.drop(['Race','Sex_x','State-county recode_y','Race recode (White, Black, Other)','Sex_y', 'County','State','State_and_county'],axis=1, inplace=True)

    df_lung = df_lung.fillna(0)
    df_lung.to_csv('lung_dataframe.csv')
