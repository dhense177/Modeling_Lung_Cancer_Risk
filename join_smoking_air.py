import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, re


#Map each recode value in df_lung to FIPS mapping file, make columns for county, state and both
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

#Maps fips codes onto lung dataframe broken out by gender
def adjust_fips(fips, df_both):
    lst = []
    for i in df_both['State & County']:
        if i not in list(fips['State & County'].values):
            lst.append('0')
        else:
            lst.append(fips.iloc[list(fips['State & County'].values).index(i)]['FIPS'])
    return lst


#Finds index of state + county combo (combined column) in smoking estimates file
def index_lookup_overall(df_lung):
    counter = 0
    lst = []
    for i in df_lung["State_and_county"]:
        if (pd.isnull(i) or i not in list(df_overall['State & County'].values)):
            lst.append(np.nan)
        else:
            ind = list(df_overall['State & County'].values).index(i)
            lst.append(ind)
        counter += 1
    return lst

#Merge instead?
#Finds index of st + county + gender combo (combined column) in smoking estimates file
def index_lookup(df_lung, df_both):
    lst = []
    for i in df_lung["Combined"]:
        if (pd.isnull(i) or i not in list(df_both['Combined'].values)):
            lst.append(np.nan)
        else:
            ind = list(df_both['Combined'].values).index(i)
            lst.append(ind)
    return lst

#Finds index of state-county recode number in air quality data
def index_lookup2(df_lung, df_pm2new):
    lst = []
    for i in df_lung['State-county recode_x']:
        if (i not in list(df_pm2new.index)):
            lst.append(np.nan)
        else:
            ind = list(df_pm2new.index).index(i)
            lst.append(ind)
    return lst

#Adds smoking data to lung dataframes
def add_smoking(lst, df_both, df_lung, name):
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
        df_lung[str(yr)+name] = [i[counter] for i in arr]
        counter += 1

#Adds air quality data to lung dataframes
def add_air(lst, df_lung, df_pm2new, name):
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
        df_lung[str(col)+name] = [i[counter] for i in arr]
        counter += 1

def smoking_changes(df):
    df_overall = df[df.Sex=='Both']
    df_both = df[df.Sex!='Both']

    lst = adjust_fips(fips, df_both)
    df_both['State-county recode'] = lst
    df_both['Gender'] = ['1' if i=='Male' else '2' for i in df_both.Sex]
    df_both['Combined'] = df_both['State-county recode']+df_both['Gender']

    lst = adjust_fips(fips, df_overall)
    df_overall['State-county recode'] = lst
    df_overall['Gender'] = ['0' for i in df_overall['State-county recode']]
    df_overall['Combined'] = df_overall['State-county recode']
    return df_both, df_overall

if __name__=='__main__':
    print("...loading pickle")
    tmp = open('rates.pickle','rb')
    df_lung = pickle.load(tmp)
    tmp.close()

    print("...loading pickle")
    tmp = open('final_rates.pickle','rb')
    df_lung_overall = pickle.load(tmp)
    tmp.close()

    df = pd.read_excel('Smoking/smoking_estimates_means.xlsx')
    df_daily = pd.read_excel('Smoking/smoking_daily_means.xlsx')



    fips = pd.read_excel('FIPS.xlsx')
    fips.FIPS = fips.FIPS.apply(lambda x: str(x).zfill(5))
    fips['State & County'] = fips.Name + ' County, ' + fips.State

    df_both, df_overall = smoking_changes(df)
    df_both_daily, df_overall_daily = smoking_changes(df_daily)




    col_changes(df_lung)
    col_changes(df_lung_overall)

    lst = index_lookup(df_lung, df_both)
    add_smoking(lst, df_both, df_lung, '_smoking_total')

    lst2 = index_lookup(df_lung, df_both_daily)
    add_smoking(lst2, df_both_daily, df_lung, '_smoking_daily')

    lst_overall = index_lookup_overall(df_lung_overall)
    add_smoking(lst_overall, df_overall, df_lung_overall, 'smoking_total')

    lst_overall2 = index_lookup_overall(df_lung_overall)
    add_smoking(lst_overall2, df_overall_daily, df_lung_overall, 'smoking_daily')

    df_air = pd.read_csv('Air_Quality_Measures_on_the_National_Environmental_Health_Tracking_Network.csv')



    df_air['CountyFips'] = df_air['CountyFips'].apply(lambda x: str(x).zfill(5))
    df_air = df_air[['MeasureName', 'CountyFips', 'ReportYear','Value','UnitName']]

    # Try a few metrics out here
    df_pm2 = df_air[df_air.MeasureName=='Annual average ambient concentrations of PM 2.5 in micrograms per cubic meter, based on seasonal averages and daily measurement (monitor and modeled data)']

    df_ozone = df_air[df_air.MeasureName=='Number of days with maximum 8-hour average ozone concentration over the National Ambient Air Quality Standard (monitor and modeled data)']

    df_pm2new = df_pm2.pivot(index='CountyFips', columns='ReportYear', values='Value')
    df_ozonenew = df_ozone.pivot(index='CountyFips', columns='ReportYear', values='Value')

    lst = index_lookup2(df_lung, df_pm2new)
    add_air(lst, df_lung, df_pm2new, "PM2.5")

    lst2 = index_lookup2(df_lung, df_ozonenew)
    add_air(lst2, df_lung, df_ozonenew, "Ozone")


    lst_overall = index_lookup2(df_lung_overall, df_pm2new)
    add_air(lst_overall, df_lung_overall, df_pm2new, "PM2.5")

    lst_overall2 = index_lookup2(df_lung_overall, df_ozonenew)
    add_air(lst_overall2, df_lung_overall, df_ozonenew, "Ozone")

    # df_lung.drop(['State-county recode_y','Sex', 'County','State'],axis=1, inplace=True)

    # df_lung_overall.drop(['State-county recode_y','Sex_x','Sex_y', 'County','State'],axis=1, inplace=True)


    df_lung.drop(df_lung[pd.isnull(df_lung).any(axis=1)].index, inplace=True)
    df_lung.to_csv('lung_dataframe.csv', index=False)

    df_lung_overall.drop(df_lung_overall[pd.isnull(df_lung_overall).any(axis=1)].index, inplace=True)
    df_lung_overall.to_csv('lung_dataframe_overall.csv', index=False)
