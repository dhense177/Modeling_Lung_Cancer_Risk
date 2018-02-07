import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, re


#Map each recode value in df_lung to FIPS mapping file, make columns for county, state and both
def col_changes(df_lung):
    codes = list(df_lung["fips"])
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


#Adjusts smoking dataframe
def smoking_changes(df):
    df_overall = df[df.Sex=='Both']

    lst = adjust_fips(fips, df_overall)
    df_overall['State-county recode'] = lst
    df_overall['Gender'] = ['0' for i in df_overall['State-county recode']]
    df_overall['Combined'] = df_overall['State-county recode']
    df_overall['State_and_county'] = df_overall['State & County']
    return df_overall

def air_changes(df_air):
    df_air['CountyFips'] = df_air['CountyFips'].apply(lambda x: str(x).zfill(5))
    df_air = df_air[['MeasureName', 'CountyFips', 'ReportYear','Value','UnitName']]
    df_air['fips'] = df_air['CountyFips']
    df_air['variable'] = df_air['ReportYear']
    return df_air

if __name__=='__main__':

    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/'

    print("...loading pickle")
    tmp = open(filepath+'final_rates.pickle','rb')
    df_lung_overall = pickle.load(tmp)
    tmp.close()

    #Read in smoking data
    df = pd.read_excel(filepath+'Data_Files/Smoking/smoking_estimates_means.xlsx')
    df_daily = pd.read_excel(filepath+'Data_Files/Smoking/smoking_daily_means.xlsx')

    #Read in FIPS mapper
    fips = pd.read_excel(filepath+'Data_Files/FIPS.xlsx')
    fips.FIPS = fips.FIPS.apply(lambda x: str(x).zfill(5))
    fips['State & County'] = fips.Name + ' County, ' + fips.State


    df_lung_overall['fips'] = df_lung_overall['State-county recode_x']
    df_lung_overall = df_lung_overall.iloc[:,-16:]

    df_overall = smoking_changes(df)
    df_overall_daily = smoking_changes(df_daily)

    #Rename cancer rates columns 2000-2014
    for i in range(2000,2015):
        df_lung_overall[i] = df_lung_overall[str(i)+'cancer_rate']
    df_lung_overall = df_lung_overall.iloc[:,-16:]


    #Unpivot dataframes
    df_lung_overall = pd.melt(df_lung_overall, id_vars=['fips'], value_vars=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014])

    df_overall = pd.melt(df_overall, id_vars=['State_and_county'], value_vars=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012])

    df_overall_daily = pd.melt(df_overall_daily, id_vars=['State_and_county'], value_vars=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012])


    col_changes(df_lung_overall)

    #Merge smoking/daily smoking data with lung dataframe
    df_lung_overall = pd.merge(df_lung_overall, df_overall, how='left', on=['State_and_county','variable'])

    df_lung_overall = pd.merge(df_lung_overall, df_overall_daily, how='left', on=['State_and_county','variable'])


    #Read in Air Quality Data
    df_air = pd.read_csv(filepath+'Data_Files/Air_Quality_Measures_on_the_National_Environmental_Health_Tracking_Network.csv')

    df_air = air_changes(df_air)



    # Try a few metrics out here
    df_pm2 = df_air[df_air.MeasureName=='Annual average ambient concentrations of PM 2.5 in micrograms per cubic meter, based on seasonal averages and daily measurement (monitor and modeled data)']

    df_ozone = df_air[df_air.MeasureName=='Number of days with maximum 8-hour average ozone concentration over the National Ambient Air Quality Standard (monitor and modeled data)']

    #Merge metrics into dataframe
    df_lung_overall = pd.merge(df_lung_overall, df_pm2, how='left', on=['fips','variable'])

    df_lung_overall = pd.merge(df_lung_overall, df_ozone, how='left', on=['fips','variable'])


    #Drop columns and nulls
    df_lung_overall.drop(['MeasureName_x', 'CountyFips_x', 'ReportYear_x','UnitName_x', 'MeasureName_y', 'CountyFips_y','ReportYear_y', 'UnitName_y'],axis=1, inplace=True)

    df_lung_overall.drop(df_lung_overall[pd.isnull(df_lung_overall).any(axis=1)].index, inplace=True)


    df_lung_overall = df_lung_overall.rename(index=str, columns={'variable':'year','value_x':'cancer_incidence','value_y':'smoking','value':'smoking_daily','Value_x':'pm25','Value_y':'ozone'})

    #Export to csv
    df_lung_overall.to_csv(filepath+'lung.csv', index=False)
