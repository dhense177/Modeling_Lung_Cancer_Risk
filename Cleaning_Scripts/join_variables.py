import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, re, os
from functools import reduce




#Adjusts smoking dataframe
def smoking_changes(df):
    df_overall = df[df.Sex=='Both']
    df_overall.rename(columns={'State & County':'State_and_county'}, inplace=True)
    return df_overall


def melt(df,dct):
    melted = pd.melt(df, id_vars=['State_and_county'], value_vars=[1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012]).rename(columns=dct)
    return melted


def air_changes(df_air):
    df_air['FIPS'] = df_air['CountyFips'].apply(lambda x: str(x).zfill(5))
    df_air = df_air[['MeasureName', 'FIPS', 'ReportYear','Value','UnitName']]
    df_air.rename(columns={'ReportYear':'Year'},inplace=True)

    #Get measures where no nulls exist
    counts = df_air.MeasureName.value_counts()
    cols = list(counts[counts.values==34199].index)

    col_names = ['PM2.5_level','PM2.5_person-days','ozone_days','ozone_person-days','PM2.5_percent']
    rename_dict = {i:j for i,j in zip(cols,col_names)}

    df_air = pd.DataFrame(df_air.pivot_table('Value',['FIPS','Year'],'MeasureName')).reset_index().rename(columns=rename_dict)[['FIPS','Year','PM2.5_level','PM2.5_person-days','ozone_days','ozone_person-days','PM2.5_percent']]

    df_air.dropna(axis=0,inplace=True)

    df_air = pd.merge(df_air, fips, how='left', on='FIPS')
    df_air.drop('FIPS',axis=1, inplace=True)
    return df_air


def process_radon(df):
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

    df.rename(columns={'activity':'Radon_mean'},inplace=True)

    return df


def process_aqi(total):
    total = pd.merge(df_aqi, mapper, how='left', on='State')
    total['State_and_county'] = total.County + " County, " + total.Postal.astype(str)

    total = total[(total.Year>1999) & (total.Year<2013)]
    total.Year = total.Year.astype(int)

    total.rename(columns={'Days PM2.5':'Days_PM2.5','Days PM10':'Days_PM10','Median AQI':'Median_AQI','Max AQI':'Max_AQI', 'Days with AQI':'Days_with_AQI'},inplace=True)
    return total.reset_index(drop=True)


#Concatenate dataframes, grab relevant columns
def concat_tri(df_dict):
    total = pd.concat(df_dict.values(),ignore_index=True)

    total =total[['YEAR','COUNTY','ST','INDUSTRY_SECTOR_CODE','INDUSTRY_SECTOR','PRIMARY_NAICS','CHEMICAL','METAL','CARCINOGEN','UNIT_OF_MEASURE','5.1_FUGITIVE_AIR','5.2_STACK_AIR','5.3_WATER','5.5.2_LAND_TREATMENT','ON-SITE_RELEASE_TOTAL','TOTAL_RELEASES']]

    total['COUNTY_ST'] = total['COUNTY'].str.title()+" County, "+total['ST']
    return total


#Groups all carcinogens together, sums up quantites (almost all in pounds)
def group_tri(total):
    total = total[total.CARCINOGEN=='YES']
    total = total[(total.YEAR>1999) & (total.YEAR<2013)]
    total = pd.DataFrame(total.groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    total.rename(columns={'YEAR':'Year','COUNTY_ST':'State_and_county','ON-SITE_RELEASE_TOTAL':'Release_Total'},inplace=True)

    return total

if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/'

################### LOAD PICKLE #########################

    print("...loading pickle")
    tmp = open(filepath+'intermediate_files/rates.pickle','rb')
    df_lung = pickle.load(tmp)
    tmp.close()

#########################################################

    fips = pd.read_excel('/home/dhense/PublicData/ZNAHealth/Data_Files/FIPS.xlsx')
    fips.FIPS = fips.FIPS.apply(lambda x: str(x).zfill(5))
    fips['State_and_county'] = fips.Name + ' County, ' + fips.State
    fips = fips[['FIPS','State_and_county']]

#################### SMOKING DATA ########################
    df = pd.read_excel(filepath+'Data_Files/Smoking/smoking_estimates_means.xlsx')
    df_daily = pd.read_excel(filepath+'Data_Files/Smoking/smoking_daily_means.xlsx')

    df_overall = smoking_changes(df)
    df_overall_daily = smoking_changes(df_daily)

    df_overall = melt(df_overall, {'variable':'Year','value':'Smoking'})
    df_overall_daily = melt(df_overall_daily, {'variable':'Year','value':'Smoking_daily'})

    df_smoking = pd.merge(df_overall, df_overall_daily, how='left', on=['State_and_county','Year'])

################### AIR QUALITY DATA ####################
    df_air = pd.read_csv(filepath+'Data_Files/Air_Quality_Measures_on_the_National_Environmental_Health_Tracking_Network.csv')

    df_air = air_changes(df_air)

####################### RADON ###########################

    radon_pickle = 'radon.pickle'
    if not os.path.isfile(filepath+'intermediate_files/'+radon_pickle):
        df1 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs1.dat')
        df2 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs2.dat')
        df3 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs3.dat')
        df4 =  pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs4.dat')
        df5 = pd.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/radon_complete/srrs5.dat')

        df = pd.concat([df1, df2, df3, df4, df5])
        df.columns = df.columns.map(str.strip)

        df = process_radon(df)

        #Using mean county-wide radon activity over years 1988-1992 due to data sparcity
        df_radon = pd.DataFrame(df.groupby(['State_and_county'])['Radon_mean'].mean()).reset_index()
        print("...saving pickle")
        tmp = open(filepath+'intermediate_files/'+radon_pickle,'wb')
        pickle.dump(df_radon,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(filepath+'intermediate_files/'+radon_pickle,'rb')
        df_radon = pickle.load(tmp)
        tmp.close()

######################### AQI ###########################

    mapper = pd.read_csv('/home/dhense/PublicData/ZNAHealth/Data_Files/State_to_county.csv')
    x = os.listdir(path=filepath+'Data_Files/EPA_AIRDATA/')

    df_dict = {}
    for textfile in x:
        df_dict['df_'+textfile[21:25]+'AQI'] = pd.read_csv(filepath+'Data_Files/EPA_AIRDATA/'+textfile)

    df_aqi = pd.concat(df_dict.values(),ignore_index=True)

    df_aqi = process_aqi(df_aqi)

    df_aqi = df_aqi[['State_and_county', 'Year', 'Days_PM2.5','Days_PM10','Median_AQI','Max_AQI', 'Days_with_AQI']]


######################## TRI #############################
    tri_filepath = filepath+'Data_Files/EPA_TRI/'
    pickle_path = filepath+'intermediate_files/'
    x = os.listdir(path=tri_filepath)

    tri_pickle = 'tri.pickle'

    if not os.path.isfile(pickle_path+tri_pickle):
        df_dict = {}
        for textfile in x:
            df_dict['df_'+textfile[6:8]] = pd.read_csv(tri_filepath+textfile)
        total = concat_tri(df_dict)
        print("...saving pickle")
        tmp = open(pickle_path+tri_pickle,'wb')
        pickle.dump(total,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(pickle_path+tri_pickle,'rb')
        total = pickle.load(tmp)
        tmp.close()

    df_tri = group_tri(total)

    county_counts = df_tri['State_and_county'].value_counts()
    counties = county_counts[county_counts.values==county_counts.values.max()].index

    df_tri = df_tri[df_tri['State_and_county'].isin(counties)].reset_index(drop=True)

############## MERGE DATAFRAMES TOGETHER ################
    data_frames = [df_lung, df_smoking, df_aqi]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['State_and_county','Year'], how='left'), data_frames)

    #df_radon has no year so must merge separately
    df_merged = pd.merge(df_merged, df_radon, how='left', on='State_and_county')

    #Limit year to 2000-2012
    df_merged = df_merged[(df_merged.Year<2013)]

    df_merged = df_merged[df_merged.isnull().any(axis=1)==False]

    county_counts = df_merged['State_and_county'].value_counts()
    counties = county_counts[county_counts.values==county_counts.values.max()].index
    df_merged = df_merged[df_merged['State_and_county'].isin(counties)].reset_index(drop=True)

################# SAVE AS PICKLE #######################
    merged_pickle = 'merged.pickle'

    print("...saving pickle")
    tmp = open(filepath+'intermediate_files/'+merged_pickle,'wb')
    pickle.dump(df_merged,tmp)
    tmp.close()

'''

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
'''
