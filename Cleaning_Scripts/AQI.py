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
    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/EPA_AIRDATA/'
    x = os.listdir(path=filepath)

    mapper = pd.read_csv('State_to_county.csv')

    df_dict = {}
    for textfile in x:
        df_dict['df_'+textfile[21:25]+'AQI'] = pd.read_csv(filepath+textfile)

    total = pd.concat(df_dict.values(),ignore_index=True)

    total = total[total.Year>1995]

    process(total)

    total = total[['State_and_county', 'year', 'Days PM2.5','Median AQI','Max AQI']]


    df_lung_overall = pd.read_csv('lung_radon.csv',converters={'Combined': lambda x: str(x)})
    df_lung_overall = pd.merge(df_lung_overall, total, how='left', on=['State_and_county','year'])
    df_lung_overall.drop(df_lung_overall[pd.isnull(df_lung_overall).any(axis=1)].index, inplace=True)


    df_lung_overall.to_csv('lung_aqi.csv', index=False)
    # total_pm25 = total.pivot(index='State_and_county',columns='Year',values='Days PM2.5').reset_index()
    # median_aqi = total.pivot(index='State_and_county',columns='Year',values='Median AQI').reset_index()
    # max_aqi = total.pivot(index='State_and_county',columns='Year',values='Max AQI').reset_index()
    # unhealthy_days = total.pivot(index='State_and_county',columns='Year',values='Unhealthy Days').reset_index()
    # very_unhealthy_days = total.pivot(index='State_and_county',columns='Year',values='Very Unhealthy Days').reset_index()
    # hazardous_days = total.pivot(index='State_and_county',columns='Year',values='Hazardous Days').reset_index()
    #
    #
    # adjust_columns(total_pm25, '_pm25_AQI')
    # adjust_columns(median_aqi, 'MED_AQI')
    # adjust_columns(max_aqi, 'MAX_AQI')
    # adjust_columns(unhealthy_days, 'Unhealthy_days')
    # adjust_columns(very_unhealthy_days, 'Very_unhealthy_days')
    # adjust_columns(hazardous_days, 'Hazardous_days')
    #
    #
    #
    # # #Join with lung dataframe
    # df_lung_overall = pd.read_csv('lung_radon.csv',converters={'Combined': lambda x: str(x)})
    #
    # df_lung_overall = pd.merge(df_lung_overall, total_pm25,how='left',on='State_and_county')
    # df_lung_overall = pd.merge(df_lung_overall, median_aqi,how='left',on='State_and_county')
    # df_lung_overall = pd.merge(df_lung_overall, max_aqi,how='left',on='State_and_county')
    # # df_lung_overall = pd.merge(df_lung_overall, unhealthy_days,how='left',on='State_and_county')
    # # df_lung_overall = pd.merge(df_lung_overall, very_unhealthy_days,how='left',on='State_and_county')
    # # df_lung_overall = pd.merge(df_lung_overall, hazardous_days,how='left',on='State_and_county')
    #
    # df_lung_overall.drop(df_lung_overall[pd.isnull(df_lung_overall).any(axis=1)].index, inplace=True)
    #
