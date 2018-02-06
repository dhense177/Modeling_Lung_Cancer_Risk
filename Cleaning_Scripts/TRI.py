import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os

#Concatenate dataframes, grab relevant columns
def concat(df_dict):
    total = pd.concat(df_dict.values(),ignore_index=True)

    total =total[['YEAR','COUNTY','ST','INDUSTRY_SECTOR_CODE','INDUSTRY_SECTOR','PRIMARY_NAICS','CHEMICAL','METAL','CARCINOGEN','UNIT_OF_MEASURE','5.1_FUGITIVE_AIR','5.2_STACK_AIR','5.3_WATER','5.5.2_LAND_TREATMENT','ON-SITE_RELEASE_TOTAL','TOTAL_RELEASES']]

    total['COUNTY_ST'] = total['COUNTY'].str.title()+" County, "+total['ST']
    return total

# Find index of county from lung dataframe in chemicals dataframe
def index_lookup(df_lung):
    lst = []
    for i in df_lung['State_and_county']:
        if (i not in list(total.COUNTY_ST.values)):
            lst.append(np.nan)
        else:
            ind = list(total.COUNTY_ST.values).index(i)
            lst.append(ind)
    return lst

# Adds county-level carconogen data to lung dataframe
def add_chems(lst):
    arr = []
    for x in lst:
        new_list = []
        for year in range(1996,2011):
            if np.isnan(x):
                new_list.append(np.nan)
            else:
                new_list.append(total.iloc[x][1:16][year])
        arr.append(new_list)

    counter = 0
    for yr in total.iloc[1][1:16].index:
        df_lung_overall[str(yr)+"_chems"] = [i[counter] for i in arr]
        counter += 1

#Groups all carcinogens together, sums up quantites (almost all in pounds)
def group(total):
    total = total[total.CARCINOGEN=='YES']
    total = total[total.YEAR>1995]
    total = pd.DataFrame(total.groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    total['State_and_county'] = total['COUNTY_ST']
    total['year'] = total['YEAR']

    return total

if __name__=='__main__':
    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/'
    x = os.listdir(path=filepath+'Data_Files/EPA_TRI/')

    tri_pickle = 'tri.pickle'

    if not os.path.isfile(filepath+tri_pickle):
        df_dict = {}
        for textfile in x:
            df_dict['df_'+textfile[6:8]] = pd.read_csv(filepath+textfile)
        total = concat(df_dict)
        print("...saving pickle")
        tmp = open(filepath+tri_pickle,'wb')
        pickle.dump(total,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(filepath+tri_pickle,'rb')
        total = pickle.load(tmp)
        tmp.close()

    total = group(total)

    #Drop columns with any Nans
    total.drop(total[pd.isnull(total).any(axis=1)].index, inplace=True)

    df_lung_overall = pd.read_csv(filepath+'lung_aqi.csv',converters={'Combined': lambda x: str(x),'State-county recode_x': lambda x: str(x)})
    lst = index_lookup(df_lung_overall)

    df_lung_overall = pd.merge(df_lung_overall, total, how='left', on=['State_and_county','year'])

    #Drop nulls
    df_lung_overall.drop(df_lung_overall[pd.isnull(df_lung_overall).any(axis=1)].index, inplace=True)

    #Export to csv
    df_lung_overall.to_csv(filepath+'lung_tri.csv', index=False)
