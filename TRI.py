import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os


def concat(df_dict):
    total = pd.concat(df_dict.values(),ignore_index=True)

    total =total[['YEAR','COUNTY','ST','INDUSTRY_SECTOR_CODE','INDUSTRY_SECTOR','PRIMARY_NAICS','CHEMICAL','METAL','CARCINOGEN','UNIT_OF_MEASURE','5.1_FUGITIVE_AIR','5.2_STACK_AIR','5.3_WATER','5.5.2_LAND_TREATMENT','ON-SITE_RELEASE_TOTAL','TOTAL_RELEASES']]

    total['COUNTY_ST'] = total['COUNTY'].str.title()+" County, "+total['ST']
    return total

def index_lookup(df_lung):
    lst = []
    for i in df_lung['State_and_county']:
        if (i not in list(total_chems.COUNTY_ST.values)):
            lst.append(np.nan)
        else:
            ind = list(total_chems.COUNTY_ST.values).index(i)
            lst.append(ind)
    return lst

def add_chems(lst):
    arr = []
    for x in lst:
        new_list = []
        for year in range(1987,2014):
            if np.isnan(x):
                new_list.append(np.nan)
            else:
                new_list.append(total_chems.iloc[x][1:29][year])
        arr.append(new_list)

    counter = 0
    for yr in total_chems.iloc[1][1:28].index:
        df_lung[str(yr)+"_chems"] = [i[counter] for i in arr]
        counter += 1

if __name__=='__main__':
    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/EPA_TRI/'
    x = os.listdir(path=filepath)

    tri_pickle = 'tri.pickle'

    if not os.path.isfile(tri_pickle):
        df_dict = {}
        for textfile in x:
            df_dict['df_'+textfile[6:8]] = pd.read_csv(filepath+textfile)
        total = concat(df_dict)
        print("...saving pickle")
        tmp = open(tri_pickle,'wb')
        pickle.dump(total,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(tri_pickle,'rb')
        total = pickle.load(tmp)
        tmp.close()

    total = total[total.CHEMICAL.isin(['ARSENIC','ARSENIC COMPOUNDS','ASBESTOS (FRIABLE)','CADMIUM','CADMIUM COMPOUNDS'])]

    arsenic = pd.DataFrame(total[total.CHEMICAL=='ARSENIC'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    arsenic_comp = pd.DataFrame(total[total.CHEMICAL=='ARSENIC COMPOUNDS'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    asbestos = pd.DataFrame(total[total.CHEMICAL=='ASBESTOS (FRIABLE)'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())

    cadmium = pd.DataFrame(total[total.CHEMICAL=='CADMIUM'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    cadmium_comp = pd.DataFrame(total[total.CHEMICAL=='CADMIUM COMPOUNDS'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())



    arsenic = arsenic.pivot(index='COUNTY_ST',columns='YEAR',values='ON-SITE_RELEASE_TOTAL').reset_index()

    arsenic_comp = arsenic_comp.pivot(index='COUNTY_ST',columns='YEAR',values='ON-SITE_RELEASE_TOTAL').reset_index()

    asbestos = asbestos.pivot(index='COUNTY_ST',columns='YEAR',values='ON-SITE_RELEASE_TOTAL').reset_index()

    cadmium = cadmium.pivot(index='COUNTY_ST',columns='YEAR',values='ON-SITE_RELEASE_TOTAL').reset_index()

    cadmium_comp = cadmium_comp.pivot(index='COUNTY_ST',columns='YEAR',values='ON-SITE_RELEASE_TOTAL').reset_index()

    total_chems = pd.concat([arsenic, arsenic_comp, asbestos, cadmium, cadmium_comp])

    df_lung = pd.read_csv('lung_dataframe.csv',converters={'Combined': lambda x: str(x),'State-county recode_x': lambda x: str(x)})
    lst = index_lookup(df_lung)

    add_chems(lst)

    grouped_df = df_lung.groupby('State-county recode_x')[df_lung.columns].sum()
    #grouped_df[grouped_df.isnull().any(axis=1)]
