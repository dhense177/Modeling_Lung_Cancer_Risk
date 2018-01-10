import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os


def concat(df_dict):
    total = pd.concat(df_dict.values(),ignore_index=True)
    total = total[total.CHEMICAL.isin(['ARSENIC','ARSENIC COMPOUNDS','ASBESTOS (FRIABLE)'])]
    total =total[['YEAR','COUNTY','ST','INDUSTRY_SECTOR_CODE','INDUSTRY_SECTOR','PRIMARY_NAICS','CHEMICAL','METAL','CARCINOGEN','UNIT_OF_MEASURE','5.1_FUGITIVE_AIR','5.2_STACK_AIR','5.3_WATER','5.5.2_LAND_TREATMENT','ON-SITE_RELEASE_TOTAL','TOTAL_RELEASES']]

    total['COUNTY_ST'] = total['COUNTY'].str.title()+" County, "+total['ST']
    return total






















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


    arsenic = pd.DataFrame(total[total.CHEMICAL=='ARSENIC'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    arsenic_comp = pd.DataFrame(total[total.CHEMICAL=='ARSENIC COMPOUNDS'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())
    asbestos = pd.DataFrame(total[total.CHEMICAL=='ASBESTOS (FRIABLE)'].groupby(['YEAR','COUNTY_ST'])['ON-SITE_RELEASE_TOTAL'].sum().reset_index())


    arsenic = arsenic.pivot(index='COUNTY_ST',columns='YEAR',values='ON-SITE_RELEASE_TOTAL').reset_index()
