import numpy as np
import pandas as pd
import pickle,os,csv

clean_df = pd.DataFrame(columns=['Year','State postal abbr','State FIPS','County FIPS','Registry','Race','Origin','Sex','Age','Population'])

#Group lung cancer incidence data by county, gender & year and get counts of cases per group
#Pivot dataframe by year
def group(df):
    df_grouped = df.groupby(['State-county recode','Sex','Age','Year of diagnosis']).size().reset_index(name='Count')

    df_grouped['Combined'] = df_grouped['State-county recode'].astype(str)+df_grouped.Sex.astype(str)+df_grouped.Age.astype(str)

    df_grouped = df_grouped.pivot(index='Combined',columns='Year of diagnosis',values='Count').reset_index()

    for year in range(1973,2000):
        df_grouped.drop(year, axis=1, inplace=True)

    df_grouped['State-county recode']=df_grouped.Combined.str[:5]
    df_grouped['Sex']=df_grouped.Combined.str[5:6]
    df_grouped['Age']=df_grouped.Combined.str[6:]

    df_grouped.fillna(0,inplace=True)
    return df_grouped

# Grouped without gender
def group_overall(df):
    df_grouped = df.groupby(['State-county recode','Year of diagnosis']).size().reset_index(name='Count')

    df_grouped['Combined'] = df_grouped['State-county recode'].astype(str)

    df_grouped = df_grouped.pivot(index='Combined',columns='Year of diagnosis',values='Count').reset_index()

    for year in range(1973,2000):
        df_grouped.drop(year, axis=1, inplace=True)

    df_grouped['State-county recode']=df_grouped.Combined.str[:5]
    df_grouped.fillna(0,inplace=True)

    return df_grouped

#Group population data by county, gender & year and get population size per group
#Pivot dataframe
def group_pop(df):
    df['Combined'] = df['State-county recode'].astype(str)+df.Sex.astype(str)+df.Age.astype(str)
    df.Year = df.Year.astype(int)
    pop_binned = pd.DataFrame(df.groupby(['Combined','Year'])['Population'].sum().reset_index())
    pop_final = pop_binned.pivot(index='Combined',columns='Year',values='Population').reset_index()

    for year in range(1969,2000):
        pop_final.drop(year, axis=1, inplace=True)

    pop_final['State-county recode']=pop_final.Combined.str[:5]
    pop_final['Sex']=pop_final.Combined.str[5:6]
    pop_final['Age']=pop_final.Combined.str[6:]

    return pop_final

#Grouped without gender
def group_pop_overall(df):

    df['Combined'] = df['State-county recode'].astype(str)
    df.Year = df.Year.astype(int)
    pop_binned = pd.DataFrame(df.groupby(['Combined','Year'])['Population'].sum().reset_index())
    pop_final = pop_binned.pivot(index='Combined',columns='Year',values='Population').reset_index()

    for year in range(1969,2000):
        pop_final.drop(year, axis=1, inplace=True)

    pop_final['State-county recode']=pop_final.Combined.str[:5]

    return pop_final




if __name__=='__main__':
    filepath = '/home/davidhenslovitz/Galvanize/ZNAHealth/'

    print("...loading pickle")
    framed_pickle = 'framed.pickle'
    tmp = open(filepath+framed_pickle,'rb')
    total = pickle.load(tmp)
    tmp.close()

    pop_pickle = 'pop.pickle'
    rates_pickle = 'rates.pickle'
    rates_overall_pickle = 'rates_overall.pickle'


    print("...loading pickle")
    tmp = open(filepath+pop_pickle,'rb')
    processed = pickle.load(tmp)
    tmp.close()


    if not os.path.isfile(filepath+rates_overall_pickle):

        grouped = group(total)
        grouped2 = group_pop(processed)

        grouped_overall = group_overall(total)
        grouped2_overall = group_pop_overall(processed)

        #Combine lung cancer incidence data with population data
        combined = pd.merge(grouped,grouped2,how='left',on='Combined')
        combined_overall = pd.merge(grouped_overall,grouped2_overall,how='left',on='Combined')

        #rates per 100,000
        for i in range(2000,2015):
            combined[i] = (combined[str(i)+"_x"]/combined[str(i)+"_y"])*100000
            combined_overall[i] = (combined_overall[str(i)+"_x"]/combined_overall[str(i)+"_y"])*100000

        #Replace 0's with nans, get rid of counties with null values
        combined = combined.replace(0,np.nan)
        combined.drop(combined[pd.isnull(combined).any(axis=1)].index, inplace=True)

        combined_overall = combined_overall.replace(0,np.nan)
        combined_overall.drop(combined_overall[pd.isnull(combined_overall).any(axis=1)].index, inplace=True)

        print("...saving pickle")
        tmp = open(filepath+rates_pickle,'wb')
        pickle.dump(combined,tmp)
        tmp.close()

        print("...saving pickle")
        tmp = open(filepath+rates_overall_pickle,'wb')
        pickle.dump(combined_overall,tmp)
        tmp.close()
    else:

        print("...loading pickle")
        tmp = open(filepath+rates_pickle,'rb')
        combined = pickle.load(tmp)
        tmp.close()

        print("...loading pickle")
        tmp = open(filepath+rates_overall_pickle,'rb')
        combined_overall = pickle.load(tmp)
        tmp.close()

    print(combined_overall.head())
