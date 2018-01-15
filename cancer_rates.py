import numpy as np
import pandas as pd
import pickle,os,csv

clean_df = pd.DataFrame(columns=['Year','State postal abbr','State FIPS','County FIPS','Registry','Race','Origin','Sex','Age','Population'])

#Group lung cancer incidence data by county, gender & year and get counts of cases per group
#Pivot dataframe by year for time series analysis
def group(df):
    df_grouped = df.groupby(['State-county recode','Sex','Year of diagnosis']).size().reset_index(name='Count')

    df_grouped['Combined'] = df_grouped['State-county recode'].astype(str)+df_grouped.Sex.astype(str)

    df_grouped = df_grouped.pivot(index='Combined',columns='Year of diagnosis',values='Count').reset_index()

    for year in range(1973,2000):
        df_grouped.drop(year, axis=1, inplace=True)

    df_grouped['State-county recode']=df_grouped.Combined.str[:5]
    df_grouped['Sex']=df_grouped.Combined.str[5:]

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
    df_grouped['Sex']=df_grouped.Combined.str[5:]

    df_grouped.fillna(0,inplace=True)
    return df_grouped

#Group population data by county, gender & year and get population size per group
#Pivot dataframe for time series analysis
def group_pop(df):
    pop_binned = pd.DataFrame(df.groupby(['State-county recode','Sex','Year'])['Population'].sum().reset_index())
    pop_binned['Combined'] = pop_binned['State-county recode'].astype(str)+pop_binned.Sex.astype(str)
    pop_binned.Year = pop_binned.Year.astype(int)
    pop_final = pop_binned.pivot(index='Combined',columns='Year',values='Population').reset_index()

    for year in range(1973,2000):
        pop_final.drop(year, axis=1, inplace=True)

    pop_final['State-county recode']=pop_final.Combined.str[:5]
    pop_final['Sex']=pop_final.Combined.str[5:]
    return pop_final

#Grouped without gender
def group_pop_overall(df):
    pop_binned = pd.DataFrame(df.groupby(['State-county recode','Year'])['Population'].sum().reset_index())
    pop_binned['Combined'] = pop_binned['State-county recode'].astype(str)
    pop_binned.Year = pop_binned.Year.astype(int)
    pop_final = pop_binned.pivot(index='Combined',columns='Year',values='Population').reset_index()

    for year in range(1973,2000):
        pop_final.drop(year, axis=1, inplace=True)

    pop_final['State-county recode']=pop_final.Combined.str[:5]
    pop_final['Sex']=pop_final.Combined.str[5:]
    return pop_final


def parse(df):
    indices = [(0,4),(4,6),(6,8),(8,11),(11,13),(13,14),(14,15),(15,16),(16,18),(18,28)]
    dct={}
    for num in range(len(df)):
        string = ""
        x = df.iloc[num][0]
        for i in range(len(indices)):
            string += x[indices[i][0]:indices[i][1]] + ", "
            dct[num] = string[:-2]
    for i, k in enumerate(dct.values()):
        dct[i] = k.split(', ')

    df_parsed = pd.DataFrame.from_dict(dct, orient='index')
    return df_parsed

#Creates 3 age bins
def agebins(df_new):
    df_new.loc[ df_new.Age.apply(pd.to_numeric) <= 30, 'Age'] = 0
    df_new.loc[(df_new.Age.apply(pd.to_numeric) > 30) & (df_new.Age.apply(pd.to_numeric) <= 60), 'Age'] = 1,'Sex'
    df_new.loc[ df_new.Age.apply(pd.to_numeric) > 60, 'Age'] = 2

def col_changes(df_new):
    df_new.columns = clean_df.columns
    df_new['State-county recode']=df_new['State FIPS']+df_new['County FIPS']
    df_new.Population = df_new.Population.apply(pd.to_numeric)

if __name__=='__main__':
    print("...loading pickle")
    framed_pickle = 'framed.pickle'
    tmp = open(framed_pickle,'rb')
    total = pickle.load(tmp)
    tmp.close()

    pop_pickle = 'pop.pickle'
    rates_pickle = 'rates.pickle'
    rates_overall_pickle = 'rates_overall.pickle'

    if not os.path.isfile(pop_pickle):
        population = pd.read_table('/home/davidhenslovitz/Galvanize/ZNAHealth/SEER_1973_2014_TEXTDATA/populations/white_black_other/yr1973_2014.seer9/singleages.txt',header=None)

        # Won't load on my machine without splitting the data into two groups and then concatenating
        population1 = population[:1422890]
        population2 = population[1422890:]

        df1 = parse(population1)
        df2 = parse(population2)

        processed = pd.concat([df1, df2])
        col_changes(processed)
        agebins(processed)
        print("...saving pickle")
        tmp = open(pop_pickle,'wb')
        pickle.dump(processed,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(pop_pickle,'rb')
        processed = pickle.load(tmp)
        tmp.close()

    if not os.path.isfile(rates_pickle):
        grouped = group(total)
        grouped2 = group_pop(processed)

        #Combine lung cancer incidence data with population data
        combined = pd.merge(grouped2,grouped,how='left',on='Combined')

        grouped_overall = group_overall(total)
        grouped2_overall = group_pop_overall(processed)

        #Combine lung cancer incidence data with population data
        combined_overall = pd.merge(grouped2_overall,grouped_overall,how='left',on='Combined')
        for i in range(2000,2015):
            combined[i] = combined[str(i)+"_y"]/combined[str(i)+"_x"]
            combined_overall[i] = combined_overall[str(i)+"_y"]/combined_overall[str(i)+"_x"]
        combined.drop(combined[pd.isnull(combined).any(axis=1)].index, inplace=True)
        combined.drop(combined[combined['State-county recode_x']=='15005'].index,
        inplace=True)

        combined_overall.drop(combined_overall[pd.isnull(combined_overall).any(
        axis=1)].index, inplace=True)

        print("...saving pickle")
        tmp = open(rates_pickle,'wb')
        pickle.dump(combined,tmp)
        tmp.close()

        print("...saving pickle")
        tmp = open(rates_overall_pickle,'wb')
        pickle.dump(combined_overall,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(rates_pickle,'rb')
        combined = pickle.load(tmp)
        tmp.close()

        print("...loading pickle")
        tmp = open(rates_overall_pickle,'rb')
        combined_overall = pickle.load(tmp)
        tmp.close()

    print(combined_overall.head())
