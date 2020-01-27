import numpy as np
import pandas as pd
import pickle,os,csv



def parse(df):
    indices = [(0,4),(4,6),(6,8),(8,11),(11,13),(13,14),(14,15),(15,16),(16,18),(18,26)]
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

#Creates 2 age bins (<69 and >69)
def agebins(processed):
    bins = [0, 14, 18]
    labels = [1,2]
    processed['Age'] = pd.cut(processed['Age'].astype(int),bins=bins, labels=labels)
    processed.Age = processed.Age.fillna(1)
    processed.Age = processed.Age.astype(str)


def col_changes(df_new):
    df_new.columns = clean_df.columns
    df_new['FIPS']=df_new['State FIPS'].apply(lambda x: str(x).zfill(2))+df_new['County FIPS'].apply(lambda x: str(x).zfill(3))
    df_new.Population = df_new.Population.apply(pd.to_numeric)
    df_new.Year = df_new.Year.astype(int)

if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/intermediate_files/'
    clean_df = pd.DataFrame(columns=['Year','State postal abbr','State FIPS','County FIPS','Registry','Race','Origin','Sex','Age','Population'])


    if not os.path.isfile(filepath+'Population.csv'):
        population = pd.read_table('/home/dhense/PublicData/ZNAHealth/Restricted_data/us.1969_2016.19ages.adjusted.txt',header=None)

        # Won't load on my machine without splitting the data into two groups and then concatenating
        population1 = population[:7000000]
        population2 = population[7000000:]
        print(population1.head())

        df1 = parse(population1)
        print('half way!')
        df2 = parse(population2)
        print('Done stage 2!')

        processed = pd.concat([df1, df2])

        processed.to_csv('filepath+Population.csv', index=False)

    pop_pickle = 'pop.pickle'

    if not os.path.isfile(filepath+pop_pickle):
        processed = pd.read_csv(filepath+'Population.csv', names=clean_df.columns)
        processed = processed.drop(processed.index[0])
        col_changes(processed)
        agebins(processed)


        print("...saving pickle")
        tmp = open(filepath+pop_pickle,'wb')
        pickle.dump(processed,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(filepath+pop_pickle,'rb')
        processed = pickle.load(tmp)
        tmp.close()
