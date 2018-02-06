import numpy as np
import pandas as pd
import pickle,os,csv

clean_df = pd.DataFrame(columns=['Year','State postal abbr','State FIPS','County FIPS','Registry','Race','Origin','Sex','Age','Population'])


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


if __name__=='__main__':

    population = pd.read_table('/home/davidhenslovitz/Galvanize/ZNAHealth/us.1969_2016.19ages.adjusted.txt',header=None)

    # Won't load on my machine without splitting the data into two groups and then concatenating
    population1 = population[:7000000]
    population2 = population[7000000:]
    print(population1.head())

    df1 = parse(population1)
    print('half way!')
    df2 = parse(population2)
    print('Done stage 2!')

    processed = pd.concat([df1, df2])

    processed.to_csv('/home/davidhenslovitz/Galvanize/ZNAHealth/population.csv', index=False)
