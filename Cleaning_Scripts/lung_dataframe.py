import numpy as np
import pandas as pd
import pickle,os,csv

def frame(filepath):
    df_lung = pd.read_csv(filepath, converters={'State-county recode': lambda x: str(x), 'CS Schema v0204+': lambda x: str(x)})

    df_lung = df_lung.replace(r'\s+', np.nan, regex=True)

    #Filter to get only lung cancer
    df_lung = df_lung[df_lung['CS Schema v0204+']=='063']
    df_lung.drop('CS Schema v0204+',axis=1,inplace=True)

    #Select relevant columns
    df_lung = df_lung[['Race/Ethnicity','Sex','Age at diagnosis', 'Year of diagnosis', 'Year of Birth','Race recode (White, Black, Other)','Race recode (W, B, AI, API)','Origin recode NHIA (Hispanic, Non-Hisp)','State-county recode']]

    #Rename Year at diagnosis to Year, State-county recode to FIPS
    df_lung.rename(columns={'Year of diagnosis':'Year','State-county recode':'FIPS'}, inplace=True)

    #Remove rows where age=999
    df_lung = df_lung[df_lung['Age at diagnosis']!=999]

    #Create age buckets
    median = df_lung['Age at diagnosis'].median()
    df_lung.loc[df_lung['Age at diagnosis'].apply(pd.to_numeric) <= median, 'Age'] = '1'
    df_lung.loc[df_lung['Age at diagnosis'].apply(pd.to_numeric) > median, 'Age'] = '2'

    #Turn all Race identifiers greater than 3 to category 3 (other)
    df_lung.loc[df_lung['Race recode (White, Black, Other)'].apply(pd.to_numeric)>3,'Race recode (White, Black, Other)']=3

    return df_lung



if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/intermediate_files/'

    framed_pickle = 'framed.pickle'

    if not os.path.isfile(filepath+framed_pickle):
        total = frame(filepath+'RESPIR.csv')
        print("...saving pickle")
        tmp = open(filepath+framed_pickle,'wb')
        pickle.dump(total,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(filepath+framed_pickle,'rb')
        total = pickle.load(tmp)
        tmp.close()


    print(total.shape)
    print(total.head())
