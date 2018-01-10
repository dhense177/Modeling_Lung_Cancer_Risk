import numpy as np
import pandas as pd
import pickle,os,csv

#Create unfitered dataframe
def unfiltered(df_lung):
    unfiltered = df_lung.groupby('State-county recode_x')[df_lung.columns].sum()
    unfiltered = unfiltered.loc[(unfiltered!=0).all(axis=1)]  #get rid of rows with any zeros

#create dataframe broken out by gender
def gender(df_lung):
    filtered_gender = df_lung.copy()
    filtered_gender['Combined'] = df_lung.Combined.str[0:5]+df_lung.Combined.str[6]
    filtered_gender = filtered_gender.groupby('Combined')[filtered_gender.columns].sum()
    filtered_gender = filtered_gender.loc[(filtered_gender!=0).all(axis=1)]

    for i in filtered_gender.index.str[0:5]:
        if i+'1' not in filtered_gender.index:
            filtered_gender.drop(i+'2',axis=0, inplace=True)
        elif i+'2' not in filtered_gender.index:
            filtered_gender.drop(i+'1',axis=0, inplace=True)
    return filtered_gender

#create dataframe broken out by race
def race(df_lung):
    filtered_race = df_lung.copy()
    filtered_race['Combined'] = df_lung.Combined.str[0:6]
    filtered_race = filtered_race.groupby('Combined')[filtered_race.columns].sum()
    filtered_race = filtered_race.loc[(filtered_race!=0).all(axis=1)]

    for i in filtered_race.index.str[0:5]:
        if (i+'1' not in filtered_race.index) or (i+'2' not in filtered_race.index) or (i+'3' not in filtered_race.index):
            if i+'1' in filtered_race.index:
                filtered_race.drop(i+'1',axis=0, inplace=True)
            if i+'2' in filtered_race.index:
                filtered_race.drop(i+'2',axis=0, inplace=True)
            if i+'3' in filtered_race.index:
                filtered_race.drop(i+'3',axis=0, inplace=True)
    return filtered_race

if __name__=='__main__':
    df_lung = pd.read_csv('lung_dataframe.csv',converters={'Combined': lambda x: str(x)})

    gender_df = gender(df_lung)
    race_df = race(df_lung)

    print(gender_df.shape)
