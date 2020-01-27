import numpy as np
import pandas as pd
import pickle,os,csv



#Group lung cancer incidence data by county, gender, age & year and get counts of cases per group
def group(df):
    df_grouped = df.groupby(['FIPS','Sex','Age','Year']).size().reset_index(name='Count')

    #Remove years prior to 2000
    df_grouped = df_grouped.loc[df_grouped['Year']>1999].reset_index(drop=True)

    return df_grouped


#Group population data by county, gender & year and get population size per group
def group_pop(df):
    pop_binned = pd.DataFrame(df.groupby(['FIPS','Sex','Age','Year'])['Population'].sum().reset_index())

    #Remove years prior to 2000
    pop_binned = pop_binned.loc[pop_binned['Year']>1999].reset_index(drop=True)

    return pop_binned


def all_data(combined):
    county_counts = combined['FIPS'].value_counts()
    counties = county_counts[county_counts.values==60].index

    combined = combined[combined['FIPS'].isin(counties)].reset_index(drop=True)
    return combined


def standardized_rate(combined, pop_total):
    combined = pd.merge(combined, pop_total, how='left',on=['FIPS','Year']).rename(columns = {'Population_x':'Group_Population','Population_y':'Total_Population'})

    combined['Weighted_Rate'] = (combined['Group_Population']/combined['Total_Population'])*(combined['Unstandardized_Rate'])

    combined = pd.DataFrame(combined.groupby(by=['FIPS','Year','Total_Population'])['Weighted_Rate'].sum()).reset_index().rename(columns = {'Weighted_Rate':'Cancer_Rate','Total_Population':'Population'})

    return combined

#Map each recode value in df_lung to FIPS mapping file, make columns for county, state and both
def col_changes(df_lung):
    codes = list(df_lung['FIPS'])
    lst = []
    lst2 = []
    for i in codes:
        lst.append(fips[fips.FIPS==i]['Name'].values)
        lst2.append(fips[fips.FIPS==i]['State'].values)

    df_lung['County'] = lst
    df_lung['County'] = df_lung['County'].str.get(0)
    df_lung['State'] = lst2
    df_lung['State'] = df_lung['State'].str.get(0)
    df_lung['State_and_county'] = df_lung['County']+' County, '+df_lung['State']

    return df_lung

if __name__=='__main__':
    filepath = '/home/dhense/PublicData/ZNAHealth/intermediate_files/'

########## LOAD PICKLE FILES #################
    framed_pickle = 'framed.pickle'
    pop_pickle = 'pop.pickle'
    rates_pickle = 'rates.pickle'

    print("...loading pickle")
    tmp = open(filepath+framed_pickle,'rb')
    total = pickle.load(tmp)
    tmp.close()


    print("...loading pickle")
    tmp = open(filepath+pop_pickle,'rb')
    processed = pickle.load(tmp)
    tmp.close()
############################################

    if not os.path.isfile(filepath+rates_pickle):

        grouped = group(total)
        grouped2 = group_pop(processed)

        #Total population per county per year (not broken out by sex and age)
        pop_total = pd.DataFrame(grouped2.groupby(by=['FIPS','Year'])['Population'].sum()).reset_index()

        #Combine lung cancer incidence data with population data
        combined = pd.merge(grouped,grouped2,how='left',on=['FIPS','Sex','Age','Year'])

        #Get rid of rows where Population values are null
        combined = combined[combined['Population'].isnull()==False]

        #Find counties where data available for all years (2000-2014)
        combined = all_data(combined)

        #rates per 100,000 non-standardized
        combined['Unstandardized_Rate'] = (combined['Count']/combined['Population'])*100000

        #age and gender standardized rates per 100,000
        combined = standardized_rate(combined, pop_total)

        #Add county name, state name and combination of both to dataframe
        #Read in FIPS mapper
        fips = pd.read_excel('/home/dhense/PublicData/ZNAHealth/Data_Files/FIPS.xlsx')
        fips.FIPS = fips.FIPS.apply(lambda x: str(x).zfill(5))
        fips['State & County'] = fips.Name + ' County, ' + fips.State

        combined = col_changes(combined)

########## SAVE AS PICKLE FILE ###############

        print("...saving pickle")
        tmp = open(filepath+rates_pickle,'wb')
        pickle.dump(combined,tmp)
        tmp.close()


###### IF PICKLE ALREADY EXISTS, LOAD #########

    else:

        print("...loading pickle")
        tmp = open(filepath+rates_pickle,'rb')
        combined = pickle.load(tmp)
        tmp.close()

###############################################
