import numpy as np
import pandas as pd
import pickle,os,csv



if __name__=='__main__':

    rates_pickle = 'rates.pickle'
    rates_overall_pickle = 'rates_overall.pickle'

    rates_final = 'final_rates.pickle'

    if not os.path.isfile(rates_final):
        print("...loading pickle")
        tmp = open(rates_pickle,'rb')
        combined = pickle.load(tmp)
        tmp.close()

        print("...loading pickle")
        tmp = open(rates_overall_pickle,'rb')
        combined_overall = pickle.load(tmp)
        tmp.close()


        combined2 = pd.merge(combined, combined_overall.iloc[:,16:32], how='left', on='State-county recode_x',)

        for i in range(2000, 2015):
            combined2[str(i)+'cancer_rate'] = (combined2[str(i)+"_y_x"]/combined2[str(i)+"_y_y"])*combined2[i]
        combined2.drop(['Sex_x', 2015, 2016, 'State-county recode_y', 'Sex_y', 'Age_y'], axis=1, inplace=True)

        #grouped by county
        combined_new = pd.DataFrame(combined2.groupby('State-county recode_x')[combined2.columns].sum().reset_index())

        print("...saving pickle")
        tmp = open(rates_final,'wb')
        pickle.dump(combined_new,tmp)
        tmp.close()
    else:
        print("...loading pickle")
        tmp = open(rates_final,'rb')
        combined_new = pickle.load(tmp)
        tmp.close()
