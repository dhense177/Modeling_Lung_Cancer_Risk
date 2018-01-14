import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle,os,csv

indices = [(0,8),(8,18),(18,19),(19,21),(22,23),(23,24),(24,27),(27,31),(34,36),(36,38),(38,42),(42,46),(46,47),(47,51),(51,52),(52,56),(56,57),(57,58),(58,59),(59,60),(60,63),(63,65),(65,67),(67,68),(68,70),(70,72),(72,85),(85,87),(87,91),(91,92),(92,93),(93,94),(94,95),(95,98),(98,101),(101,104),(104,106),(106,109),(109,112),(112,115),(115,118),(118,121),(121,124),(124,127),(127,129),(129,131),(131,133),(133,135),(135,136),(136,137),(137,138),(140,146),(146,152),(152,158),(158,160),(160,161),(161,162),(162,164),(165,166),(169,171),(173,174),(175,177),(190,191),(191,193),(198,203),(203,207),(207,211),(217,220),(220,223),(223,224),(225,227),(227,229),(229,232),(232,233),(233,234),(234,235),(235,236),(236,238),(238,240),(240,241),(241,242),(244,245),(245,250),(254,259),(259,264),(264,265),(265,266),(266,267),(267,269),(269,271),(271,272),(272,273),(273,274),(274,275),(275,276),(276,277),(277,278),(278,279),(279,281),(281,284),(284,287),(287,290),(290,293),(293,296),(296,299),(299,300),(300,304),(304,305),(310,311),(311,314),(314,317),(317,320),(320,323),(323,325),(325,327),(327,329),(329,331),(331,334),(334,337),(337,340),(340,341),(341,342),(347,348),(348,349),(349,350),(350,351),(351,352),(352,354),(354,356),(356,358),(358,360),(360,362)]

clean_df = pd.DataFrame(columns=["Patient ID number", "Registry ID", "Marital Status at DX", "Race/Ethnicity", "NHIA Derived Hispanic Origin", "Sex", "Age at diagnosis", "Year of Birth", "Sequence Number—Central", "Month of diagnosis", "Year of diagnosis", "Primary Site", "Laterality", "Histology (92-00) ICD-O-2", "Behavior (92-00) ICD-O-2","HISTOLOGIC TYPE ICD-O-3","BEHAVIOR CODE ICD-O-3", "Grade", "Diagnostic Confirmation", "Type of Reporting Source", "EOD—Tumor Size", "EOD—Extension", "EOD—Extension Prost Path", "EOD—Lymph Node Involv", "Regional Nodes Positive", "Regional Nodes Examined", "EOD—Old 13 Digit", "EOD—Old 2 Digit", "EOD—Old 4 Digit", "Coding System for EOD", "Tumor Marker 1", "Tumor Marker 2", "Tumor Marker 3", "CS Tumor Size", "CS Extension", "CS Lymph Nodes", "CS Mets at Dx", "CS Site-Specific Factor 1", "CS Site-Specific Factor 2", "CS Site-Specific Factor 3", "CS Site-Specific Factor 4", "CS Site-Specific Factor 5", "CS Site-Specific Factor 6", "CS Site-Specific Factor 25", "Derived AJCC T", "Derived AJCC N", "Derived AJCC M", "Derived AJCC Stage Group", "Derived SS1977", "Derived SS2000", "Derived AJCC—Flag", "CS Version Input Original", "CS Version Derived", "CS Version Input Current", "RX Summ—Surg Prim Site", "RX Summ—Scope Reg LN Sur", "RX Summ—Surg Oth Reg/Dis", "RX Summ—Reg LN Examined", "Reason for no surgery", "RX Summ—Surgery Type", "RX Summ—Scope Reg 98-02", "SEER Record Number", "SEER Type of Follow-up", "Age Recode <1 Year olds", "Site Recode ICD-O-3/WHO 2008", "Recode ICD-O-2 to 9", "Recode ICD-O-2 to 10", "ICCC site recode ICD-O-3/WHO 2008", "ICCC site rec extended ICD-O-3/WHO 2008", "Behavior Recode for Analysis", "Histology Recode—Broad Groupings", "Histology Recode—Brain Groupings", "CS Schema v0204+", "Race recode (White, Black, Other)", "Race recode (W, B, AI, API)", "Origin recode NHIA (Hispanic, Non-Hisp)", "SEER historic stage A", "AJCC stage 3rd edition (1988-2003)", "SEER modified AJCC Stage 3 rd ed (1988-2003)", "SEER Summary Stage 1977 (1995-2000)", "SEER Summary Stage 2000 (2001-2003)", "First malignant primary indicator", "State-county recode", "Cause of Death to SEER site recode", "COD to site rec KM", "Vital Status recode", "IHS Link", "Summary stage 2000 (1998+)", "AYA site recode/WHO 2008", "Lymphoma subtype recode/WHO 2008", "SEER Cause-Specific Death Classification", "SEER Other Cause of Death Classification", "CS Tumor Size/Ext Eval", "CS Lymph Nodes Eval", "CS Mets Eval", "Primary by international rules", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)", "CS Schema -AJCC 6th ed (previously called v1)", "CS Site-Specific Factor 8", "CS Site-Specific Factor 10", "CS Site-Specific Factor 11", "CS Site-Specific Factor 13", "CS Site-Specific Factor 15", "CS Site-Specific Factor 16", "Lymph vascular invasion", "Survival months", "Survival months flag", "Insurance recode (2007+)", "Derived AJCC-7 T", "Derived AJCC-7 N", "Derived AJCC-7 M", "Derived AJCC-7 Stage Grp", "Breast Adjusted AJCC 6 th T (1988+)", "Breast Adjusted AJCC 6 th N (1988+)", "Breast Adjusted AJCC 6 th M (1988+)", "Breast Adjusted AJCC 6 th Stage (1988+)", "CS Site-Specific Factor 7", "CS Site-Specific Factor 9", "CS Site-Specific Factor 12", "Derived HER2 Recode (2010+)", "Breast Subtype (2010+)", "Lymphomas: Ann Arbor Staging (1983+)", "CS Mets at Dx-Bone", "CS Mets at Dx-Brain", "CS Mets at Dx-Liver", "CS Mets at Dx-Lung", "T value - based on AJCC 3rd (1988-2003)", "N value - based on AJCC 3rd (1988-2003)", "M value - based on AJCC 3rd (1988-2003)", "Total Number of In Situ/malignant Tumors for Patient", "Total Number of Benign/Borderline Tumors for Patient"])



def parse_rows(df):
        dct = {}
        for num in range(len(df)):
            string = ""
            x = df.iloc[num][0]
            for i in range(len(indices)):
                string += x[indices[i][0]:indices[i][1]] + ", "
            dct[num] = string[:-2]
            if num%10000 ==0:
                print (num)
        return dct


def create_df(dct, df):
    for i, k in enumerate(dct.values()):
        dct[i] = k.split(', ')
    df_new = pd.DataFrame.from_dict(dct, orient='index')
    df_new.columns = df.columns
    return df_new






if __name__=='__main__':
    filepath1 = '/home/davidhenslovitz/Galvanize/ZNAHealth/SEER_1973_2014_TEXTDATA/incidence/yr1973_2014.seer9/RESPIR.TXT'
    filepath2 = '/home/davidhenslovitz/Galvanize/ZNAHealth/SEER_1973_2014_TEXTDATA/incidence/yr1992_2014.sj_la_rg_ak/RESPIR.TXT'
    filepath3 = '/home/davidhenslovitz/Galvanize/ZNAHealth/SEER_1973_2014_TEXTDATA/incidence/yr2000_2014.ca_ky_lo_nj_ga/RESPIR.TXT'

    filepaths = [filepath1, filepath2, filepath3]

    master_df = pd.DataFrame()



    for textfile in filepaths:
        df = pd.read_table(textfile, header=None)
        parsed = parse_rows(df)
        new_df = create_df(parsed, clean_df)
        master_df = master_df.append(new_df)


    master_df.to_csv('RESPIR.csv', index=False)
