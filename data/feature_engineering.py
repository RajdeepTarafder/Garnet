# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from numpy import inf

def OXIDE(data: pd.DataFrame):
    data_rev = data
    #SiO conversion
    df_SiO =data_rev.copy()
    #taking out  df/SiOO2)
    df_SiO=df_SiO.div(df_SiO['SiO2'], axis=0)
    # rename columns
    df_SiO.columns = '('+ df_SiO.columns +'/'+ 'SiO2)'

    df_SiO=df_SiO.round(6)
    df_SiO.drop(columns=['(SiO2/SiO2)'], axis=1, inplace=True)

    df_SiO = df_SiO.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)


    #CaO converSiOon
    df_CaO=data_rev.copy()
    #taking out  df/CaO)
    df_CaO=df_CaO.div(df_CaO['CaO'], axis=0)
    # rename columns
    df_CaO.columns = '('+ df_CaO.columns +'/'+ 'CaO)'

    df_CaO=df_CaO.round(6)
    df_CaO.drop(columns=['(CaO/CaO)'], axis=1, inplace=True)

    df_CaO = df_CaO.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_CaO= df_CaO.replace(inf, 0)
    df_CaO= df_CaO.replace(np.nan, 0)

    #MgO conversion
    df_MgO=data_rev.copy()
    #taking out  df/MgO)
    df_MgO=df_MgO.div(df_MgO['MgO'], axis=0)
    # rename columns
    df_MgO.columns = '('+ df_MgO.columns +'/'+ 'MgO)'

    df_MgO=df_MgO.round(6)
    df_MgO.drop(columns=['(MgO/MgO)'], axis=1, inplace=True)

    df_MgO = df_MgO.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_MgO= df_MgO.replace(inf, 0)
    df_MgO= df_MgO.replace(np.nan, 0)

    #MnO conversion
    df_MnO=data_rev.copy()
    #taking out  df/MnO)
    df_MnO=df_MnO.div(df_MnO['MnO'], axis=0)
    # rename columns
    df_MnO.columns = '('+ df_MnO.columns +'/'+ 'MnO)'

    df_MnO=df_MnO.round(6)
    df_MnO.drop(columns=['(MnO/MnO)'], axis=1, inplace=True)

    df_MnO = df_MnO.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_MnO= df_MnO.replace(inf, 0)
    df_MnO= df_MnO.replace(np.nan, 0)

    #Al2O3 conversion
    df_Al2O3=data_rev.copy()
    #taking out  df/Al2O3)
    df_Al2O3=df_Al2O3.div(df_Al2O3['Al2O3'], axis=0)
    # rename columns
    df_Al2O3.columns = '('+ df_Al2O3.columns +'/'+ 'Al2O3)'

    df_Al2O3=df_Al2O3.round(6)
    df_Al2O3.drop(columns=['(Al2O3/Al2O3)'], axis=1, inplace=True)

    df_Al2O3 = df_Al2O3.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Al2O3= df_Al2O3.replace(inf, 0)
    df_Al2O3= df_Al2O3.replace(np.nan, 0)




    #Cr2O3 conversion
    df_Cr2O3=data_rev.copy()
    #taking out  df/Cr2O3)
    df_Cr2O3=df_Cr2O3.div(df_Cr2O3['Cr2O3'], axis=0)
    # rename columns
    df_Cr2O3.columns = '('+ df_Cr2O3.columns +'/'+ 'Cr2O3)'

    df_Cr2O3=df_Cr2O3.round(6)
    df_Cr2O3.drop(columns=['(Cr2O3/Cr2O3)'], axis=1, inplace=True)

    df_Cr2O3 = df_Cr2O3.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Cr2O3= df_Cr2O3.replace(inf, 0)
    df_Cr2O3= df_Cr2O3.replace(np.nan, 0)

    #FeO conversion
    df_FeO=data_rev.copy()
    #taking out  df/FeOtotal)
    df_FeO=df_FeO.div(df_FeO['FeOtotal'], axis=0)
    # rename columns
    df_FeO.columns = '('+ df_FeO.columns +'/'+ 'FeO)'

    df_FeO=df_FeO.round(6)
    df_FeO.drop(columns=['(FeOtotal/FeO)'], axis=1, inplace=True)

    df_Fe = df_FeO.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_FeO= df_FeO.replace(inf, 0)
    df_FeO= df_FeO.replace(np.nan, 0)
    data_1= pd.concat([df_SiO,df_CaO,df_MgO,df_MnO,df_Al2O3,df_Cr2O3,df_FeO],axis=1)
    
    return data_1
    
def Cation(data: pd.DataFrame):
    data_rev=data
    #Si conversion
    df_Si=data_rev.copy()
    #taking out  df/SiO2)
    df_Si=df_Si.div(df_Si['SiO2'], axis=0)
    # rename columns
    df_Si.columns = '('+ df_Si.columns +'/'+ 'SiO2)'

    df_Si=df_Si.round(6)
    df_Si.drop(columns=['(SiO2/SiO2)'], axis=1, inplace=True)
    #taking out  df/Si
    df_Si=df_Si.mul([0.752,1.179,0.791,0.847,0.836,1.491,1.071], axis='columns')
    # rename columns
    df_Si.rename(columns = {'(TiO2/SiO2)':'Ti/Si','(Al2O3/SiO2)':'Al/Si','(Cr2O3/SiO2)':'Cr/Si','(MnO/SiO2)':'Mn/Si','(FeOtotal/SiO2)':'Fe/Si','(MgO/SiO2)':'Mg/Si','(CaO/SiO2)':'Ca/Si'}, inplace = True)

    df_Si = df_Si.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)


    #Ca conversion
    df_Ca=data_rev.copy()
    #taking out  df/CaO)
    df_Ca=df_Ca.div(df_Ca['CaO'], axis=0)
    # rename columns
    df_Ca.columns = '('+ df_Ca.columns +'/'+ 'CaO)'

    df_Ca=df_Ca.round(6)
    df_Ca.drop(columns=['(CaO/CaO)'], axis=1, inplace=True)
    #taking out  df/Ca
    df_Ca=df_Ca.mul([0.933,0.702,0.550,0.369,0.781,0.791,1.391], axis='columns')
    # rename columns
    df_Ca.rename(columns = {'(TiO2/CaO)':'Ti/Ca','(Al2O3/CaO)':'Al/Ca','(Cr2O3/CaO)':'Cr/Ca','(MnO/CaO)':'Mn/Ca','(FeOtotal/CaO)':'Fe/Ca','(MgO/CaO)':'Mg/Ca','(SiO2/CaO)':'Si/Ca'}, inplace = True)

    df_Ca = df_Ca.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Ca= df_Ca.replace(inf, 0)
    df_Ca= df_Ca.replace(np.nan, 0)

    #Mg conversion
    df_Mg=data_rev.copy()
    #taking out  df/MgO)
    df_Mg=df_Mg.div(df_Mg['MgO'], axis=0)
    # rename columns
    df_Mg.columns = '('+ df_Mg.columns +'/'+ 'MgO)'

    df_Mg=df_Mg.round(6)
    df_Mg.drop(columns=['(MgO/MgO)'], axis=1, inplace=True)
    #taking out  df/Mg
    df_Mg=df_Mg.mul([0.671,0.505,0.395,0.265,0.561,0.568,0.719], axis='columns')
    # rename columns
    df_Mg.rename(columns = {'(TiO2/MgO)':'Ti/Mg','(Al2O3/MgO)':'Al/Mg','(Cr2O3/MgO)':'Cr/Mg','(MnO/MgO)':'Mn/Mg','(FeOtotal/MgO)':'Fe/Mg','(SiO2/MgO)':'Si/Mg','(CaO/MgO)':'Ca/Mg'}, inplace = True)

    df_Mg = df_Mg.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Mg= df_Mg.replace(inf, 0)
    df_Mg= df_Mg.replace(np.nan, 0)

    #Mn conversion
    df_Mn=data_rev.copy()
    #taking out  df/MnO)
    df_Mn=df_Mn.div(df_Mn['MnO'], axis=0)
    # rename columns
    df_Mn.columns = '('+ df_Mn.columns +'/'+ 'MnO)'

    df_Mn=df_Mn.round(6)
    df_Mn.drop(columns=['(MnO/MnO)'], axis=1, inplace=True)
    #taking out  df/Mn
    df_Mn=df_Mn.mul([1.181,0.888,0.696,0.467,0.987,1.760,1.265], axis='columns')
    # rename columns
    df_Mn.rename(columns ={'(TiO2/MnO)':'Ti/Mn','(Al2O3/MnO)':'Al/Mn','(Cr2O3/MnO)':'Cr/Mn','(MgO/MnO)':'Mg/Mn','(FeOtotal/MnO)':'Fe/Mn','(SiO2/MnO)':'Si/Mn','(CaO/MnO)':'Ca/Mn'}, inplace = True)

    df_Mn = df_Mn.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Mn= df_Mn.replace(inf, 0)
    df_Mn= df_Mn.replace(np.nan, 0)

    #Al conversion
    df_Al=data_rev.copy()
    #taking out  df/Al2O3)
    df_Al=df_Al.div(df_Al['Al2O3'], axis=0)
    # rename columns
    df_Al.columns = '('+ df_Al.columns +'/'+ 'Al2O3)'

    df_Al=df_Al.round(6)
    df_Al.drop(columns=['(Al2O3/Al2O3)'], axis=1, inplace=True)
    #taking out  df/Al
    df_Al=df_Al.mul([1.181,0.888,0.696,0.467,0.987,1.760,1.265], axis='columns')
    # rename columns
    df_Al.rename(columns = {'(TiO2/Al2O3)':'Ti/Al','(MnO/Al2O3)':'Mn/Al','(Cr2O3/Al2O3)':'Cr/Al','(MgO/Al2O3)':'Mg/Al','(FeOtotal/Al2O3)':'Fe/Al','(SiO2/Al2O3)':'Si/Al','(CaO/Al2O3)':'Ca/Al'}, inplace = True)

    df_Al = df_Al.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Al= df_Al.replace(inf, 0)
    df_Al= df_Al.replace(np.nan, 0)




    #Cr conversion
    df_Cr=data_rev.copy()
    #taking out  df/Cr2O3)
    df_Cr=df_Cr.div(df_Cr['Cr2O3'], axis=0)
    # rename columns
    df_Cr.columns = '('+ df_Cr.columns +'/'+ 'Cr2O3)'

    df_Cr=df_Cr.round(6)
    df_Cr.drop(columns=['(Cr2O3/Cr2O3)'], axis=1, inplace=True)
    #taking out  df/Cr
    df_Cr=df_Cr.mul([2.530,1.903,1.491,2.116,2.143,3.771,2.710], axis='columns')
    # rename columns
    df_Cr.rename(columns = {'(TiO2/Cr2O3)':'Ti/Cr','(MnO/Cr2O3)':'Mn/Cr','(Al2O3/Cr2O3)':'Al/Cr','(MgO/Cr2O3)':'Mg/Cr','(FeOtotal/Cr2O3)':'Fe/Cr','(SiO2/Cr2O3)':'Si/Cr','(CaO/Cr2O3)':'Ca/Cr'}, inplace = True)

    df_Cr = df_Cr.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Cr= df_Cr.replace(inf, 0)
    df_Cr= df_Cr.replace(np.nan, 0)

    #Fe conversion
    df_Fe=data_rev.copy()
    #taking out  df/FeOtotal)
    df_Fe=df_Fe.div(df_Fe['FeOtotal'], axis=0)
    # rename columns
    df_Fe.columns = '('+ df_Fe.columns +'/'+ 'FeO)'

    df_Fe=df_Fe.round(6)
    df_Fe.drop(columns=['(FeOtotal/FeO)'], axis=1, inplace=True)
    #taking out  df/Fe
    df_Fe=df_Fe.mul([1.196,0.900,0.705,0.473,1.013,1.783,1.281], axis='columns')
    # rename columns
    df_Fe.rename(columns = {'(TiO2/FeO)':'Ti/Fe','(MnO/FeO)':'Mn/Fe','(Al2O3/FeO)':'Al/Fe','(MgO/FeO)':'Mg/Fe','(Cr2O3/FeO)':'Cr/Fe','(SiO2/FeO)':'Si/Fe','(CaO/FeO)':'Ca/Fe'}, inplace = True)

    df_Fe = df_Fe.applymap(lambda x: math.log10(x) if((x != 1.0) & (x!=0)) else 0)
    df_Fe= df_Fe.replace(inf, 0)
    df_Fe= df_Fe.replace(np.nan, 0)
    data_1= pd.concat([df_Si,df_Ca,df_Mg,df_Mn,df_Al,df_Cr,df_Fe],axis=1)
    
    return data_1

def cation_num(data: pd.DataFrame):
    data_rev=data
    #Fe/Mg/Ca/Mn no conversion
    df_n=data_rev.copy()

    df_n =df_n[['FeOtotal','MnO','MgO','CaO']]


    df_n1 = df_n['FeOtotal'].div(71.8444)
    df_n2 = df_n['MgO'].div(40.305)
    df_n3 = df_n['CaO'].div(56.0774)


    #df_n =(df_n['Mno']/54.9380)


    df_n= pd.concat([df_n1,df_n2,df_n3],axis=1)



    df_n['(Fe+Mg)']= df_n['MgO'] + df_n['FeOtotal']
    df_n['(Ca+Mg)']=df_n['CaO'] + df_n['MgO'] 



    df_n['Mg/(Fe+Mg)']=df_n['MgO'].div(df_n['(Fe+Mg)'], axis=0)
    df_n['Ca/(Ca+Mg)']=df_n['CaO'].div(df_n['(Ca+Mg)'], axis=0)
    data_1= pd.concat([df_n['Ca/(Ca+Mg)'],df_n['Mg/(Fe+Mg)']],axis=1)
    
    
    return data_1




