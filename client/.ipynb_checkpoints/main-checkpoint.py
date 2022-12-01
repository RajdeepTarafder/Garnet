# -*- coding: utf-8 -*-
import tmp
import os
import numpy as np
import pandas as pd
from global_variable import OPTION, SECTION, IMPUTING_STRATEGY, MODE_OPTION, REGRESSION_MODELS,\
    CLASSIFICATION_MODELS, CLUSTERING_MODELS, DECOMPOSITION_MODELS, WORKING_PATH, DATA_OPTION,\
    TEST_DATA_OPTION, MODEL_OUTPUT_IMAGE_PATH, STATISTIC_IMAGE_PATH, DATASET_OUTPUT_PATH,\
    GEO_IMAGE_PATH, DATASET_PATH, MISS_VALUES,FEATURE_ENGINEERING,LABEL_COMP,LABEL_PROV
from data.data_readiness import read_data, show_data_columns, num2option, create_sub_data_set, basic_info, np2pd, \
    num_input, limit_num_input
from data.imputation import Simimputer,knnimputer,String_ch
from plot.statistic_plot import basic_statistic,VIF_plot, correlation_plot, distribution_plot, is_null_value, probability_plot, \
    ratio_null_vs_filled
from utils.base import clear_output, log
from process.regress import RegressionModelSelection
from process.classify import ClassificationModelSelection
from process.cluster import ClusteringModelSelection
from process.decompose import DecompositionModelSelection
from LabelEncode.Label import Label_conv
from data.feature_engineering import OXIDE,Cation,cation_num

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_IMAGE_PATH, exist_ok=True)
os.makedirs(STATISTIC_IMAGE_PATH, exist_ok=True)
os.makedirs(DATASET_OUTPUT_PATH, exist_ok=True)
os.makedirs(GEO_IMAGE_PATH, exist_ok=True)


def main():
    print("Geochemistry Py - User Behaviour Testing Demo")
    print("....... Initializing .......")
    logger = log(WORKING_PATH, "test.log")
    logger.info("Geochemistry Py - User Behaviour Testing Demo")

    # Read the data
    logger.debug("Data Uploaded")
    print("-*-*- Data Uploaded -*-*-")
    print("Data Option:")
    num2option(DATA_OPTION)
    is_own_data = limit_num_input(DATA_OPTION, SECTION[0], num_input)
    if is_own_data == 1:
        slogan = "Data Set Name (including the stored path and suffix. e.g. /Users/Sany/data/aa.xlsx): "
        data = read_data(is_own_data=is_own_data, prefix=SECTION[0], slogan=slogan)
    else:
        print("Testing Data Option:")
        num2option(TEST_DATA_OPTION)
        test_data_num = limit_num_input(TEST_DATA_OPTION, SECTION[0], num_input)
        file_name = ''
        if test_data_num == 1:
            file_name = 'Training_(C vs M).xlsx'
        elif test_data_num == 2:
            file_name = 'Training_data_Comp.xlsx'
        elif test_data_num == 3:
            file_name = 'Training_data_Meta.xlsx'
        data = read_data(file_name=file_name)
    show_data_columns(data.columns)
    clear_output()
    
    df_label = data.copy()
    logger.debug("Labellinng")
    print("-*-*- Is Your Data labelled -*-*-")
    num2option(OPTION)
    leb_no = limit_num_input(OPTION, SECTION[1], num_input)
    if leb_no ==1 :
        label_data=create_sub_data_set(df_label)        
    elif leb_no ==2 :
        # Create the subset of data if pre-labelling is absent with the data
        logger.debug("Data Labels Selected")
        print("-*-*- Selecte Data for Labelling -*-*-")
        show_data_columns(df_label.columns)
        data_Label = create_sub_data_set(df_label)
        print("The Selected Data Set:")
        print(data_Label)
        label_data = Label_conv(data_Label)
    clear_output()
    # Create the processing data set
    logger.debug("Data Selected")
    print("-*-*- Data Selected -*-*-")
    show_data_columns(data.columns)
    data_processed = create_sub_data_set(data)
    clear_output()
    print("The Selected Data Set:")
    print(data_processed)
    is_null_value(data_processed)
    ratio_null_vs_filled(data_processed)
    # this variable used for imputing
    clear_output()
       
    #Imputation
    logger.debug("Imputation")
    print("-*-*- Type of Missing Values -*-*-")
    df= data_processed.copy()
    for i in range(1,4):
        num2option(MISS_VALUES)
        print("Which VALUES do you want to apply?")
        val_miss = limit_num_input(MISS_VALUES, SECTION[1], num_input)
        if val_miss == 1:
            print("-*-*- Strategy for Missing Values -*-*-")
            num2option(IMPUTING_STRATEGY)
            print("Which strategy do you want to apply?")
            strategy_num = limit_num_input(IMPUTING_STRATEGY, SECTION[1], num_input)
            if strategy_num == 4:
                data_processed_imputed_np = knnimputer(df)
                data_processed_imputed = np2pd(data_processed_imputed_np, df.columns)
            else:
                data_processed_imputed_np = Simimputer(df, IMPUTING_STRATEGY[strategy_num - 1])
                data_processed_imputed = np2pd(data_processed_imputed_np, df.columns)
        else:
            data_processed_imputed = String_ch(df)

        df = data_processed_imputed
        print (df)
        print("-*-*- More Missing Values to replace -*-*-") 
        num2option(OPTION)
        print("Which VALUES do you want to apply?")
        rep = limit_num_input(OPTION, SECTION[1], num_input)
        if rep == 2:
            data_processed_imputed = df
            break    
    clear_output()
    
    
    
    # basic Plots
    print('Basic Statistical Information: ')
    basic_info(data_processed_imputed)
    basic_statistic(data_processed_imputed)
    probability_plot(data_processed_imputed.columns, data_processed_imputed, data_processed_imputed)
    correlation_plot(data_processed_imputed.columns, data_processed_imputed)
    VIF_plot(data_processed_imputed)
    distribution_plot(data_processed_imputed.columns, data_processed_imputed)
    clear_output()
    
    #Feature Engineering
    logger.debug("Feature Engineering")
    print("-*-*- Mode Options -*-*-")   
    df_feat_imp = data_processed_imputed.copy()
    df_feat=pd.DataFrame()
    df_imp = pd.DataFrame()
    for i in range(0,4):
        num2option(FEATURE_ENGINEERING)
        Fea_imp = limit_num_input(FEATURE_ENGINEERING, SECTION[1], num_input)
        if Fea_imp == 1:
            df_feat=OXIDE(df_feat_imp)
        elif Fea_imp == 2:
            df_feat=Cation(df_feat_imp)
        elif Fea_imp == 3:
            df_feat=cation_num(df_feat_imp)
        else:
            break
        
        df_imp =  pd.concat([ df_feat,df_imp],axis=1)
        print (df_imp)
        print("-*-*- Add More values  -*-*-") 
        num2option(OPTION)
        rep = limit_num_input(OPTION, SECTION[1], num_input)
        print("More Features can lead to bias model and overfitting")
        if rep == 1:
            data_processed_imputed = df_imp
        elif rep == 2:
            data_processed_imputed = df_imp
            break
    clear_output()


    # Mode selection
    logger.debug("Mode Selection")
    print("-*-*- Mode Options -*-*-")
    num2option(MODE_OPTION)
    mode_num = limit_num_input(MODE_OPTION, SECTION[2], num_input)
    clear_output()
    
    Label_y=[]
    # divide X and y data set when it is supervised learning
    logger.debug("Data Split")
    if mode_num == 1 or mode_num == 2:
        print("-*-*- Data Split -*-*-")
        print("Divide the processing data set into X (feature value) and Y (target value) respectively.")
        # create X data set
        print("Selected sub data set to create X data set:")
        show_data_columns(data_processed_imputed.columns)
        print('The selected X data set:')
        X = data_processed_imputed
        print('Successfully create X data set.')
        VIF_plot(X)
        clear_output()
        # create Y data set
        print('The selected Y data set: Label data')
        y = label_data
        print('Successfully create Y data set.')
        clear_output()
    elif mode_num == 4:
        X = data_processed_imputed
        print("-*-*- Label for PCA model output -*-*-")
        num2option(TEST_DATA_OPTION)
        label_1_inp = limit_num_input(TEST_DATA_OPTION, SECTION[2], num_input)
        if label_1_inp==1:
            y = label_data
            Label_y=LABEL_CM
        elif label_1_inp==2:
            y = label_data
            Label_y=LABEL_COMP.copy
        elif label_1_inp==3:
            y = label_data
            Label_y =LABEL_PROV
    else:
        # unsupervised learning
        X = data_processed_imputed
        y = None
    
    # Model option for users
    logger.debug("Model Selection")
    print("-*-*- Model Selection -*-*-:")
    Modes2Models = {1: REGRESSION_MODELS, 2: CLASSIFICATION_MODELS,
                    3: CLUSTERING_MODELS, 4: DECOMPOSITION_MODELS}
    Modes2Initiators = {1: RegressionModelSelection, 2: ClassificationModelSelection,
                        3: ClusteringModelSelection, 4: DecompositionModelSelection}
    MODELS = Modes2Models[mode_num]
    # print(MODELS)
    num2option(MODELS)
    all_models_num = len(MODELS) + 1
    # all_models_num = 0
    print(str(all_models_num) + " - All models above to be trained")
    print("Which model do you want to apply?(Enter the Corresponding Number):")
    # FIXME how to train all the algorithms at once
    MODELS.append("all_models")
    # print(MODELS)
    model_num = limit_num_input(MODELS, SECTION[2], num_input)
    

    # Model trained selection
    logger.debug("Model Training")
    if model_num != all_models_num:
        # run the designated model
        model = MODELS[model_num - 1]
        run = Modes2Initiators[mode_num](model)
        run.activate(X, y)
    else:
        # gain all models result in the specific mode
        for i in range(len(MODELS)):
            run = Modes2Initiators[mode_num](MODELS[i])
            run.activate(X, y)


if __name__ == "__main__":
    tmp.tmp()
    main()
