import pandas as pd
import os
from Utility_Classes.data_preprocessing import Preprocessor
from Utility_Classes.logger import App_Logger
from Utility_Classes.clustering import KMeansClustering
import pickle

def predict_current_file(filepath, filename):
    ##############################Data Loading##############################
    df = pd.read_csv(os.path.join(filepath,  filename))
    ##############################Data Preprocessing##############################
    #Encode 'SEX' column as binary just to make it more readable
    df['SEX_MALE'] = df['SEX'].map(lambda x: 1 if x== 1 else 0)
    df.drop('SEX', inplace=True, axis = 1)
    logfile =  open(r'Logs/prediction_log.log', 'a+')
    pre_obj = Preprocessor(logfile, App_Logger())
    #Scale the numerical columns using StandardScaler
    numerical_columns_to_scale = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
                                'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    scaled_numeric_df = pre_obj.scale_numerical_columns(df, numerical_columns_to_scale)
    #Get dummy variables for the catagorical columns
    cat_columns_to_encode = ['EDUCATION', 'MARRIAGE']
    df_dummies = pre_obj.encode_categorical_columns(df, cat_columns_to_encode)
    #Reset index
    df_dummies = df_dummies.reset_index(drop=True)
    scaled_numeric_df = scaled_numeric_df.reset_index(drop=True)
    #Combine catagorical and numeric features
    df_combined= pd.concat([df_dummies, scaled_numeric_df], axis=1)
    x = df_combined.copy()
    ##############################Clustering##############################
    km = KMeansClustering(logfile, App_Logger())
    num_clusters = km.elbow_plot(x)
    data_with_clusters = km.create_clusters(x, num_clusters)
    ##############################Prediction##############################
    x_test = data_with_clusters
    rf_model = pickle.load(open(r'Saved_Models/RandomForestClassifier.pkl', 'rb'))
    xgb_model = pickle.load(open(r'Saved_Models/XGBClassifier.pkl', 'rb'))
    y_pred_rf = rf_model.predict(x_test)
    y_pred_xgb = xgb_model.predict(x_test)
    df_y_pred_rf = pd.DataFrame(y_pred_rf)
    df_y_pred_xgb = pd.DataFrame(y_pred_xgb)
    df_y_pred_rf.to_csv(r'Prediction_Output_Files/rf_pred_'+ filename)
    df_y_pred_xgb.to_csv(r'Prediction_Output_Files/xgb_pred_'+ filename)
    return "Success"
