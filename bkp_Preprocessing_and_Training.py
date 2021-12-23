import os
import pandas as pd
from data_preprocessing import Preprocessor
from logger import App_Logger
from clustering import KMeansClustering
from model_perfomence import ModelFinder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


filepath = r'Training_Batch_Files'

##############################Data Loading##############################
#Load all the csv files from the path specified
first_file_found = 0
for input_file in os.listdir(filepath):
    if input_file.endswith('csv'):       
        if first_file_found == 0:  # For the first file load it as it is, so that we get the column names as well 
            first_file_found += 1  # from the next file we do, just have to append them one by one
            df = pd.read_csv(os.path.join(filepath,  input_file))
        else:
            temp_df = pd.read_csv(os.path.join(filepath,  input_file))
            df = pd.concat([df, temp_df])

##############################Data Preprocessing##############################
#Encode 'SEX' column as binary just to make it more readable
df['SEX_MALE'] = df['SEX'].map(lambda x: 1 if x== 1 else 0)
df.drop('SEX', inplace=True, axis = 1)



logfile =  open('preprocessing_log.log', 'a+')
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
y = df['default payment next month']

##############################Clustering##############################
km = KMeansClustering(logfile, App_Logger())
num_clusters = km.elbow_plot(x)
data_with_clusters = km.create_clusters(x, num_clusters)

##############################Model Training##############################
x = data_with_clusters
y = df['default payment next month']

#Instantiate the models
xgb = XGBClassifier()
lr = LogisticRegression()
rf = RandomForestClassifier()

#Define the parameters that need to be searched 
xgb_params = { "n_estimators": [50],"max_depth": range(3, 6, 2)}
lr_params = {"penalty": ["l1", "l2"]}
rf_params = {'max_features': [0.5, 0.6]}

#Prepare the list of models and respective params
models = [xgb, rf]
params = [xgb_params, rf_params]

#Use find_best_model method from ModelFinder class to get the perfomence for each model
mf = ModelFinder(x,y)
results = mf.find_best_model(models, params, cv=3)
mf.plot_model_perfomence(results) #Plot model perfomence





