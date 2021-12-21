import os
import pandas as pd
filepath = r'E:\Me\Project\INeuron\creditCardDefaulters\creditCardDefaulters\code\creditCardDefaulters\Training_Batch_Files'
columns = [
'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',
'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3',
'PAY_AMT4','PAY_AMT5','PAY_AMT6','default payment next month']

df = pd.DataFrame(columns=columns)

df = pd.DataFrame()
for input_file in os.listdir(filepath):
    if input_file.endswith('csv'):
        temp_df = pd.read_csv(os.path.join(filepath,  input_file))
        df = pd.concat([df, temp_df])

