from numpy.lib.function_base import select
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training as well as prediction.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def impute_missing_values(self, data, cols_with_missing_values, column_type):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
            Input: Dataframe, column names having mising values, data type
            Output: A Dataframe which has all the missing values imputed.

        """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        self.cols_with_missing_values=cols_with_missing_values
        self.column_type = column_type       
        try:
            if self.column_type == 'catagorical':
                self.imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            elif self.column_type == 'numeric':
                self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()
            

    def scale_numerical_columns(self, data, numerical_columns):
        """
            Method Name: scale_numerical_columns
            Description: This method scales the numerical values using the Standard scaler.
            Input: Dataframe, numerical columns that need scaling
            Output: A dataframe with scaled numrical columns
        """
        self.logger_object.log(self.file_object,
                               'Entered the scale_numerical_columns method of the Preprocessor class')

        self.data=data

        try:
            self.num_df = self.data.loc[:, numerical_columns]
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)

            self.logger_object.log(self.file_object, 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()
    def encode_categorical_columns(self,data, columns_to_encode):
        """
            Method Name: encode_categorical_columns
            Description: This method encodes the categorical values to numeric values.
            Input: dataframe, catagorical columns that need to be encoded
            Output: A dataframe containing the dummy values for the columns specified
        """
        self.logger_object.log(self.file_object, 'Entered the encode_categorical_columns method of the Preprocessor class')

        try:
            self.data = data.loc[:, columns_to_encode].astype('category').copy()
            self.df_dummies = pd.get_dummies(self.data.loc[:, columns_to_encode], drop_first=True)
            self.logger_object.log(self.file_object, 'Encoding catagorical values successful')
            return self.df_dummies

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def handle_imbalanced_dataset(self,x,y):
        """
            Method Name: handle_imbalanced_dataset
            Description: This method handles the imbalanced dataset to make it a balanced one.
            Input: Dataframe, Series
            Output: new balanced feature and target columns
        """
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            self.rdsmple = RandomOverSampler()
            self.x_sampled,self.y_sampled  = self.rdsmple.fit_sample(x,y)
            self.logger_object.log(self.file_object,
                                   'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return self.x_sampled,self.y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()


