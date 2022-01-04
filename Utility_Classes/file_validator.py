import pandas as pd
import os

class FileValidator:
    def __init__(self) -> None:
        pass

    def validate_file_structure(self, filepath, n_columns):
        self.filepath = filepath
        self.n_columns = n_columns
        self.valid_file_list = []
        self.invalid_file_list = []

        for input_file in os.listdir(filepath):
            if input_file.endswith('csv'):       
                self.df = pd.read_csv(os.path.join(self.filepath,  input_file))
                self.current_n_columns = self.df.shape[1]
                #Check if we have expected number of columns in the prediction data or not
                if self.n_columns == self.current_n_columns:
                    self.null_counts = self.df.columns.value_counts()
                    #Check if there is all NaN for any column
                    self.check = (self.null_counts.values == self.df.shape[0]).any()
                    if self.check == False:
                        self.valid_file_list.append(input_file)
                    else:
                        self.invalid_file_list.append(input_file)
                else:
                        self.invalid_file_list.append(input_file)
            else:
                        self.invalid_file_list.append(input_file)
        return self.valid_file_list, self.invalid_file_list


    def validate_file_name(self):
        pass


# filepath = r'E:\Me\Project\INeuron\creditCardDefaulters\creditCardDefaulters\code\creditCardDefaulters\Prediction_Batch_Files'

# fv = FileValidator()
# valid, invalid = fv.validate_file_structure(filepath, 23)
# print(valid)
# print(invalid)