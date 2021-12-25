import os
from Utility_Classes.file_validator import FileValidator
from Utility_Classes.Preprocessing_and_Prediction import predict_current_file

def predict_all_files(filepath):
    ##############################Check for valid files#####################
    fv = FileValidator()
    n_expected_columns = 23
    valid_files, invalid_files = fv.validate_file_structure(filepath, n_expected_columns)

    #Predict for all valid csv files from the path specified
    for input_file in os.listdir(filepath):
        if input_file in valid_files: 
            return_message = predict_current_file(filepath, input_file) 
            if return_message == "Success":
                print(return_message)
    return f'Prediction_Output_Files/'           
    

