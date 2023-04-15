###############################################################################
### Script name:     encode_data.py                                  ###
### Script function: encodes the attributes of preprocessed dataframe such  ###
###                  that each value is numerical can be used to train      ###
###                  clustering model                                       ###
### Authors:         Justin Dannemiller, Keith Machina, and Bailey Wimer    ###
### Last Modified: 04/14/2023                                               ###
###############################################################################

import pandas as pd # Needed for dataframe manipulation
from preprocess_data import preprocess_datasets
from sklearn.preprocessing import LabelEncoder # Needed for encoding categorical data
import numpy as np

if __name__ == "__main__":
    ### Preprocess the dataset and store in a dataframe
    resulting_dataframe = preprocess_datasets()
    
    ## Split dataframe into two such that one dataframe stores the classifications
    ## and the other stores all the other data. This way we can map clustering 
    ## results back to true classifications
    column_names = list(resulting_dataframe.columns)
    class_column_name = 'True Classification'
    # Remove class column from column list
    column_names.remove(class_column_name)

    # Create df to store classes
    classifications_df = resulting_dataframe.copy(deep=False) 
    # Drop all columns but class
    classifications_df = classifications_df.drop(labels = column_names, axis=1)

    # Create df to store rest of data
    model_input_df = resulting_dataframe.copy(deep=True)
    # Drop only class column
    class_column_name = [class_column_name]
    model_input_df = model_input_df.drop(class_column_name, axis=1)
    print("Rest of data")
    print(model_input_df.columns)

    ### Encode the categorical attributes to obtain numeric representations
    ### that can be used for model training

    # Create list of categorical attributes that need to be encoded
    categorical_columns = ['Location', 'Type', 'Activity']

    ## Since hour is continuous and cyclical, encode it using sine and
    ## cosine function (e.,g since hour 24 is closer to 1 than 22)



