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
from sklearn.preprocessing import OneHotEncoder # Needed for encoding categorical data
import numpy as np # Needed for number processing and array manipulation
import os # path creation

DEBUG_MODE = False
WRITE_DF_TO_CSV = True

if __name__ == "__main__":
    # If debugging, skip these steps, and read dataframes directly from csv
    # to prevent need to recreate dataframes every run
    if not DEBUG_MODE:
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

        # Write dataframes to csv when boolean is set as such to prevent need to recreate
        # the dataframes every time
        if WRITE_DF_TO_CSV:
            model_input_csv_path = os.getcwd() + "/Processed_Dataframes/model_input.csv"
            model_input_df.to_csv(model_input_csv_path)
            classifications_csv_path = os.getcwd() + "/Processed_Dataframes/classifications.csv"
            classifications_df.to_csv(classifications_csv_path)

    # If in debug mode, read the dataframes directly from csv
    else:
        model_input_csv_path = os.getcwd() + "/Processed_Dataframes/model_input.csv"
        model_input_df = pd.read_csv(model_input_csv_path)
        classifications_csv_path = os.getcwd() + "/Processed_Dataframes/classifications.csv"
        classifications_df = pd.read_csv(classifications_csv_path)

    print(model_input_df)
    ### Encode the categorical attributes to obtain numeric representations
    ### that can be used for model training
    encoded_df = model_input_df.copy(deep=False)
    # Create list of categorical attributes that need to be encoded
    categorical_attributes = ['Location', 'Type', 'Activity']
    # For each of the categorical variables, fit an encoder and transform
    # to obtain one hot encoding
    encoder_dict = {}
    location_encoder = OneHotEncoder()
    location_reshaped = np.array(model_input_df['Location']).reshape(-1, 1)
    encoded_values = location_encoder.fit_transform(location_reshaped)
    print(encoded_values.toarray()[:5])
    for attr in categorical_attributes:
        # Create the encoder
        encoder_dict[attr] = OneHotEncoder()
        # Convert attribute values to two-dimensional matrix for one hot
        # encoder 
        unencoded_attr_values = model_input_df[attr]
        reshaped_values = np.array(unencoded_attr_values).reshape(-1, 1)
        # Fit the encoder to the classifications of the underlying attribute
        encoder_dict[attr].fit(reshaped_values)
        # Transform encoder to retrieve encoded value
        encoded_attr_values = encoder_dict[attr].fit_transform(reshaped_values)
        # Replaced the attribute in dataframe with corresponding encoded column
        # encoded_df.replace(encoded_df.loc[:, attr], encoded_attr_values)
    ## Since hour is continuous and cyclical, encode it using sine and
    ## cosine function (e.,g since hour 24 is closer to 1 than 22)



