###############################################################################
### Script name:     encode_data.py                                         ###
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

## Converts a datetime.time object from %H%M%S format to seconds
# Takes string form of time object and returns number of seconds
def convert_time_to_secs(time_str):
    # Split the string using ":" as delimiter (assumes time is delimited 
    # with colons
    return time_str.total_seconds()

    # time_components = time_str.split(":")
    # hours = int(time_components[0])
    # minutes = int(time_components[1])
    # seconds = int(time_components[2])

    # total_seconds = 3600*hours + 60*minutes + seconds
    # return total_seconds

# Main function of this program, takes the preprocessed dataframe
# creates an encoded dataframe from them
# Returns encoded dataframe, the input dataframe containing unencoded data
# and the classifications dataframe storing true classes of each sample
def get_encoded_df():
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
            model_input_df.to_csv(model_input_csv_path, index=False)
            classifications_csv_path = os.getcwd() + "/Processed_Dataframes/classifications.csv"
            classifications_df.to_csv(classifications_csv_path, index=False)

    # If in debug mode, read the dataframes directly from csv
    else:
        model_input_csv_path = os.getcwd() + "/Processed_Dataframes/model_input.csv"
        model_input_df = pd.read_csv(model_input_csv_path)
        classifications_csv_path = os.getcwd() + "/Processed_Dataframes/classifications.csv"
        classifications_df = pd.read_csv(classifications_csv_path)

    ### Encode the categorical attributes to obtain numeric representations
    ### that can be used for model training
    encoded_df = pd.DataFrame()
    # Create list of categorical attributes that need to be encoded
    categorical_attributes = ['Location', 'Type', 'Place', 'Person', 'Day of Week']
    # For each of the categorical variables, fit an encoder and transform
    # to obtain one hot encoding
    encoder_dict = {}
    for attr in categorical_attributes:
        # Get the names of all unique classes
        unique_class_names = model_input_df[attr].unique()
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

        # Store one hot encoded attribute in dataframe
        encoded_attr_df = pd.DataFrame(encoded_attr_values.toarray(), 
                                       columns=unique_class_names)
        # Add this encoded attribute df to the encoded dataframe
        encoded_df = pd.concat([encoded_df, encoded_attr_df], axis=1)

    ## Since hour is continuous and cyclical, encode it using sine and
    ## cosine function (e.,g since hour 24 is closer to 1 than 22)
    hours_in_day = 24
    encoded_df['cos_hour'] = np.cos(2*np.pi*model_input_df['Hour of Day'].astype(float)/hours_in_day)
    encoded_df['sin_hour'] = np.sin(2*np.pi*model_input_df['Hour of Day'].astype(float)/hours_in_day)


    ## Convert Duration attribute as number of seconds
    encoded_df['Duration'] = ''
    for i in range(len(model_input_df.index)):
        time_str = model_input_df.loc[i, 'Duration']
        time_in_sec = convert_time_to_secs(time_str)
        encoded_df.at[i, 'Duration'] = time_in_sec
    
    return encoded_df, model_input_df, classifications_df



