###############################################################################
### Script name:     preprocess_data.py                                  ###
### Script function: processes the original datset to produce more          ###
###                  meaningful features from the time start and time end   ###
###                  attributes                                             ###
### Authors:         Justin Dannemiller, Keith Machina, and Bailey Wimer    ###
### Last Modified: 04/14/2023                                               ###
###############################################################################

import pandas as pd # Needed for data reading 
from datetime import datetime # Needed for numerous time-dependent attributes to be created 

## Takes names of .txt or .csv files storing data and reads their stored data 
## into pandas dataframe which is returned as output
# file_name is string
# names_of_columns is a list of strings where each string gives names of a
# specific column
def load_data(file_name, names_of_columns):
    # Read the data into a dataframe, skip first 2 rows (header and 
    # header delimiter)
    dataframe = pd.read_csv(file_name, delim_whitespace=True, header = None, skiprows=2)
    # Specify the names of columns in dataframe
    dataframe.columns = names_of_columns
    return dataframe

## Adds additional time-based attributes to the dataframe based upon the 
## start datetime and end datetime of object
## Adds activity duration, hour of day of actviity, and day of week
def add_time_based_attrs(dataframe):
    dataframe['Duration'] = ''
    dataframe['Hour of Day'] = ''
    dataframe['Day of Week'] = ''
    ## Calculate duration of each activity and insert value into dataframe
    for df_index in range(len(dataframe.index)):
        # Concatenate time and day into single string
        start_datetime_str = dataframe.loc[df_index]['Start Day'] + " " + \
                        dataframe.loc[df_index]['Start Time']
        end_datetime_str = dataframe.loc[df_index]['End Day'] + " " + \
                        dataframe.loc[df_index]['End Time'] 
        # Convert string forms of datetime into datetime objects
        input_format = '%Y-%m-%d %H:%M:%S'
        start_datetime_obj = datetime.strptime(start_datetime_str, input_format)
        end_datetime_obj = datetime.strptime(end_datetime_str, input_format)

        # Calculate duration of activity
        activity_duration = end_datetime_obj - start_datetime_obj
        dataframe.at[df_index, 'Duration'] = activity_duration

        ## Hour of day and day of week are based on starting time
        # Get and store activity's hour of day
        dataframe.at[df_index, 'Hour of Day'] = start_datetime_obj.hour
        # Get and store activity's day of week
        dataframe.at[df_index, 'Day of Week'] = start_datetime_obj.strftime("%A")

    return dataframe

### Store the processed data 
## Either store in a dictionary or in another .txt file
# Dictionary would be advantageous if we make all of these procedures into
# functions to be called by other scripts

# Performs all preprocessing steps for dataframe and returns 
def preprocess_datasets():
    ### Load ADL Data and Sensor data from txt files
    path_to_data_files = "Data/UCI ADL Binary Dataset/"
    ## Set file names
    personA_ADL_file = path_to_data_files + "OrdonezA_ADLs.txt"
    personB_ADL_file = path_to_data_files + "OrdonezB_ADLs.txt"
    personA_sensor_file = path_to_data_files + "OrdonezA_Sensors.txt"
    personB_sensor_file = path_to_data_files + "OrdonezB_Sensors.txt"

    ## Load the files into dataframes
    # Define names of columns for both types of input files
    ADLs_column_names= []
    ADLs_column_names.append("Start Day")
    ADLs_column_names.append("Start Time")
    ADLs_column_names.append("End Day")
    ADLs_column_names.append("End Time")
    # make sensors column name a shallow copy of adls column names list
    sensors_column_names = ADLs_column_names[:] # They have same first two columns
    ADLs_column_names.append("Activity")
    sensors_column_names.append("Location")
    sensors_column_names.append("Type")
    sensors_column_names.append("Place")


    # Read the files and store into dataframes
    personA_ADL_df = load_data(personA_ADL_file, ADLs_column_names)
    personB_ADL_df = load_data(personB_ADL_file, ADLs_column_names)
    personA_sensor_df = load_data(personA_sensor_file, sensors_column_names)
    personB_sensor_df = load_data(personB_sensor_file, sensors_column_names)

    ### Combine ADL Data and Sensor data into the same dataframe for each 
    ### person
    ## Create shallow copy of sensor dataframe
    personA_df = personA_sensor_df.copy(deep=False)
    personB_df = personB_sensor_df.copy(deep=False)

    # ## Add new column for activity
    # personA_df['Activity'] = ''
    # personB_df['Activity'] = ''

    ## Set Activity values based on those in ADL dataframe
    # Compare the times of the sensor readings to the times of the activities
    # The activity for a given sensor reading is that whose whose time range
    # contains the time range of the sensor row
    # Person A dataframe
    for new_df_index in range(len(personA_df.index)):
        for ADL_df_index in range(len(personA_ADL_df.index)):
            # Determine if time range of a given performed activity contains
            # the time range of sensor reading
            start_day_contained = personA_df.loc[new_df_index, "Start Day"] >= \
                                  personA_ADL_df.loc[ADL_df_index, "Start Day"]
            start_time_contained = personA_df.loc[new_df_index, "Start Time"] >= \
                                   personA_ADL_df.loc[ADL_df_index, "Start Time"]
            end_day_contained = personA_df.loc[new_df_index, "End Day"] <= \
                                personA_ADL_df.loc[ADL_df_index, "End Day"]
            end_time_contained = personA_df.loc[new_df_index, "End Time"] <= \
                                 personA_ADL_df.loc[ADL_df_index, "End Time"]
            if (start_day_contained and start_time_contained and \
                end_day_contained and end_time_contained):
                personA_df.at[new_df_index, 'True Classification'] = \
                        personA_ADL_df.loc[ADL_df_index]['Activity']
                # After activity of sensor reading has been found, there is no need
                # to compare to any other activity
                break
            # Otherwise, if there is no activity to associate with the given sensor
            # reading, then assign it "No activity"
            else:
                personA_df.at[new_df_index, 'True Classification'] = 'No Activity'

    # Person B dataframe
    for new_df_index in range(len(personB_df.index)):
        # # Set activity to "No Activity" as default for cases in which activity can not
        # # be mapped to a sensor sample (happens when ADL data is not available for 
        # # corresponding sensor data)
        # personB_df.at[new_df_index, 'Activity'] = 'No Activity'
        for ADL_df_index in range(len(personB_ADL_df.index)):
            # Determine if time range of a given performed activity contains
            # the time range of sensor reading
            start_day_contained = personB_df.loc[new_df_index, "Start Day"] >= \
                                  personB_ADL_df.loc[ADL_df_index, "Start Day"]
            start_time_contained = personB_df.loc[new_df_index, "Start Time"] >= \
                                   personB_ADL_df.loc[ADL_df_index, "Start Time"]
            end_day_contained = personB_df.loc[new_df_index, "End Day"] <= \
                                personB_ADL_df.loc[ADL_df_index, "End Day"]
            end_time_contained = personB_df.loc[new_df_index, "End Time"] <= \
                                 personB_ADL_df.loc[ADL_df_index, "End Time"]
            if (start_day_contained and start_time_contained and \
                end_day_contained and end_time_contained):
                personB_df.at[new_df_index, 'True Classification'] = \
                        personB_ADL_df.loc[ADL_df_index]['Activity']
                # After activity of sensor reading has been found, there is no need
                # to compare to any other activity
                break
            # Otherwise, if there is no activity to associate with the given sensor
            # reading, then assign it "No activity"
            else:
                personB_df.at[new_df_index, 'True Classification'] = 'No Activity'


    ### Convert time start and time end into more meaningful attriutes
    ## Create new attributes: activity duration, day of week, and hour of day
    ## Add new column for activity duration 
    personA_df = add_time_based_attrs(personA_df)
    personB_df = add_time_based_attrs(personB_df)

    ## Remove the Start Day, Start Time, End Day, and End Time columns as they are
    ## not longer needed
    columns_to_drop = ['Start Day', 'Start Time', 'End Day', 'End Time']
    personA_df = personA_df.drop(columns_to_drop, axis=1)
    personB_df = personB_df.drop(columns_to_drop, axis=1)

    ## Combine the two dataframes into one
    # First add another attribute for the true label of each row
    personA_df["Person"] = 'Person A'
    personB_df["Person"] = 'Person B'

    combined_df = pd.concat([personA_df, personB_df], ignore_index=True)
    return combined_df



    

# if __name__ == "__main__":
#     preprocess_datasets()