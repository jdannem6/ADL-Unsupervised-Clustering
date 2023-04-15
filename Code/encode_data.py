###############################################################################
### Script name:     encode_data       .py                                  ###
### Script function: encodes the attributes of preprocessed dataframe such  ###
###                  that each value is numerical can be used to train      ###
###                  clustering model                                       ###
### Authors:         Justin Dannemiller, Keith Machina, and Bailey Wimer    ###
### Last Modified: 04/14/2023                                               ###
###############################################################################

from preprocess_data import preprocess_datasets




if __name__ == "__main__":
    
    resulting_dataframe = preprocess_datasets()
    print(resulting_dataframe)