###############################################################################
### Script name:     perform_clustering.py                                  ###
### Script function: Takes the encoded dataframe, performs PCA upon it and  ###
###                  performs unsupervised clustering on resulting          ###
###                  components                                             ###
### Authors:         Justin Dannemiller, Keith Machina, and Bailey Wimer    ###
### Last Modified: 04/14/2023                                               ###
###############################################################################

from encode_data import get_encoded_df
from sklearn.decomposition import PCA # Needed for performing PCA on encoded df
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import plotly.express as px
import os

## Given a matrix A, rearranges its rows to maximize the sum of its diagonal elements.
#  Returns the rearranged matrix.
def maximize_diagonal(A):
    # Find the number of elements in each row of A
    # and use that number to make an array of row indices
    # that will be used to swap rows around
    n = A.shape[0]
    row_indices = np.arange(n)
    max_sum = 0

    # For each row in the array try to switch positions
    # to calculate the sum of the diagonal in every configuration
    for i in range(n):
        for j in range(i+1, n):
            new_row_indices = np.copy(row_indices)

            # Swap indices
            new_row_indices[i], new_row_indices[j] = new_row_indices[j], new_row_indices[i]
            new_sum = 0

            # Calculate the sum of the diagonal with the swap
            for x in range(n):
                new_sum += A[new_row_indices[x]][x]

            # Track the largest sum and update the row indices
            if new_sum > max_sum:
                max_sum = new_sum
                row_indices = new_row_indices

    # Put the array in the order of the maximum diagonal sum and return it
    new_arr = np.zeros(A.shape)
    for n, x in enumerate(row_indices):
        new_arr[n] = A[x]
    return new_arr, max_sum

# Given a set of labels for the pca_df dataframe,
# generates the confusion matrix as well as
# a 2d and 3d projection of the clusters
def test_and_display_model(labels):
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = pca_df.index.values
    cluster_map['cluster'] = labels
    print(cluster_map['cluster'])

    cluster_map["True Classification"] = class_df['True Classification']

    cluster_map_csv_path = os.getcwd() + "/Processed_Dataframes/kmeans_result.csv"
    cluster_map.to_csv(cluster_map_csv_path, index=False)
    ## Calculate confusion matrix
    # index map simply maps the classification names onto an index
    # this is useful for confusion matrix calculation
    index_map = list(set(class_df['True Classification']))

    confusion_matrix = np.zeros((11,11))
    for i in range(len(cluster_map.index)):
        index_of_pred = cluster_map.loc[i, 'cluster']
        actual_class = cluster_map.loc[i, 'True Classification']
        index_of_actual = index_map.index(actual_class)
        confusion_matrix[index_of_actual][index_of_pred] +=1

        
    print("Confusion Matrix:")
    # Prints the optimized confusion matrix
    # found by rearranging confusion matrix to map 
    # clusters to the appropriate classes
    results = maximize_diagonal(confusion_matrix)
    print(results[0])
    print(len(pca_df))
    print("Accuracy:", results[1]/len(pca_df))
        

    # Create and train a t-SNE model to decrease the dimensionality from 35 to 3
    tsne = TSNE(n_components=3, random_state=0)
    proj_3d = tsne.fit_transform(norm_encoded_df)

    # Display the flattened data on a 3D scatterplot with colors representing clusters
    fig = px.scatter_3d(
        proj_3d, x=0, y=1, z=2,
        color=labels
    )

    fig.update_traces(marker_size=4)
    fig.show()

    # Create and train another t-SNE model, but this time map to 2D
    tsne = TSNE(n_components=2, random_state=0)
    proj_2d = tsne.fit_transform(norm_encoded_df)

    # Show a 2D scatter plot of the clustered data
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c=labels)
    plt.show()


if __name__ == "__main__":
    # Load the encoded dataframe
    encoded_df, input_df, class_df = get_encoded_df()
    print(encoded_df)

    ## Perform PCA on encoded dataframe to reduce number of features to a 
    ## smaller, more meaningful set of features
    # First, normalize the data to ensure variance is distributed well
    # across features
    desired_feature_count = 24
    norm_encoded_df = (encoded_df - encoded_df.mean())/encoded_df.std()
    print(norm_encoded_df)
    # Create PCA model
    pca_model = PCA(n_components=desired_feature_count)
    # Fit to desired feature count
    pca_model.fit(norm_encoded_df)

    # Peform PCA on encoded dataframe and store as separate dataframe
    pca_df = pd.DataFrame(pca_model.transform(norm_encoded_df), 
                          columns=['Feature %s' % i 
                                   for i in range(desired_feature_count)])


    # fit pca model to normalized encoded dataframe and apply pca it get pca
    # component results
    pca_result = pca_model.fit_transform(norm_encoded_df)

    ## Apply K-Means with same number of clusters as total number of activities
    # Added the following hyper-parameters: init, n_init random_state, max_iter, algothm
    # n_init determines the number of times the algorithm will run of the different cluster centroid points initiate
    #random_state set to 1234 helps us reproduce the same results while running the model on different occasions. 
    # This is for reproducability purposes. max_iter set to 1000 means that the model will run for 600 iterations in a 
    # single run. Default value is 300. the algorithm elkan has many advantages including its fast speed of convergence.
    kmeans_model = KMeans(n_clusters=11, n_init='auto', init='k-means++', random_state=1234, max_iter=300, algorithm='elkan') 
    kmeans_model.fit(pca_df)
    test_and_display_model(kmeans_model.labels_)

    ## Applying the Gaussian Mixture Model.
    # n_components set to 10 means that there are 10 distributions making up our model which can be 
    # indirectly translated as the elements we are trying to cluster. This is not necesarrily true in all cases.
    #warm start set to True enables reuse of the learned model from previous training instances.
    gmm_model = GaussianMixture(n_components=11, random_state=42).fit(pca_df) 
    test_and_display_model(gmm_model.predict(pca_df))


    ## Applying MeanShift Algorithm
    #the bandwith value determines the number of clusters that will be identifies. 
    #Its is not direct like in KMeans. A bandwidth value results in large number of clusters being identified
    #The inverse is also true, a smaller bandwith value results in a larger number of clusters identified.
    #bin seeding set to true speeds up convergence because the model doesnt strive to initialize many seeds
    meanshift_model = MeanShift(bandwidth=6, cluster_all=True, bin_seeding=True, max_iter=1000).fit(pca_df) 
    test_and_display_model(meanshift_model.labels_)
