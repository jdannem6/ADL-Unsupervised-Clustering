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
from sklearn.cluster import KMeans
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
    return new_arr, row_indices



if __name__ == "__main__":
    # Load the encoded dataframe
    encoded_df, input_df, class_df = get_encoded_df()
    print(encoded_df)

    ## Perform PCA on encoded dataframe to reduce number of features to a 
    ## smaller, more meaningful set of features
    # First, normalize the data to ensure variance is distributed well
    # across features

    ## Consider using StandardScalar instead of this method of normalization
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
    
    # plt.plt(pca_model.explained_variance_ratio_)
    # plt.ylabel('Explained Variance')
    # plt.xlabel('Components')
    # plt.show()


pca_result = pca_model.fit_transform(norm_encoded_df)
print('Explained variation per principal component: {}'.format(pca_model.explained_variance_ratio_))

# >> Explained variation per principal component: [0.36198848 0.1920749 ]

print('Cumulative variance explained by n principal components: {:.2%}'.format(np.sum(pca_model.explained_variance_ratio_)))

## Apply K-Means
kmeans_model = KMeans(n_clusters=11)
print(pca_df)
kmeans_model.fit(pca_df)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = pca_df.index.values
cluster_map['cluster'] = kmeans_model.labels_
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
# found by rearranging rows to map 
# clusters to the appropriate classes
print(maximize_diagonal(confusion_matrix)[0])
    

# Create and train a t-SNE model to decrease the dimensionality from 35 to 3
tsne = TSNE(n_components=3, random_state=0)
proj_3d = tsne.fit_transform(norm_encoded_df)

# Display the flattened data on a 3D scatterplot with colors representing clusters
fig = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=kmeans_model.labels_
)

fig.update_traces(marker_size=4)
fig.show()

# Create and train another t-SNE model, but this time map to 2D
tsne = TSNE(n_components=2, random_state=0)
proj_2d = tsne.fit_transform(norm_encoded_df)

# Show a 2D scatter plot of the clustered data
plt.scatter(proj_2d[0], proj_2d[1], c=kmeans_model.labels_)
plt.show()

## Confusion matrix indicates result is not that good for person A but decent for person B
# different results each time you run