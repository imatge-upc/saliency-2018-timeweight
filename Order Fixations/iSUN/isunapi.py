# -*- coding: utf-8 -*-
from scipy import misc
import scipy.io
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import mean_shift as ms
from sklearn.cluster import KMeans
 # Load the ground truth saliency map
def isun_saliency_map(FILENAME):
     saliency = scipy.io.loadmat('data/saliency/'+FILENAME+'.mat')
     mapp =saliency['I']
     return mapp
 # Get data from 'train'
def get_image_data(data, image_id):
     dataset_name = data.keys()[0]
     name = data[dataset_name][image_id]['image'][0][0]
     fixations = data[dataset_name][image_id]['gaze'][0]['fixation'][0]
     locations = data[dataset_name][image_id]['gaze'][0]['location'][0]
     timestamps = data[dataset_name][image_id]['gaze'][0]['timestamp'][0]
     size = data[dataset_name][image_id]['resolution'][0][0]

     return name, fixations, locations, timestamps, size
 # Create and plot a saliency map
def build_saliency_map(merged_locations, size, sigma=19):
     #create saliency map
     sal_map = np.zeros((size[0],size[1]))
     for x,y in merged_locations:
         if x < size[1] and y < size[0] :
             sal_map[int(y-1)][int(x-1)] = 1

     sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
     sal_map -= np.min(sal_map)
     sal_map /= np.max(sal_map)
     return sal_map
 # Plot a saliency map
def plot_saliency_map(merged_locations, size, sigma=19):
     sal_map = build_saliency_map(merged_locations, size, sigma)
     plt.imshow(sal_map)

# ## Merge the locations of all the workers of an image
def merge_locations(locations):
     merged_locations = np.zeros((1,2))

     for i in range(locations.shape[0]):
         merged_locations = np.concatenate((merged_locations, locations[i]))

     merged_locations = np.delete(merged_locations, 0, axis=0)
     return merged_locations


 # Returns the index of the image
def image_index(name, train):
     images = train['training']

     names = []
     for i in range(len(images)):
         names.append(train['training'][i]['image'][0][0])
     image_id = names.index(name)

     return image_id

def sal_map_sequence(locations, timestamps, time_bin, worker=-1):
     # Find the number of bins
     n_bins = 0
     for w in range(len(timestamps)):
         if timestamps[w].max() / time_bin > n_bins:
             n_bins = timestamps[w].max() / time_bin

     n_bins += 1

     # binned_locations < time bin < all the locations for that time bin
     binned_locations = np.ndarray(shape=(n_bins,), dtype=object)
     # Create array inside each time bin to hold all the positions
     for i in range(n_bins):
         binned_locations[i] = np.zeros((1,2))


     # Distribute all locaitons in their correspondent bins
     if worker == -1:
         workers = range(timestamps.shape[0])
     else:
         workers = [worker]
     for w in workers: #for each worker w

         for i in range(len(locations[w])): # for each location/timestamp of the worker

             n_bin = timestamps[w][i][0] / time_bin
             binned_locations[n_bin] =  np.vstack([binned_locations[n_bin],locations[w][i]])


     # Delete the first entry of each bin, because its just 0,0
     for i in range(n_bins):
         binned_locations[i] = np.delete(binned_locations[i], 0, axis=0)

     return binned_locations


 # Draws a gri of saliency maps and time

def sal_map_sequence_grid(locations, timestamps, t_bin, size, worker=-1):

     binned_locations = sal_map_sequence(locations, timestamps, t_bin, worker)
     rows  = len(binned_locations)/float(4)
     rows = math.ceil(rows)

     fig, axs = plt.subplots(int(rows),4, figsize=(15, 7), facecolor='w', edgecolor='k')
     fig.subplots_adjust(hspace = 0, wspace=.00)

     axs = axs.ravel()
     for b in range(len(binned_locations)):

         sal_map = build_saliency_map(binned_locations[b], size)
         axs[b].imshow(sal_map)
         axs[b].set_title('Snapshot t=%d'% b, fontsize=10, fontweight='bold')

 # Draws a grid with the saliency maps of different workers
def sal_map_sequence_compare_workers(locations, timestamps, t_bin, size):
     # Init data
     n_workers = locations.shape[0]
     worker_binned_locations = np.ndarray(shape=(n_workers,), dtype=object)


     # Save bins (with locations) of each worker separately
     for w in range(n_workers):
         worker_binned_locations[w] = sal_map_sequence(locations, timestamps, t_bin, worker=w)
#     # Plot each worker separately
     rows = worker_binned_locations[0].shape[0]
     fig, axs = plt.subplots(rows,n_workers, figsize=(15, 25), facecolor='w', edgecolor='k')
     fig.subplots_adjust(hspace = .0, wspace=.00)
     for w in range(n_workers):
         for b in range(rows):
             sal_map = build_saliency_map(worker_binned_locations[w][b], size)
             axs[b,w].imshow(sal_map)
             if b == 0:
                 axs[b,w].set_title('Worker %d'% w, fontsize=10, fontweight='bold')


def saliency_snapshots_from_fixations(fixations):

     # Find number of bins

     n_bins = 0
     for w in range(len(fixations)):
         if fixations[w].shape[0] > n_bins:
             n_bins = fixations[w].shape[0]


     # binned_locations < time bin < all the locations for that time bin

     binned_fixations = np.ndarray(shape=(n_bins,), dtype=object)


     # Create array inside each time bin to hold all the positions

     for i in range(n_bins):
         binned_fixations[i] = np.zeros((1,2))


     # Assign fixations to bins

     for w in range(len(fixations)): #for each worker w

         for i in range(len(fixations[w])): # for each location/timestamp of the worker

             binned_fixations[i] =  np.vstack([binned_fixations[i],fixations[w][i]])


     # Delete the first entry of each bin, because its just 0,0

     for i in range(n_bins):
         binned_fixations[i] = np.delete(binned_fixations[i], 0, axis=0)

     return binned_fixations

def get_ordered_fixations_with_meanshift(locations,timestamps, bandwidth):

     # Create array to store fixations
     meanshift_fixations = np.copy(locations)

     for i in range(len(locations)):
         X = np.hstack([locations[i], timestamps[i]])

         mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')
         mean_shift_result = mean_shifter.cluster(X, kernel_bandwidth = bandwidth)

         original_points =  mean_shift_result.original_points
         shifted_points = mean_shift_result.shifted_points
         cluster_assignments = mean_shift_result.cluster_ids
         meanshift_fixations[i] = unique_rows(shifted_points[:,:2])


     return meanshift_fixations


def unique_rows(a):
     a = a.astype(int)
     mask = np.ones(len(a), dtype=bool) # True indica els valors que ens quedem

     for i in range(len(a)):
         if i != 0:
             if np.equal(a[i], a[i-1])[0]:
                 mask[i] = False

     return a[mask]

def add_index_feature(array_):


     array = np.copy(array_)

     # Add time column

     for i in range(array.shape[0]):
         time = np.asarray(range(array[i].shape[0])).reshape((array[i].shape[0],1))
         array[i] = np.hstack([array[i], time])


     # Add time column

     array_with_time = np.zeros((1,3))

     for i in range(array.shape[0]):
         array_with_time = np.concatenate((array_with_time, array[i]))

     array_with_time = np.delete(array_with_time, 0, axis=0)
     return(array_with_time)



def order_kmeans_labels(labels, merged_fixations):
     # Find centroid mean time
     n_clusters = max(labels)+1
     centroid_mean_times = [0] * n_clusters

     for c in range(n_clusters):

         indices = np.argwhere(labels == c)

         centroid_mean_times[c] = np.mean(merged_fixations[indices, 2])

#     #
#     # clase falsa 0   [ mean time ]
#     # clase falsa 1   [ mean time]
#     #


     # Reorder points
     centroid_mean_times = np.asarray(centroid_mean_times).reshape((len(centroid_mean_times), 1))
     i = np.asarray(range(n_clusters)).reshape((n_clusters,1))
     centroid_mean_times = np.hstack([centroid_mean_times, i])

     i = centroid_mean_times[:, 0].argsort()
     centroid_mean_times = centroid_mean_times[i]

     # New labels
     new_labels = np.copy(labels)
     for c in range(n_clusters):
         indices = np.argwhere(labels == centroid_mean_times[c,1]) # Where there was c
         new_labels[indices] = c

     return new_labels

def plot_kmeans(img, data,labels, titles, axs, size):

     label_to_color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y']
     label_to_marker = ['s','s' ,'s', 's', 's', 's', 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^', '^', '*','*' ,'*', '*', '*', '*', '*', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'd', 'd', 'd', 'd', 'd', 'd']

     for i in range(len(data)): # For each image

         x,y = i / size[0] , i % size[0]

         # find n clusters
         n_clusters = max(labels[i])+1

         # List of names
         clusters = [0]*n_clusters
         legend = []
         for c in range(n_clusters):
             legend.append("c%d" % c)


         for j in range(data[i].shape[0]): # for each cluster
             clusters[labels[i][j]] = axs[x,y].scatter(data[i][j][0], data[i][j][1], c=label_to_color[labels[i][j]],
                                   marker=label_to_marker[labels[i][j]], s=100)

         axs[x,y].imshow(img)
         axs[x,y].set_title(titles[i], fontsize=20, fontweight='bold')
         axs[x,y].legend(clusters, legend)


def merged_fixations_with_scaled_index(meanshift_fixations, scale):
     mf = add_index_feature(meanshift_fixations)
     mf[:,2] *= scale
     return mf


def get_kmeans(meanshift_fixations, scale_param):
     data = []
     labels = []
     titles = []
     n_clusters = max([x.shape[0] for x in meanshift_fixations])

     for s in scale_param:
         # Store the fixations
         fix_with_index = merged_fixations_with_scaled_index(meanshift_fixations, s)
         data.append(fix_with_index)

         # Store the labels
         kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fix_with_index)
         kmeans_labels = order_kmeans_labels(kmeans.labels_, fix_with_index)
         labels.append(kmeans_labels)

        # Store title
         titles.append("Scale = %d" % s)


     return data, labels, titles
