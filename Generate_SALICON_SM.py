import scipy.io as io
from scipy import ndimage
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import os
import png


#fix_train_folder='../Datasets/SALICON/SALICON_fixations/train/'
#fix_val_folder='../Datasets/SALICON/SALICON_fixations/val/'

fix_train_folder='../Datasets/SALICON/fixations2015/train/'
fix_val_folder='../Datasets/SALICON/fixations2015/val/'

#Read mat files:
fix_train=[fix_train_folder + f for f in os.listdir(fix_train_folder) if f.endswith('.mat')]
fix_val = [fix_val_folder + f for f in os.listdir(fix_val_folder) if f.endswith('.mat')]


def merge_fixations(fixations):
    merged_fixations = np.zeros((1,2)) # Creates a matrix with one value [[0,0]] has size (1,2)

    for i in range(fixations.shape[0]):
        merged_fixations = np.concatenate((merged_fixations, fixations[i][0]))

    merged_fixations = np.delete(merged_fixations, 0, axis=0) #deletes the [0,0] value
    return merged_fixations

def merge_fixations_sal2015(fixations):
    fixations=fixations[0]
    merged_fixations = np.zeros((1,2)) # Creates a matrix with one value [[0,0]] has size (1,2)

    for i in range(fixations.shape[0]):
        merged_fixations = np.concatenate((merged_fixations, fixations[i]))

    merged_fixations = np.delete(merged_fixations, 0, axis=0) #deletes the [0,0] value
    return merged_fixations

def build_probability_map(merged_locations, size, sigma):

    #create saliency map
    sal_map = np.zeros((size[0],size[1]),dtype=float)

    for x,y in merged_locations:
        if x < size[1] and y < size[0] :
            sal_map[int(y)-1][int(x)-1] = 1

    sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
    prob_map = sal_map / np.sum(sal_map.flatten())

    return prob_map

### WSM

def weighted_points(points, params, weight_type):

    n_points = points.shape[0]


    if weight_type == 'exp':
        x = np.linspace(0,1,100)
        weights = np.exp(-params * x)

    elif weight_type == 'linear':
        x = np.linspace(0,1,100)
        weights = np.maximum(params * x + 1, 0)

    elif weight_type == 'exp_asc':
        x = np.linspace(0,1,100)
        weights = np.exp(params* x)-0.9

    weight = weights[:n_points].reshape(n_points,1)

    points_with_weights = np.hstack((points, weight))

    return points_with_weights

def weighted_points_all_workers(image_fixations, params, weight_type='exp'):

    '''
    Recieves an array of points of shape (x,y) and returns and adds a third column (x,y,w) that
    represents the weight

    '''
    worker_fixations_weighted = 0

    for w in range(image_fixations.shape[0]):
        worker_fixations = image_fixations[w][0]
        w_p = weighted_points(worker_fixations, params, weight_type=weight_type)

        if type(worker_fixations_weighted) == int:
            worker_fixations_weighted = w_p
        else:
            worker_fixations_weighted = np.vstack((worker_fixations_weighted, w_p))

    return worker_fixations_weighted

def weighted_points_all_workers_sal2015(image_fixations, params, weight_type='exp'):

    '''
    Recieves an array of points of shape (x,y) and returns and adds a third column (x,y,w) that
    represents the weight

    '''
    worker_fixations_weighted = 0

    for w in range(image_fixations.shape[0]):
        worker_fixations = image_fixations[w]
        w_p = weighted_points(worker_fixations, params, weight_type=weight_type)

        if type(worker_fixations_weighted) == int:
            worker_fixations_weighted = w_p
        else:
            worker_fixations_weighted = np.vstack((worker_fixations_weighted, w_p))

    return worker_fixations_weighted


def build_weighed_probability_map(merged_locations, size,sigma=19):  #This map is normalized as a probability map
    '''
        Build weighted probability map for one worker

    '''

    #create saliency map
    sal_map = np.zeros((size[0],size[1]))

    for x,y,w in merged_locations:
        if x < size[1] and y < size[0] :
            sal_map[int(y)-1][int(x)-1] = w


    sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
    prob_map = sal_map / np.sum(sal_map.flatten())
    return prob_map

def create_and_save_maps_v1(fix_mats,trainorval):
    for i in range(len(fix_mats)):
        fix=io.loadmat(fix_mats[i])
        fixations=fix['gaze']['fixations']
        size=fix['resolution'][0]
        name=fix['image'][0]
        merged_fixations=merge_fixations(fixations)
        salmap=build_probability_map(merged_fixations, size, sigma=19)
        salmap= (salmap * 255).astype(np.uint8)
        #io.savemat('SALMAPS/'+ trainorval +'/'+ name + '.mat', {'N': salmap})
        png.from_array(salmap, 'L').save('SALMAPS/'+ trainorval +'/'+ name + '.png')
        print 'Image '+ str(i) + ' out of '+ str(len(fix_mats)) + ' ( '+ trainorval +')'

def create_and_save_weighted_maps_v1(fix_mats,trainorval):
    for i in range(len(fix_mats)):
        fix=io.loadmat(fix_mats[i])
        fixations=fix['gaze']['fixations']
        size=fix['resolution'][0]
        name=fix['image'][0]
        weighted_fixations = weighted_points_all_workers(fixations, 15)
        w_prob_map = build_weighed_probability_map(weighted_fixations, size, sigma=19)
        w_prob_map= (w_prob_map * 255).astype(np.uint8)
        #io.savemat('wSALMAPS/'+ trainorval +'/'+ name + '.mat', {'W': w_prob_map})
        png.from_array(w_prob_map, 'L').save('wSALMAPS/'+ trainorval +'/'+ name + '.png')
        print 'Image '+ str(i) + ' out of '+ str(len(fix_mats)) + ' ( '+ trainorval +')'

def build_weighted_sal_map(merged_locations, size,sigma=19):
    '''
        Build weighted probability map for one worker

    '''

    #create saliency map
    sal_map = np.zeros((size[0],size[1]))


    for x,y,w in merged_locations:
        if x < size[1] and y < size[0] :
            sal_map[int(y)-1][int(x)-1] = w


    sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
    # Normalize gaussian
    sal_map -= np.min(sal_map)
    sal_map /= np.max(sal_map)
    return sal_map

def build_saliency_map(merged_locations, size, sigma=19):

    #create saliency map
    sal_map = np.zeros((size[0],size[1]))

    for x,y in merged_locations:
        if x < size[1] and y < size[0] :
            sal_map[int(y)-1][int(x)-1] = 1


    sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)

    # Normalize gaussian
    sal_map -= np.min(sal_map)
    sal_map /= np.max(sal_map)

    return sal_map

def create_and_save_maps_v2(fix_mats,trainorval):
    for i in range(len(fix_mats)):
        fix=io.loadmat(fix_mats[i])
        fixations=fix['gaze']['fixations']
        size=fix['resolution'][0]
        name=fix['image'][0]
        #merged_fixations=merge_fixations(fixations)
        merged_fixations=merge_fixations_sal2015(fixations)
        salmap=build_saliency_map(merged_fixations, size, sigma=19)
        salmap= (salmap * 255).astype(np.uint8)
        print 'Image '+ str(i) + ' out of '+ str(len(fix_mats)) + ' ( '+ trainorval +')'
        #io.savemat('SALMAPS_v2/'+ trainorval +'/'+ name + '.mat', {'N': salmap})
        png.from_array(salmap, 'L').save('SALMAPS2015/'+ trainorval +'/'+ name + '.png')

def create_and_save_weighted_maps_v2(fix_mats,trainorval):
    for i in range(len(fix_mats)):
        fix=io.loadmat(fix_mats[i])
        fixations=fix['gaze']['fixations']
        size=fix['resolution'][0]
        name=fix['image'][0]
        #weighted_fixations = weighted_points_all_workers(fixations, 15) v2
        weighted_fixations = weighted_points_all_workers_sal2015(fixations[0], 40) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        w_sal_map = build_weighted_sal_map(weighted_fixations, size, sigma=19)
        w_sal_map= (w_sal_map * 255).astype(np.uint8)
        #io.savemat('wSALMAPS_v2/'+ trainorval +'/'+ name + '.mat', {'W': w_sal_map})
        print 'Image '+ str(i) + ' out of '+ str(len(fix_mats)) + ' ( '+ trainorval +')'
        png.from_array(w_sal_map, 'L').save('wSALMAPS2015_40/'+ trainorval +'/'+ name + '.png') #v2

def create_fix_map(fixations,size): #Fixation maps for the evaluation process
    merged_fixations=merge_fixations(fixations)#create saliency map
    fix_map = np.zeros((size[0],size[1]),dtype=float)

    for x,y in merged_fixations:
        if x < size[1] and y < size[0] :
            fix_map[int(y)][int(x)] = 1
    return fix_map

def create_and_save_fix_maps(fix_mats):
    for i in range(len(fix_mats)):
        fix=io.loadmat(fix_mats[i])
        fixations=fix['gaze']['fixations']
        size=fix['resolution'][0]
        name=fix['image'][0]
        fix_map=create_fix_map(fixations,size)
        fix_map= (fix_map * 255).astype(np.uint8)
        #io.savemat('fix_SALMAPS/val/'+ name + '.mat', {'I': fix_map})
        png.from_array(fix_map, 'L').save('fix_SALMAPS/val/'+ name + '.png')
        print 'Image '+ str(i) + ' out of '+ str(len(fix_mats))

##V1 #Probability map normalization

#NSM
#create_and_save_maps(fix_train,'train')
#create_and_save_maps_v1(fix_val,'val') #Used to compute KLDIV change to 30 ?

#WSM
#create_and_save_weighted_maps(fix_train,'train')
#create_and_save_weighted_maps_v1(fix_val,'val') #Used to compute KLDIV

##V2 gaussian normalization

#NSM
#create_and_save_maps_v2(fix_train,'train')
#create_and_save_maps_v2(fix_val,'val')

#WSM

create_and_save_weighted_maps_v2(fix_train,'train')
create_and_save_weighted_maps_v2(fix_val,'val')

#Create fixation maps (validation set)
#create_and_save_fix_maps(fix_val)
