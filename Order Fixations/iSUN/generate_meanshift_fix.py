import numpy as np
import scipy.io
import matplotlib.image as mpimg
import isunapi as api
import time
import sys

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def save_training_ms_fixations(train):
	'''
	#Load training data
	iSUN_PATH='../../Datasets/iSUN/'
	MAT_TRAIN_SUN = iSUN_PATH + 'training.mat'
	train = scipy.io.loadmat(MAT_TRAIN_SUN)
	'''
	# Create array for new dataset
	n_samples=train['training'].shape[0]
	output_array = np.zeros((n_samples), dtype=np.ndarray)

	for i in range(n_samples-1):
		# Obtain image info
		name, fixations, locations, timestamps, size = api.get_image_data(train, i) 

		# Obtain meanshift fixations
		meanshift_fixations = api.get_ordered_fixations_with_meanshift(locations, timestamps, [32,32,200]) #Bandwidth
	
		output_array[i] = meanshift_fixations

		print 'Fixations of image '+str(i)+' done!'

	timestr = time.strftime("%Y%m%d-%H%M%S")
	scipy.io.savemat(iSUN_PATH +'ms_training_fixations/meanshift_fixations_'+timestr, {'meanshift_fixations':output_array})
    
def one_position_ms_fix(trainpos,mattosave,i):
	name, fixations, locations, timestamps, size = api.get_image_data(train, i)
	meanshift_fixations = api.get_ordered_fixations_with_meanshift(locations, timestamps, [32,32,200])
	mattosave['meanshift_fixations'][0][i]=meanshift_fixations
	scipy.io.savemat('meanshift_fixations_20180220-203846', mattosave)


#Load training data
iSUN_PATH='../../Datasets/iSUN/'
MAT_TRAIN_SUN = iSUN_PATH + 'training.mat'
train = scipy.io.loadmat(MAT_TRAIN_SUN)
save_training_ms_fixations(train)

#modify meanshift fixations for only one image
''' 
ms_PATH='meanshift_fixations_20180220-203846.mat'
ms = scipy.io.loadmat(ms_PATH)
one_position_ms_fix(train,ms,5999)
'''
