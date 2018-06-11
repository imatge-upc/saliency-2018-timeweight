from scipy import misc
import scipy.io as io
import time
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
''' #To know how much weight the file has
import os
#resize Meanshift_fixations.mat files
MAT_FILE='meanshift_fixations_20170316-031700.mat'
print os.path.getsize(MAT_FILE)
exit()
'''

# Get data from 'train'
def get_image_data(data, image_id):

    dataset_name = next(iter(data))

    name = data[dataset_name][image_id]['image'][0][0]
    fixations = data[dataset_name][image_id]['gaze'][0]['fixation'][0]
    locations = data[dataset_name][image_id]['gaze'][0]['location'][0]
    timestamps = data[dataset_name][image_id]['gaze'][0]['timestamp'][0]
    size = data[dataset_name][image_id]['resolution'][0][0]

    return name, fixations, locations, timestamps, size

def resize_image_data(resized_data,data,image_id,img_rows=480,img_cols=640):

    name, fixations, locations, timestamps, size = get_image_data(data, image_id)

    old_rows=size[0]
    old_cols=size[1]
    new_size=[img_rows,img_cols]

    new_locations=locations.copy()
    for us in range(locations.shape[0]): #All users
        for l in range(locations[us].shape[0]): # All fixations
            new_locations[us][l][0]=(float(locations[us][l][0])/old_cols)*img_cols
            new_locations[us][l][1]=(float(locations[us][l][1])/old_rows)*img_rows

    new_fixations=fixations.copy()
    for u in range(fixations.shape[0]): #All users
        for f in range(fixations[u].shape[0]): # All fixations
            new_fixations[u][f][0]=(float(fixations[u][f][0])/old_cols)*img_cols
            new_fixations[u][f][1]=(float(fixations[u][f][1])/old_rows)*img_rows


    dataset_name = next(iter(resized_data))

    resized_data[dataset_name][image_id]['gaze'][0]['fixation'][0]=new_fixations
    resized_data[dataset_name][image_id]['gaze'][0]['location'][0]=new_locations
    size = resized_data[dataset_name][image_id]['resolution'][0][0]=new_size

def resize_ms_fixations_file(res_data,data,size, img_rows=480, img_cols=640):
    old_rows=size[0]
    old_cols=size[1]
    #print data
    for u in range(data.shape[1]): #All users
        for f in range(data[0][u].shape[0]): # All fixations
            #print data[0][u][f][0]
            res_data[0][u][f][0]=(float(data[0][u][f][0])/old_cols)*img_cols
            res_data[0][u][f][1]=(float(data[0][u][f][1])/old_rows)*img_rows

def resize_mat(path):

    #resize iSUN mat file
    ISUN_MAT_FILE = path
    dataset_mat = io.loadmat(ISUN_MAT_FILE)
    resized_mat=dataset_mat.copy()

    dataset_name=next(iter(dataset_mat)) #Retorna la primersize = resized_data[dataset_name][image_id]['resolution'][0][0]=new_sizea key del dicionari

    for img_id in range(dataset_mat[dataset_name].shape[0]):
        resize_image_data(resized_mat,dataset_mat,img_id)


    timestr = time.strftime("%Y%m%d-%H%M%S")
    io.savemat('resized_'+dataset_name+'_'+timestr, resized_mat)

def resize_images():
    ### Resize imatges
    #imgs_absolute_path='/home/marta/Documents/Recerca_2017/TFG/Datasets/iSUN/images/'
    imgs_absolute_path='images/'
    rez_480x640_imgs='rez_480x640_imgs/'
    images = [imgs_absolute_path + f for f in os.listdir(imgs_absolute_path) if f.endswith('.jpg')]

    for i, path in enumerate(images):
        original_image = cv2.imread(path)
        if original_image is not None :
            res_image=cv2.resize(original_image,(640,480))
            new_path=os.path.basename(path)
            misc.imsave(rez_480x640_imgs+'resized_'+ new_path, res_image)
            #cv2.imwrite(rez_480x640_imgs+'resized_'+ path , res_image)
            print('Image '+ str(i) + ' done!')

def resize_Meanshift_train_fixations_mat(path,ms_file):
    #resize Meanshift_fixations.mat files
    MAT_FILE=ms_file
    ISUN_MAT_FILE = path
    dataset_mat = io.loadmat(ISUN_MAT_FILE)
    dataset_name=next(iter(dataset_mat))

    ms=io.loadmat(MAT_FILE)
    res_ms=ms.copy()
    ms_fixations = ms['meanshift_fixations'][0]

    resized_ms_fix=ms_fixations.copy()

    for img_id in range(0,ms_fixations.shape[0]):
        _, _, _, _, size = get_image_data(dataset_mat, img_id)
        resize_ms_fixations_file(resized_ms_fix[img_id],ms_fixations[img_id],size)

    res_ms['meanshift_fixations'][0]=resized_ms_fix
    timestr = time.strftime("%Y%m%d-%H%M%S")
    io.savemat('ms_training_fixations/resized_meanshift_fixations_20180220-203846', res_ms)

def check_resized_mat_resolution(rez_path,img_rows=480,img_cols=640):
    dataset_mat=io.loadmat(rez_path)
    dataset_name=next(iter(dataset_mat))
    new_size=np.array([img_rows,img_cols],dtype=np.uint32)
    #new_size.astype(uint32)
    print dataset_mat[dataset_name]['resolution'].dtype
    #print dtype()
    dataset_mat[dataset_name]['resolution'] = np.array(dataset_mat[dataset_name]['resolution'], dtype=np.uint32)
    #print dataset_mat[dataset_name].shape[0]
    for image_id in range(dataset_mat[dataset_name].shape[0]):
        if (image_id==125):
            dataset_mat[dataset_name][image_id]['resolution'][0][0] = np.array([img_rows, img_cols],dtype=np.uint32)
            print type(dataset_mat[dataset_name][image_id]['resolution'][0][0][0])
    io.savemat(rez_path,dataset_mat)

path='training.mat'
ms_file='ms_training_fixations/meanshift_fixations_20180220-203846.mat'
#resize_mat(path)
#resize_images()
#resize_Meanshift_train_fixations_mat(path,ms_file)

rez_path='resized_training_20180220-141451.mat'
check_resized_mat_resolution(rez_path)
