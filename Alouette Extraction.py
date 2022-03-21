# Databricks notebook source
# MAGIC %md
# MAGIC # Function definitions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

"""
Functions that are common for each submodule
"""
import glob
import random

def record_loss(df,function_name,subdir_location,
                columns_to_extract=['file_name'],loss_extraction=[]):
    """Generate a dataframe that records loss due to a self-imposed filter or a runtime programming error
    
    :param df: dataframe containing information of which image files did not pass a self-imposed filter or lead to runtime programming errors
    :type df: class: `pandas.core.frame.DataFrame`
    :param function_name: function or self-imposed filter leading to a loss
    :type function_name: str
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param columns_to_extract: list of columns of df to extract, defaults to ['file_name']
    :type columns_to_extract: list, optional
    :param loss_extraction: whether a custom series is to be used to extract selected rows from the dataframe, defaults to []
    :type loss_extraction: class: `pandas.core.series.Series`, optional
    :returns: df_loss_extraction,loss_extraction i.e. dataframe containing file names leading to runtime errors or that do not pass pre-established filters (metadata size, ionogram size) as well as boolean series indicating which row of data to remove (==1)
    :rtype: (class: `pandas.core.frame.DataFrame`,class: `pandas.core.series.Series`)
    """   
    if len(loss_extraction) == 0:
        # function should return NA if there an error
        loss_extraction = df.isna().any(axis=1)
    # Record the files whose extraction was not successful
    df_loss_extraction = df[loss_extraction].copy()
    df_loss_extraction = df_loss_extraction[columns_to_extract]
    df_loss_extraction['func_name'] = function_name
    df_loss_extraction[ 'subdir_name'] = subdir_location
    
    return df_loss_extraction,loss_extraction


def generate_random_subdirectory(regex_subdirectory):
    """Extract random subdirectory
    
    :param regex_subdirectory: regular expression to extract subdirectory paths ex: 'E:/master/R*/[0-9]*/'
    :type regex_subdirectory: str
    :returns: sample_subdirectory: path of random subdirectory
    :rtype: str
    """
    # All the subdirectory i.e. ./R014207948/1743-9/
    list_all_subdirectory = glob.glob(regex_subdirectory)
    
    # Randomly pick a subdirectory
    sample_subdirectory = list_all_subdirectory[random.randint(0,len(list_all_subdirectory) - 1)]
    
    return sample_subdirectory

def generate_random_image_from_subdirectory(subdirectory,regex_images):
    """Extract random raw image from a subdirectory
    
    :param subdirectory: path of subdirectory
    :type subdirectory: str
    :param regex_images: regular expression to extract images ex: '*.png'
    :type regex_images: str
    :returns: sample_subdirectory: path of random raw image in subdirectory
    :rtype: str
    """
    # All the images
    list_all_img = glob.glob(subdirectory+regex_images)
    
    # Randomly pick an image file
    sample_img_file_path = list_all_img[random.randint(0,len(list_all_img) - 1)]
    
    
    return sample_img_file_path

# COMMAND ----------

# MAGIC %md
# MAGIC ##Image Segmentation

# COMMAND ----------

# MAGIC %md
# MAGIC ### extract ionogram from scan

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Code to extract ionogram part of a raw scanned image
"""
# Library imports
import numpy as np


def limits_ionogram(raw_img, row_or_col,
                    starting_index_col=15):
    """Returns the upper and lower limits of the ionogram part of the scan by row or column using mean-based thresholding
    
    :param starting_img: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned image
    :type starting_img: class: `numpy.ndarray`
    :param row_or_col: 0 for column or 1 for row
    :type row_or_col: int
    :param starting_index_col: where the ionogram starting column should be after to protect against cuts, defaults to 15
    :type starting_index_col: int, optional
    :return:  limits[0],limits[-1] i.e. the upper and lower limits of the ionogram part of the scan by row (row_or_col=1) or column (row_or_col=0)
    :rtype: int,int
            
            
    """
    
    # Mean pixel values by by row/col
    mean_values = np.mean(raw_img, row_or_col)
    
    # Normalized mean values
    normalized_mean_values = (mean_values - np.min(mean_values))/np.max(mean_values)
    
    # Threshold is the overall mean value of the entire image
    threshold = np.mean(normalized_mean_values)
    
    if row_or_col == 0:
        #Protect against scans that includes cuts from another ionogram ex:R014207956\2394-1B\51.png 
        limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold and i > starting_index_col]
    else:
        limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold]
        
    return limits[0],limits[-1]


def extract_ionogram(raw_img_array):
    """Extract ionogram part of a raw scanned image and return coordinates delimiting its limits
    
    :param raw_img_array: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned image
    :type raw_img_array: class: `numpy.ndarray`
    :return: (limits, ionogram) i.e. (list of coordinates delimiting the limits of the ionogram part of a raw scanned image formatted as [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit], UTF-8 grayscale 2D array of values ranging from [0,255] representing ionogram part of scanned image)
    :rtype: (list,numpy.ndarray)
    :raises Exception: returns [],np.nan if there is an error
        
    """
    try:

        # Extract coordinate delimiting the ionogram part of the scan
        x_axis_left_limit ,x_axis_right_limit = limits_ionogram(raw_img_array, 0)
        y_axis_upper_limit, y_axis_lower_limit = limits_ionogram(raw_img_array, 1)

        # Extract ionogram part
        ionogram = raw_img_array[y_axis_upper_limit:y_axis_lower_limit,x_axis_left_limit:x_axis_right_limit]
        limits = [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit]
        return (limits, ionogram)
    
    # In case if ionogram extraction fails:
    except: 
        return ([],np.nan)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trim raw metadata

# COMMAND ----------

# -*- coding: utf-8 -*-
%pip install opencv-python
"""
Code to trim extracted raw metadata
"""

# Check for packages and install
import pkg_resources
required = {'opencv-python'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    %pip install *missing

# Library imports
import math

import numpy as np
import cv2


def connected_components_metadata_location(meta,min_count=50, max_count=1000):
    """Use the connected component algorithm to find the location of the metadata
    
    :param meta: binarized UTF-8 2D array of values (0 or 1) array containing metadata
    :type meta: class: `numpy.ndarray`
    :param min_count: minimum number of pixels to be considered metadata dot/num, defaults to 50
    :type min_count: int, optional
    :param max_count: maximum number of pixels to be considered metadata dot/num, defaults to 1000
    :type max_count: int, optional
    :return: metadata labelled by the connected component algorithm ie UTF-8 2D array of values where each value correspond to belonging to a metadata group
    :rtype: class: `numpy.ndarray`
    """
    # Run the connected component algorithm to label the metadata rectangle
    _, labelled = cv2.connectedComponents(meta)
    
    # Dictionary of label:counts
    unique, counts = np.unique(labelled, return_counts = True)
    dict_components = dict(zip(unique, counts ))
    
    # Remove outliers ie pixels not part of metadata
    dict_subset = {}
    dict_outlier = {}
    for k,v in dict_components.items():
        if v > min_count and v < max_count:
            dict_subset[k] = v
        else: 
            dict_outlier[k] = v
            
    key_list_to_remove = list(dict_outlier.keys())
    if len(key_list_to_remove) != 0:
        for k in key_list_to_remove:
            labelled[labelled==k] = 0
            
    return labelled

def grouped_limits_metadata(meta, row_or_col,
            offset_right=20,grouped=10):
    """Returns the upper and lower limits of the metadata part of the raw metadata (rectangle of scan with metadata) by row or column using mean-based thresholding
    
    :param meta: UTF-8 2D array of values  representing metadata
    :type meta: class: `numpy.ndarray`
    :param row_or_col: 0 for column or 1 for row
    :type row_or_col: int
    :param offset_right: offset (number of pixels) to avoid including ionogram, defaults to 20
    :type  offset_right: int, optional
    :param grouped: size (number of pixels) of group, defaults to 10
    :type  grouped: int, optional
    :return:  grouped*limits[0],grouped*limits[-1]] i.e. the grouped upper and lower limits of the raw metadata (rectangle of scan with metadata) by row (row_or_col=1) or column (row_or_col=0)
    :rtype: int,int
    """

    sum_values = np.sum(meta, row_or_col)
    sum_values = sum_values[0:-offset_right]
    mean_values = [sum(sum_values[i:i + grouped])/(meta.shape[row_or_col]) 
                        for i in range(0,len(sum_values),grouped)]
    
    if math.ceil(len(mean_values) / grouped) > len(mean_values):
        mean_values = mean_values.append(sum(sum_values[i + grouped:-1])/grouped)
    
    normalized_mean_values = (mean_values - np.min(mean_values))/(np.max(mean_values)- np.min(mean_values))
    threshold = 0
    limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold ]
 
    return grouped*limits[0],grouped*limits[-1]



def leftside_metadata_trimming(connected_meta,meta_binary,
                               offset = 20, max_width_metadata=300,threshold_mean_ionogran_chunk_left=0.01):
    
    """Metadata trimming protocol (mean-based thresholding) for metadata located on the left of ionograms
    
    :param connected_meta: metadata labelled by the connected component algorithm ie UTF-8 2D array of values where each value correspond to belonging to a metadata group
    :type connected_meta: class: `numpy.ndarray`
    :param meta: binarized UTF-8 2D array of values (0 or 1) array containing metadata
    :type meta: class: `numpy.ndarray`
    :param offset: offset (number of pixels) to avoid including ionogram, defaults to 20
    :type  offset: int, optional
    :param max_width_metadata: maximum width of trimmed rectangles with metadata, defaults to 300
    :type  max_width_metadata: int, optional
    :param threshold_mean_ionogran_chunk_left: minmum mean area to signal the presence of ionogram chunks, defaults to 0.01
    :type  max_width_metadata: int, optional
    :return: trimmed metadata i.e.  trimmed binarized UTF-8 2D array of values (0 or 1) array containing metadata
    :rtype: class: `numpy.ndarray`
    """
    
    h_raw,w_raw = meta_binary.shape
    y_axis_upper_limit_meta, y_axis_lower_limit_meta = grouped_limits_metadata(connected_meta, 1)
    x_axis_left_limit_meta ,x_axis_right_limit_meta = grouped_limits_metadata(connected_meta, 0)
    
    offset_right = offset

    while abs(x_axis_right_limit_meta - x_axis_left_limit_meta) > max_width_metadata:
        x_axis_left_limit_meta ,x_axis_right_limit_meta = grouped_limits_metadata(connected_meta, 0,offset_right)
        offset_right = offset_right + 10
        
    
    y_axis_upper_limit_meta_offset = max(y_axis_upper_limit_meta-offset,0)
    y_axis_lower_limit_meta_offset = min(y_axis_lower_limit_meta+ offset,h_raw)
    
    x_axis_left_limit_meta_offset = max(x_axis_left_limit_meta-offset,0)
    x_axis_right_limit_meta_offset = min(x_axis_right_limit_meta+ offset,w_raw)
    
    trimmed_metadata = meta_binary[y_axis_upper_limit_meta_offset:y_axis_lower_limit_meta_offset,
                                   x_axis_left_limit_meta_offset:x_axis_right_limit_meta_offset]
    
    #ionogram chunk on left
    if np.mean(np.mean(trimmed_metadata[:,0:offset+1],0)) != 0 and np.mean(np.mean(trimmed_metadata[:,offset+1:2*offset+1],0)) < threshold_mean_ionogran_chunk_left:
        x_axis_left_limit_meta ,_ = grouped_limits_metadata(trimmed_metadata[:,offset+1::], 0)
        trimmed_metadata = meta_binary[y_axis_upper_limit_meta_offset :y_axis_lower_limit_meta_offset ,
                                   x_axis_left_limit_meta :x_axis_right_limit_meta_offset ]
        
    return trimmed_metadata



def bottomside_metadata_trimming(connected_meta,opened_meta,
                                 h_window=100,w_window=700,starting_y = 0, starting_x=15,step_size=10,trim_if_small=10):

    '''Metadata trimming protocol (sliding windowing) for metadata located on the bottom of ionograms
    
    :param connected_meta: metadata labelled by the connected component algorithm ie UTF-8 2D array of values where each value correspond to belonging to a metadata group
    :type connected_meta: class: `numpy.ndarray`
    :param opened_meta:  UTF-8 2D array of values representing raw metadata after morphological operations including opening
    :type opened_meta: nclass: `numpy.ndarray`
    :param h_window: height of sliding window, defaults to 100
    :type h_window: int, optional
    :param w_window: width of sliding window, defaults to 700
    :type w_window: int, optional
    :param starting_y: by how many pixels from the top to start windowing process, defaults to 0
    :type starting_y: int, optional
    :param starting_x: by how many pixels from the left to start the windowing process, defaults to 15
    :type starting_x: int, optional
    :param step_size: by how much sliding window moves to the right and/or bottom, defaults to 10
    :type step_size: int, optional
    :param trim_if_small: by how many pixels to trim metadata's height or width if they are smaller than h_window or w_window,defaults to 11
    :type trim_if_small: int, optional
    :return: trimmed metadata i.e.  trimmed UTF-8 2D array of values containing metadata (window with highest mean area)
    :rtype: class: `numpy.ndarray`
    '''
    def sliding_window(image,starting_y,starting_x,h_window,w_window,step_size):
        '''Sliding window generator object'''
        h_img,w_img = image.shape
        for y in range(starting_y, h_img- h_window,step_size):
            for x in range(starting_x, w_img- w_window, step_size):
                yield y,x,image[y:y+h_window, x:x +w_window ]
          
    h_raw,w_raw = opened_meta.shape
    
    if h_window + step_size  >= h_raw:
        h_window = h_raw -trim_if_small
    if w_window + step_size>= w_raw:
        w_window = w_raw -trim_if_small
    
    s = sliding_window(connected_meta,starting_y,starting_x,h_window,w_window,step_size)

    max_window = connected_meta[starting_y:h_window+starting_y,
                 starting_x:w_window+starting_x ]
    max_mean = np.mean(max_window)
    y_max= starting_y
    x_max = starting_x
    for y,x,window in s:
        tmp = window
        mean = np.mean(tmp)
        if mean > max_mean:
            max_window = tmp
            max_mean  = mean
            y_max= y
            x_max =x

    trimmed_metadata =  opened_meta[y_max:y_max+h_window,x_max:x_max+w_window]

    return trimmed_metadata



def trimming_metadata(raw_metadata,type_metadata,
                     median_kernel_size=5,
                     opening_kernel_size = (3,3)):
    """Trim the rectangle containing the metadata to the smallest workable area
    
    :param raw_metadata: UTF-8 grayscale 2D array of values ranging from [0,255] representing rectangular metadata part of raw scanned image
    :type raw_metadata: class: `numpy.ndarray`
    :param type_metadata: where the detected metadata is located compared to the ionogram i.e. 'left', 'right', 'top', 'bottom'
    :type type_metadata: str
    :param median_kernel_size: size of the filter for median filtering morphological operation, defaults to 5
    :type median_kernel_size: int, optional
    :param opening_kernel_size: size of the filter for opening morphological operation, defaults to (3,3)
    :type opening_kernel_size: (int,int), optional
    :return: trimmed metadata i.e.  trimmed UTF-8 2D array of values  containing metadata 
    :rtype: class: `numpy.ndarray`
    :raises Exception: returns np.nan if there is an error
        
    """
    try:

        # Median filtering to remove salt and pepper noise
        median_filtered_meta = cv2.medianBlur(raw_metadata,median_kernel_size)
        
        # Opening operation: Erosion + Dilation
        kernel_opening = np.ones(opening_kernel_size,np.uint8)
        opened_meta = cv2.morphologyEx(median_filtered_meta,cv2.MORPH_OPEN,kernel_opening)
        
        # Binarizatinon for connected component algorithm
        _,meta_binary = cv2.threshold(opened_meta, 127,255,cv2.THRESH_BINARY)    
        
        # Run connected component algorithm
        connected_meta = connected_components_metadata_location(meta_binary)
        
        if type_metadata == 'left':
            trimmed_metadata = leftside_metadata_trimming(connected_meta,meta_binary)
        else:
            trimmed_metadata =  bottomside_metadata_trimming(connected_meta,opened_meta)
    
    
        return trimmed_metadata
    except:
        return np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC ### extract metadata from scan

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Code to extract metadata part of a raw scanned image
"""

# Library imports
import numpy as np

def extract_metadata(raw_img_array, limits_ionogram):
    """Extract metadata part of a raw scanned image and return coordinates delimiting its limits
    
    :param raw_img_array: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned simage
    :type raw_img_array: class: `numpy.ndarray`
    :param limits: limits delimiting the ionogram part of the raw scanned image i.e. [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit]
    :type limits: list
    :return: (type_metadata,raw_metadata) i.e. (location of metadata where {0:'left',1:'right',2:'top', 3:'bottom'} in the image,  UTF-8 grayscale 2D array of values ranging from [0,255] representing metadata part of raw scanned image)
    :rtype: (int, class: `numpy.ndarray`)
    """
    
    # Limits delimiting the ionogram part of the image
    x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit = limits_ionogram
    
    
    # Extract each rectangular block besides the ionogram
    rect_left = raw_img_array[:,0:x_axis_left_limit]
    rect_right = raw_img_array[:,x_axis_right_limit::]
    rect_top = raw_img_array[0:y_axis_upper_limit ,:]
    rect_bottom = raw_img_array[y_axis_lower_limit:: ,:]
    
    # Assumption: The location of the metadata will correspond to the rectangle with the highest area
    rect_list = [rect_left,rect_right,rect_top,rect_bottom ]
    rect_areas = [rect.shape[0] *rect.shape[1] for rect in rect_list]
    dict_mapping_meta = {0:'left',1:'right',
                         2:'top', 3:'bottom'}
    type_metadata_idx = np.argmax(rect_areas) 
    raw_metadata = rect_list[type_metadata_idx]
    type_metadata = dict_mapping_meta[type_metadata_idx ]
    
    return (type_metadata,raw_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### segment image in subdir

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ionogram grid detection

# COMMAND ----------

# MAGIC %md
# MAGIC ### grid mapping

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Code to determine the grid from all the ionograms in a folder 
using adjusted peak detection (From the weighed sum of all the image plots in a subsubdirectory,
the indices corresponding to the grid values are determined)
"""
# Library imports
import numpy as np
import pandas as pd
import scipy.signal as signal


#Determined from determine_default_grid_values
HZ = [1.5,2.0,2.5,3.5,4.5,5.5,6.5,7.0,7.5,8.5,9.5,10.5,11.5]
DEFAULT_HZ_COORD = [149,273,335,390,496,604,711,837,898,964,1128,1314,1444]
MEAN_HZ = [0.5*(num + DEFAULT_HZ_COORD[i+1])for i, num in enumerate(DEFAULT_HZ_COORD[:-1])]
UPPER_LIMIT_HZ_COORD =[89] + MEAN_HZ 
LOWER_LIMIT_HZ_COORD = MEAN_HZ + [1510]

KM_DEFAULT_100 = 55
KM_DEFAULT_200 = 110

def all_stack(df_img):
    """Returns the equally weighed sum of all the correctly extracted ionogram plot areas in a subsubdirectory 
    
    :param df_img: Dataframe contaning all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param cutoff_width: the width of an ionogram should be within cutoff_width of the median width of all the ionogram in a subdirectory (should be the same as the one used in scan2data.image_segmentation.segment_images_in_subdir.segment_images)
    :type cutoff_width: int
    :param cutoff_height: the height of an ionogram should be within cutoff_height of the median height of all the ionogram in a subdirectory (should be the same as the one used in scan2data.image_segmentation.segment_images_in_subdir.segment_images)
    :type cutoff_height: int
    :returns: weighed_sum i.e. equally weighed sum of all the extracted ionogram plot areas in a subsubdirectory
    :rtype: class: `numpy.ndarray`
    """
    
    # Pad the image if needed
    max_h = df_img["height"].max()
    max_w =  df_img["width"].max()
    median_h = int(np.median(df_img["height"]))
    median_w = int(np.median(df_img["width"]))
    

    df_img["padded"] = df_img["ionogram"].apply(lambda img: np.pad( img, ((0,max_h-img.shape[0]),(0,max_w-img.shape[1])),mode="constant",constant_values=1))
    
    #Weighed sum of the ionograms in a subdirectory
    weight = len(df_img.index)
    weighed_sum = weight * np.sum((df_img["padded"]).tolist(), axis = 0)
    

    return weighed_sum[0:-(max_h-median_h),0:-(max_w-median_w)]


def indices_highest_peaks(img, row_or_col,
                      peak_prominence_threshold=0.1,distance_between_peaks=5):
    """Determines and returns the indices of peak median values from the rows or column of an image
    
    :param img: grayscale image in the form of an 2D uint8 array
    :type img: class: `numpy.ndarray`
    :param row_or_col: 0 for colum or 1 for row
    :type row_or_col: int
    :param peak_prominence_threshold: the threshold to detect peaks that correspond to the grid lines, defaults to 0.1
    :type peak_prominence_threshold: int, optional
    :param distance_between_peaks: the minimum number of samples between subsequent peaks, defaults to 5
    :type distance_between_peaks: int, optional
    :returns: select_peaks i.e. array of the indices of peak median values from the rows or column of an image
    :rtype: class: `numpy.ndarray`
    
    
    """
    # Median values along each row or column
    median_values = np.median(img,row_or_col)
    
    # Normalize median values so they are between 0 and 1
    median_values_normalized = (median_values - np.min(median_values))/(np.max(median_values)-np.min(median_values))
    
    # Prepare peaks for peak detection function: the peaks should be pointing upwards
    peaks_function = 1 + -1*median_values_normalized 
    
    # Detect all peaks
    select_peaks,_ = signal.find_peaks(peaks_function,distance=distance_between_peaks,prominence = peak_prominence_threshold)
    
    #Remove edges from peaks
    h,w = img.shape
    if row_or_col == 0:
        select_peaks = select_peaks[:w]
    else:
        select_peaks = select_peaks[:h]

    return select_peaks
    
def adjust_arr_peaks(weighed_sum,arr_peaks,desired_length,row_or_col,
                  distance_between_peaks=30,peak_prominence_threshold=0.1,n_tries=1000,update_amount=0.01):
    """Adjust an array of peaks to the desired length and returns it. 
    
     :param weighed_sum: equally weighed sum of all the image plot areas in a subsubdirectory 
     :type weighed_sum: class: `numpy.ndarray`
     :param arr_peaks: array of peaks to adjust to the desired length
     :type arr_peaks: class: `numpy.ndarray`
     :param desired_length: number of elements desired in array
     :type desired_length: int
     :param row_or_col: 0 for colum or 1 for row
     :type row_or_col: int
     :param distance_between_peaks: the minimum number of samples between subsequent peaks, defaults to 30
     :type distance_between_peaks: int, optional
     :param peak_prominence_threshold: the threshold to detect peaks that correspond to the grid lines, defaults to 0.1
     :type peak_prominence_threshold: int, optional
     :param n_tries: the number of maximum tries to adjust arr, defaults to 1000
     :type n_tries: int, optional
     :param update_amount: by how much peak_prominence_threshold is updated for each iteration, defaults to 0.01
     :type update_amount: int, optional
     :returns: select_peaks i.e. adjusted array of the indices of peak median values from the rows or column of an image
     :rtype: class: `numpy.ndarray`
     ..note:: To prevent infinite loops, the script only runs for a maximum of n_tries times
    """
    
    
    arr_peaks = indices_highest_peaks(weighed_sum, row_or_col,peak_prominence_threshold,distance_between_peaks)
    
    # Adjust if lenght is not the desired length by re-running indices_highest_peaks with different parameters
    while len(arr_peaks) != desired_length and n_tries !=0:
        
        if len(arr_peaks) > desired_length:
            # increase peak_prominence_threshold 
            peak_prominence_threshold = peak_prominence_threshold + update_amount
            arr_peaks = indices_highest_peaks(weighed_sum, row_or_col,
                                               peak_prominence_threshold,distance_between_peaks)
        else:
            # decrease peak_prominence_threshold
            peak_prominence_threshold = peak_prominence_threshold - update_amount
            arr_peaks = indices_highest_peaks(weighed_sum, row_or_col,
                                               peak_prominence_threshold,distance_between_peaks)
        n_tries = n_tries - 1
    

    return arr_peaks


def get_grid_mappings(weighed_sum,
                      use_defaults=True,min_index_row_peaks =40): 
    """Determines and returns the the mapping between coordinate values and frequency/depth values in a subdirectory
    
    :param weighed_sum: equally weighed sum of all the image plot areas in a subsubdirectory 
    :type weighed_sum: class: `numpy.ndarray`
    :param use_defaults: use default values , defaults to True
    :type use_defaults: bool
    :param min_index_row_peaks: starting index to consider for peaks to determine km lines,defaults to 40
    :type min_index_row_peaks: int, optional
    :returns:  col_peaks,row_peaks,mapping_Hz, mapping_km i.e. one-dimmensional array of detected peaks of ionogram by column, one-dimmensional array of detected  peaks of by row, dictionary mapping of depth (km) to y coordinates  , dictionary mapping of frequency (Hz) to x coordinates
    :rtype: class: `numpy.ndarray`,class: `numpy.ndarray`,class: `dict`, class: `dict`
    :raises Exception: returns np.nan,np.nan,np.nan,np.nan
    """
    # Detect peaks
    col_peaks = indices_highest_peaks(weighed_sum, 0)
    row_peaks = indices_highest_peaks(weighed_sum, 1)
    
    # Map col_peaks to Hz values
    if len(col_peaks) == len(HZ):
        mapping_Hz = dict(zip(HZ,col_peaks)) 
    
    # Map adjusted col_peaks to Hz values
    else:
        try: 
            col_peaks =adjust_arr_peaks(weighed_sum,col_peaks,len(HZ),0)
            mapping_Hz = dict(zip(HZ,col_peaks)) 
            
            # Map adjusted HZ values to default coordinates if need be
            if use_defaults:
                for i,key in enumerate(HZ):
                    if mapping_Hz[key] > UPPER_LIMIT_HZ_COORD[i] or mapping_Hz[key] < LOWER_LIMIT_HZ_COORD[i]:
                        mapping_Hz[key] = DEFAULT_HZ_COORD[i]
        except:
            if use_defaults:
                 mapping_Hz = dict(zip(HZ,DEFAULT_HZ_COORD )) 
            else:    
                return np.nan,np.nan,np.nan,np.nan
    
    
    row_peaks = row_peaks[row_peaks > min_index_row_peaks]
    
    try:
        row_100 = row_peaks[0] #Should be around 30 for 100 km
        row_200 = row_peaks[1] #Should be around 30 for 100 km
    except:
        if use_defaults:
            row_100 = KM_DEFAULT_100
            row_200 = KM_DEFAULT_200  
        else:    
            return np.nan,np.nan,np.nan,np.nan
    
    if use_defaults:
        if abs(row_100 - KM_DEFAULT_100) > abs(KM_DEFAULT_200 - KM_DEFAULT_100):
            row_100 = KM_DEFAULT_100
        if abs(row_200 - KM_DEFAULT_200) > abs(KM_DEFAULT_200 - KM_DEFAULT_100):
            row_200 = KM_DEFAULT_200   
    
    mapping_km = {100:row_100,200:row_200}

    return col_peaks,row_peaks,mapping_Hz, mapping_km

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ionogram content extraction

# COMMAND ----------

# MAGIC %md
# MAGIC ### extract select parameters

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Code to extract select parameters from extracted ionogram trace
"""

# Library imports
import sys
import numpy as np

sys.path.append('../')

def extract_fmin_and_max_depth(arr_adjusted_coord,min_depth=50,if_raw=False):
    """Extract the minimum detected frequency value and maximum detected depth
    
    :param arr_raw_coord:  one-dimmensional array of values of all the pixels corresponding to the ionogram trace
    :type arr_raw_coord: class: `numpy.ndarray`
    :param min_depth: minimum depth in km to be considered, defaults to 30
    :type min_depth: int, optional
    :param if_raw: if (x,y) rather than (Hz,km) coordinates are used, defaults to False
    :type if_raw: bool, optional
    :returns: fmin, depth_max i.e. minimum frequency detected and maximum depth detected
    :rtype: float, float
    :raises Exception: returns np.nan, np.nan
    
    """
    try:
        
        adjusted_x, adjusted_y = zip(*arr_adjusted_coord)
        adjusted_x = np.array(adjusted_x)
        adjusted_y = np.array(adjusted_y)
        
        if if_raw:
            min_depth= int(min_depth * KM_DEFAULT_100/100)
        
        thresholded = adjusted_y > min_depth
        adjusted_x_thresholded = adjusted_x[thresholded]
        adjusted_y_thresholded = adjusted_y[thresholded]

        fmin = min(adjusted_x_thresholded)
        depth_max =  max(adjusted_y_thresholded)
        return fmin, depth_max
    except:
        return np.nan, np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC ### extract all coordinates ionogram trace

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Code to extract coordinates of ionogram trace (black part) 
"""

#Library imports
import sys

import numpy as np

def extract_ionogram_windows(binary_iono,
                             stepSize=25, windowSize=(100,100)):
    """Clean ionogram by using small thresholding windows
    
    :param binary_iono: two-dimmensional uint8 array representing ionogram where the extracted threshold values are in while (1s) while the rest is in black (0s)
    :type binary_iono: class: `numpy.ndarray`
    :param stepSize: By how much window moves to the right and/or bottom
    :type stepSize: int
    :param windowSize: (height, width) of moving window
    :type windowSize: tuple
    :returns new_iono: cleaned ionogram represented by  two-dimmensional uint8 array
    :rtype:  class: `numpy.ndarray`
    """
    # TODO: impove thresholding for windowing
    threshold  = np.mean(binary_iono) *2
   
    h_iono,w_iono = binary_iono.shape
    h_window,w_window = windowSize
    new_iono = np.zeros((h_iono,w_iono) )
    
    for y in range(0, h_iono - h_window, stepSize):
        for x in range(0, w_iono - w_window, stepSize):
            box = binary_iono[y:y+h_window, x:x +w_window ]
            if np.mean(box) > threshold:
                new_iono[y:y+h_window, x:x +w_window ] = box       
    
    return new_iono


def background_substraction(raw_iono):
    """Use Gaussian Mixture-based Background/Foreground Segmentation Algorithm to clean the raw ionogram
    
    :param iono: two-dimmensional uint8 array representing raw ionogram
    :type iono: class: `numpy.ndarray`
    :returns: two-dimmensional uint8 array representing cleaned ionogram
    :rtype class: `numpy.ndarray`
    
    """

    background_substracter = cv2.createBackgroundSubtractorMOG2()
    masked_iono = background_substracter.apply(raw_iono)
    
    return 255-masked_iono

def extract_coord(iono,col_peaks,row_peaks,
                  threshold = 200,kernel_size_blurring = 5):
    """Extract (x,y) of all the pixels corresponding to the ionogram trace
    
    :param iono: two-dimmensional uint8 array representing raw ionogram
    :type iono: class: `numpy.ndarray`
    :param col_peaks: one-dimmensional array of detected peaks of ionogram by column
    :type col_peaks: class: `numpy.ndarray`
    :param row_peak: one-dimmensional array of detected  peaks of by row
    :type row_peaks: class: `numpy.ndarray`
    :param threshold: threshold of inverted pixel value to be considered ionogram data, defaults to 200
    :type threshold: int, optional
    :param kernel_size_blurring: kernel size for median filtering operation, defaults to 5
    :type kernel_size_blurring: int, optional
    :returns: arr_raw_coord0 ,arr_raw_coord: one-dimmensional array of (x,y) coordinates of all the pixels corresponding to the ionogram trace
    :rtype: class: `numpy.ndarray`,class: `numpy.ndarray`
    :raises Exception: returns np.nan,np.nan if there is an error
    
    """

    # Shape of image
    try:
        h,w = iono.shape

        # Median blurring to remove salt and pepper noise
        median_filtered_iono = cv2.medianBlur(iono, kernel_size_blurring)

        # Invert image
        inverted_iono = 255 -median_filtered_iono

        # Correct image for grid ie remove the grid
        grid = np.ones((h,w),np.uint8)
        for i in col_peaks:
            cv2.line(grid , (i, 0), (i,h), 0, 5, 1)
        for i in row_peaks:
            cv2.line(grid , (0, i), (w,i), 0, 5, 1)
        corrected_iono = np.multiply(grid,inverted_iono)

        # Assuming trace is going to be black ie mostly values close to 0 in the array
        # Thus, the inverted trace is going to be white ie values most close to 252
        # Threshold the image
        _,thresholded_iono = cv2.threshold(corrected_iono, threshold, 1, cv2.THRESH_BINARY)

        # Corrected ionogram by windowing operations
        windowed = extract_ionogram_windows(thresholded_iono)

        # y and x coordinates
        arr_y, arr_x = np.where(thresholded_iono  == 1)
        arr_raw_coord0 = np.array(list(zip(arr_x, arr_y)), dtype=np.float64)

        arr_y, arr_x = np.where(windowed  == 1)
        arr_raw_coord = np.array(list(zip(arr_x, arr_y)), dtype=np.float64)

        return arr_raw_coord0,arr_raw_coord  # raw_coord, windowed_coord    `
    except:
        return np.nan,np.nan


def map_coordinates_positions_to_values(arr_raw_coord,col_peaks,row_peaks,mapping_Hz,mapping_km):
    """Map (x,y) position coordinates of ionogram pixels to (Hz,km) values
    
    :param arr_raw_coord:  one-dimmensional array of (x,y) coordinates of all the pixels corresponding to the ionogram trace
    :type arr_raw_coord: class: `numpy.ndarray
    :param col_peaks: one-dimmensional array of detected peaks of ionogram by column
    :type col_peaks: class: `numpy.ndarray`
    :param row_peak: one-dimmensional array of detected  peaks of by row
    :type row_peaks: class: `numpy.ndarray`
    :param mapping_Hz: dictionary mapping of frequency (Hz) to x coordinates
    :type mapping_Hz: class: `dict`
    :param mapping_km:  dictionary mapping of depth (km) to y coordinates 
    :type mapping_km: class: `dict`
    :returns: arr_adjusted_coord: one-dimmensional array of (Hz,km) values of all the pixels corresponding to the ionogram trace
    :rtype: class: `numpy.ndarray`
    """

    # check if there are any coordinate values recorded for the ionogram
    if not arr_raw_coord.size:
        return arr_raw_coord

    # remove outliers ie coordinates less coordinates corresponding to 0.5 Hz or more than corresponding to 11.5 Hz
    col_peaks = np.array(list(mapping_Hz.values())) # use the modified col_peaks ie the one with exactly 13 values

    mask = np.logical_or(arr_raw_coord[:,0] < col_peaks.min(), arr_raw_coord[:,0] > col_peaks.max())
    arr_raw_coord = arr_raw_coord[~mask,:]

    # map (y,x) to (km, Hz)
    km_values, index_values_km = list(zip(*list(mapping_km.items())))
    multiplier = (km_values[1] - km_values[0])/(index_values_km[1] -index_values_km[0] )
    
    arr_adjusted_coord = arr_raw_coord.copy()
    arr_adjusted_coord[:,1] = km_values[0]+(arr_adjusted_coord[:,1]- index_values_km[0])*multiplier

    #reverse mapping_km mappings
    mapping_Hz_reversed = {mapping_Hz[freq_key]:freq_key for freq_key in mapping_Hz}
    arr_adjusted_x = np.array([])
    for coord_x in arr_adjusted_coord[:,0]:
        if coord_x in col_peaks:
            new_coord_x = mapping_Hz_reversed[coord_x ]
        else:
         # find the 2 closest values and linearly interpolate from there
            leftmost_val = col_peaks[col_peaks < coord_x].max()
            rightmost_val = col_peaks[col_peaks > coord_x].min()
            multiplier = (mapping_Hz_reversed[rightmost_val]- mapping_Hz_reversed[leftmost_val])/(rightmost_val - leftmost_val)
            new_coord_x = mapping_Hz_reversed[leftmost_val] + multiplier*(coord_x - leftmost_val)

        arr_adjusted_x = np.append(arr_adjusted_x,new_coord_x)
    arr_adjusted_coord[:,0] = arr_adjusted_x
    
    return arr_adjusted_coord

def extract_coord_subdir_and_param(df_img,subdir_location,col_peaks,row_peaks,mapping_Hz,mapping_km):
    """Extract the raw, windowed coordinates in all the raw extracted ionograms from a subdirectory, map those coordinates into (Hz, km) and extract select parameterd
    
    :param df_img: Dataframe containing all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param subdir_location: Path of the subdir_location
    :type subdir_location: string
    :param col_peaks: one-dimmensional array of detected peaks of ionogram by column
    :type col_peaks: class: `numpy.ndarray`
    :param row_peak: one-dimmensional array of detected  peaks of by row
    :type row_peaks: class: `numpy.ndarray`
    :param mapping_Hz: dictionary mapping of frequency (Hz) to x coordinates
    :type mapping_Hz: class: `dict`
    :param mapping_km:  dictionary mapping of depth (km) to y coordinates 
    :type mapping_km: class: `dict`
    :returns: df_img, df_loss: i.e.  i.e. dataframe containing extracted ionogram trace coordinates from all the extracted raw ionograms in a directory,dataframe containing file names leading to runtime errors
    :rtype: class: `pandas.core.frame.DataFrame`, class: `pandas.core.frame.DataFrame`
    
    """
    #Get (x,y) coordinates of trace
    df_img['raw_coord'],df_img['window_coord'] = zip(*df_img['ionogram'].map(lambda iono: extract_coord(iono, col_peaks,row_peaks)))

    # Remove loss
    df_loss_coord,loss_coord = record_loss(df_img,'ionogram_content_extraction.extract_all_coordinates_ionogram_trace.extract_coord_subdir', subdir_location)
    df_img= df_img[~loss_coord]

    #df_img.to_csv("U:/alouette-scanned-ionograms-processing/df_img.csv", index=False)
    
    # (Hz, km) coordinates
    df_img['mapped_coord'] = df_img['window_coord'].map(lambda windowed: map_coordinates_positions_to_values(windowed, col_peaks, row_peaks, mapping_Hz, mapping_km))

    # Select parameters extracted
    df_img['fmin'],df_img['max_depth'] =  zip(*df_img['mapped_coord'].map(lambda mapped_coord: extract_fmin_and_max_depth(mapped_coord)))
    
    # Remove loss.
    df_loss_param,loss_param= record_loss(df_img,'ionogram_content_extraction.extract_select_parameters.extract_fmin_and_max_depth',subdir_location)
    df_img= df_img[~loss_param]
    
    df_loss = df_loss_coord.append(df_loss_param)
    
    return df_img, df_loss

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata translation

# COMMAND ----------

# MAGIC %md
# MAGIC ### leftside metadata grid mapping

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Determine the grid used to translate the dot and num metadata in a subdirectory
"""
# Library import
import numpy as np
from scipy.signal import find_peaks


# List of directories containing leftside metadata with dots
LIST_DIRECTORY_DOTS = ['R014207907F','R014207908F','R014207909F','R014207929F','R014207930F','R014207940F','R014207978F','R014207979F']

# Labelling of coordinates
LABELS_NUM = [1,2,4,8]
LABELS_CAT_DOT = ['day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2','second_1', 'second_2','station_code']
LABELS_CAT_DIGIT = ['satellite_number','year','day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2',
                    'second_1', 'second_2', 'station_number_1','station_number_2']
LABELS_DICT = ['dict_cat_dot','dict_num_dot','dict_cat_digit','dict_num_digit']

#Defaults for dictionary mappings of coordinates to labels
DEFAULT_DICT_CAT_DIGIT = (53,21,661) #mean_dist_default,first_peak_default,last_peak_default
DEFAULT_DICT_NUM_DIGIT = (47,41,20) #mean_dist_default,first_peak_default,dist_btw_peaks for peak detection

DEFAULT_DICT_CAT_DIGIT_F = (43,23,540) #mean_dist_default,first_peak_default,last_peak_default for those in LIST_DIRECTORY_DOTS 
DEFAULT_DICT_NUM_DIGIT_F = (40,37,20) #mean_dist_default,first_peak_default,dist_btw_peaks for peak detection for those in LIST_DIRECTORY_DOTS 

DEFAULT_DICT_CAT_DOT = (59,20,549)##mean_dist_default,first_peak_default,last_peak_default
DEFAULT_DICT_NUM_DOT = (15,32,10) #mean_dist_default,first_peak_default,dist_btw_peaks for peak detection

def extract_centroids_and_determine_type(dilated_meta,file_name,
                      min_num_pixels=50, max_number_pixels=1000,max_area_dot=120):
    
    '''Extract the coordinates of the centroid of each metadata dot/number using the connected component algorithm as well as determines if the metadata is of type dot or number
    
    :param dilated_meta: trimmed metadata (output of image_segmentation.leftside_metadata_trimming) after a rotation and dilation morphological transformation (see translate_leftside_metadata.extract_leftside_metadata )
    :type dilated_meta: class: `numpy.ndarray`
    :param file_name: full path of starting raw image ex:G:/R014207929F/431/Image0399.png
    :type file_name: str
    :param min_num_pixels: minimum number of pixels to be considered metadata dot/num, defaults to 50
    :type min_num_pixels: int, optional
    :param max_number_pixels: maximum number of pixels to be considered metadata dot/num, defaults to 1000
    :type max_number_pixels: int, optional
    :param max_area_dot: maximum median area of a single metadata components to be considered a dot, defaults to 120
    :type max_area_dot: int, optional
    :returns: col_centroids,row_centroids,is_dot : list of col ('x') positions where metadata is detected,list of row ('y') positions where metadata is detected,whether the metadata is dots
    :rtype: class: `list`,class: `list`, bool
    :raises Exception: returns np.nan,np.nan,np.nan if there is an error
    
    '''

    try:
  
        # Use connected component algorithm to determine the centroids
        _, _, stats, centroids	=	cv2.connectedComponentsWithStats(dilated_meta)
        area_centroids = stats[:,-1]
        
        # Remove centroids who are probably not associated with metadata num/dot
        centroids_metadata = centroids[np.logical_and(area_centroids > min_num_pixels, area_centroids < max_number_pixels),:]    
        col_centroids, row_centroids = zip(*centroids_metadata)
        
        # Rounding to nearest integer
        col_centroids = list(map(round,col_centroids))
        row_centroids = list(map(round,row_centroids))
        
        #Determine if dot leftside metadata 
        area_centroids =area_centroids[np.logical_and(area_centroids > min_num_pixels, area_centroids < max_number_pixels)]    
        median_area = np.median(area_centroids)
        is_dot = False
        if any([dir_dot in file_name for dir_dot in LIST_DIRECTORY_DOTS]) and median_area < max_area_dot:
            is_dot = True
        return col_centroids,row_centroids,is_dot
    except:
        return np.nan,np.nan,np.nan


def indices_highest_peaks_hist_binning(list_coord,
                   nbins=500,peak_prominence_threshold=0.2,distance_between_peaks=30):
    
    """Determines and returns the indices of the most common values in a list of coordinates using binning
    
    :param list_coord: list of positions where metadata is detected
    :type list_coord: class: `list`
    :param nbins: number of bins used for binning operation, defaults to 500
    :type nbins: int, optional
    :param peak_prominence_threshold: the threshold to detect peaks that correspond to the peaks corresponding to the most common values, defaults to 0.2
    :type peak_prominence_threshold: int, optional
    :param distance_between_peaks: the minimum number of samples between subsequent peaks corresponding to the most common values, defaults to 30
    :type distance_between_peaks: int, optional
    :returns: select_peaks,bin_edges,counts i.e. array of the indices of  peaks corresponding to the most common values, array for the bin edges after calling np.histogram, array for counts of the number of elements in each bin after calling np.histogram  
    :rtype: class: `numpy.ndarray`,class: `numpy.ndarray`,class: `numpy.ndarray`
    
    
    """
    # Transform to numpy.array
    arr_coord = np.array(list_coord )
    
    # Remove outliers
    mean_arr = np.mean(arr_coord)
    std_arr = np.std(arr_coord)
    arr_coord_no_outlier= arr_coord[np.abs(arr_coord - mean_arr) < 3 * std_arr]
    
    # Binning
    counts,bin_edges = np.histogram(arr_coord_no_outlier,bins=nbins)
    
    # Detect all peaks
    counts_norm = (counts - np.min(counts))/(np.max(counts)- np.min(counts)) #normalization
    select_peaks_idx,_ = find_peaks(counts_norm,distance = distance_between_peaks,prominence = peak_prominence_threshold)
    
    
    return select_peaks_idx,bin_edges,counts

# Check y_peaks >0
# TODO: DEFAULTS
def get_leftside_metadata_grid_mapping(list_x_dot,list_y_dot,list_x_digit,list_y_digit,dir_name,
                      difference_ratio=0.75,use_defaults=True):
    
    """Determines and returns the the mapping between coordinate values on a metadata image and metadata labels in a subdirectory, for metadata of types dot and digits, as well as returns the histogram used to generate each mapping
    
    :param list_x_dot: list of col ('x') positions where metadata of type dot is detected
    :type list_x_dot: class: `list`
    :param list_y_dot: list of row ('y') positions where metadata of type dot is detected
    :type list_y_dot: class: `list`
    :param list_x_digit: list of col ('x') positions where metadata of type digit is detected
    :type list_x_digit: class: `list`
    :param list_y_digit: list of row ('y') positions where metadata of type digit is detected
    :type list_y_digit: class: `list`
    :param dir_name: name of directory
    :type dir_name: string
    :param difference_ratio: ratio defining when to use default values, defaults to 0.5
    :type difference_ratio: int, optional
    :param use_defaults: whether to use default values, defaults to True
    :type use_defaults: bool, optional
    :returns: all_dict_mapping,all_dict_hist: dictionary of dictionaries where each dictionary correspond to a mapping between coordinates on the image and metadata labels, dictionary of histograms used to generated each dictionary in all_dict_mapping
    :rtype: dict, dict
    """
    # Dictionary of dictionaries that map labels to coordinate point in metadata
    all_labels = [LABELS_CAT_DOT,LABELS_NUM ,LABELS_CAT_DIGIT,LABELS_NUM]
    all_dict_mapping = {}
    all_dict_hist = {}
    # Different protocols depending on the type of dictionary mappings
    for i, list_coord in enumerate([list_x_dot,list_y_dot,list_x_digit,list_y_digit]):
        type_dict = LABELS_DICT[i]
        labels = all_labels[i]
        try:
            if 'cat' in type_dict:
                if type_dict == 'dict_cat_digit':
                    if any([dir_dot in dir_name for dir_dot in LIST_DIRECTORY_DOTS]):
                        mean_dist_default,first_peak_default,last_peak_default=DEFAULT_DICT_CAT_DIGIT_F
                    else:
                        mean_dist_default,first_peak_default,last_peak_default=DEFAULT_DICT_CAT_DIGIT
            
                elif type_dict == 'dict_cat_dot':
                    mean_dist_default,first_peak_default,last_peak_default=DEFAULT_DICT_CAT_DOT
                try:
                    idx_peaks,bin_edges,counts = indices_highest_peaks_hist_binning(list_coord)
                    peaks = bin_edges[np.array(idx_peaks)] #coordinate values on a metadata image probably corresponding to metadata
                    
                    n_labels = len(labels)
                    first_peak = peaks[0]
                    last_peak = peaks[-1]

                    if use_defaults and abs(last_peak -last_peak_default)  > difference_ratio*mean_dist_default:
                        last_peak = last_peak_default
                    if use_defaults and abs(first_peak -first_peak_default)  > difference_ratio*mean_dist_default:
                        first_peak = first_peak_default
                        
                    mean_dist_btw_peaks = (last_peak - first_peak)/(n_labels -1)
                    list_peaks = [int(round(first_peak + i* mean_dist_btw_peaks)) for i in range(0,n_labels )]
                    
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] = (idx_peaks,bin_edges,counts)
                

                except:
                    last_peak = last_peak_default
                    first_peak = first_peak_default
                    mean_dist_btw_peaks = mean_dist_default
                    list_peaks = [int(round(first_peak + i* mean_dist_btw_peaks)) for i in range(0,n_labels )]
                    
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] = {}
                
            elif 'num' in type_dict:
                if  type_dict == 'dict_num_digit':
                    if any([dir_dot in dir_name for dir_dot in LIST_DIRECTORY_DOTS]):
                        mean_dist_default,peak_0_default,dist_btw_peaks = DEFAULT_DICT_NUM_DIGIT_F
                    else:
                        mean_dist_default,peak_0_default,dist_btw_peaks = DEFAULT_DICT_NUM_DIGIT
                elif type_dict == 'dict_num_dot':
                    mean_dist_default,peak_0_default,dist_btw_peaks= DEFAULT_DICT_NUM_DOT

                    
                try:
                    idx_peaks,bin_edges,counts = indices_highest_peaks_hist_binning(list_coord,peak_prominence_threshold=0.3,nbins=100,distance_between_peaks=dist_btw_peaks)
                
                    peaks = bin_edges[np.array(idx_peaks)]                
                    peak_0 = peaks[0]
                    if use_defaults and abs(peak_0 -peak_0_default)  > difference_ratio*mean_dist_default:
                        peak_0 = peak_0_default
                
                    # only first three peaks are deemed relevant
                    if len(peaks) < 3:
                        max_idx = 2
                    else:
                        max_idx = 3
                
                    mean_dist_btw_peaks = np.mean([peaks[i+1]-peaks[i] for i in range(0,max_idx)])
                    if use_defaults and abs(mean_dist_btw_peaks - mean_dist_default)  > difference_ratio*dist_btw_peaks:
                        mean_dist_btw_peaks = mean_dist_default
                    list_peaks = [int(round(peak_0 + i* mean_dist_btw_peaks)) for i in range(0,len(labels))]
                
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] = (idx_peaks,bin_edges,counts)
                except:
                    peak_0 = peak_0_default
                    mean_dist_btw_peaks = mean_dist_default
                    list_peaks = [int(round(peak_0 + i* mean_dist_btw_peaks)) for i in range(0,len(labels))]
                    all_dict_mapping[type_dict] =dict(zip(list_peaks,labels))
                    all_dict_hist[type_dict] =  {}
        except:
            all_dict_mapping[type_dict] ={}
            all_dict_hist[type_dict] =  {}

            
    return all_dict_mapping,all_dict_hist

# COMMAND ----------

# MAGIC %md
# MAGIC ### translate leftside metadata

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Translate metadata (dot or numbers) located on the left of ionograms
"""

# Library imports
from itertools import chain
import sys

import pandas as pd
import numpy as np

LABELS_DICT = ['dict_cat_dot','dict_num_dot','dict_cat_digit','dict_num_digit']

def map_coord_to_metadata(list_cat_coord,list_num_coord,dict_mapping_cat, dict_mapping_num):
    """Map coordinate of metadata centroids to information
    
    :param list_cat_coord: list of metadata positions to map to categories   
    :type list_cat_coord: list
    :param list_num_coord: list of metadata positions to map to numbers 
    :type list_num_coord: list
    :param dict_mapping_cat: dictionary used to map coordinate positions to categories
    :type dict_mapping_cat: dict
    :param dict_mapping_num: dictionary used to map coordinate positions to numbers
    :type dict_mapping_num: dict
    :returns: dict_metadata
    :rtype: dict
    
    """
    
    try:
        
        list_coord = zip(list_cat_coord,list_num_coord)
        coord_mapping_cat = dict_mapping_cat.keys()
        coord_mapping_num = dict_mapping_num.keys()
        
        dict_metadata={}
        for cat_coord, num_coord in list_coord:
            cat_key = min(coord_mapping_cat, key=lambda x:abs(x-cat_coord))
            num_key = min(coord_mapping_num, key=lambda x:abs(x-num_coord))
            
            cat = dict_mapping_cat[cat_key]
            num = dict_mapping_num[num_key]
            
            # TODO: improve for many num
            if cat in dict_metadata:
                dict_metadata[cat].append(num)
            else:
                dict_metadata[cat] = [num]
        
        return dict_metadata
    except:
        return np.nan
    

        
def get_leftside_metadata(df_img,subdir_location,
                                dilation_kernel_size = (1,1)):
    """Reads metadata located on the left of ionograms, whether they are of type dot or type num
    
    :param df_img: Dataframe containing all the correctly extracted ionogram plot areas in a subsubdirectory (output of image_segmentation.segment_images_in_subdir.segment_images)
    :type df_img: class: `pandas.core.frame.DataFrame`
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param dilation_kernel_size: size of the filter for dilation morphological operation, defaults to (1,1)
    :type dilation_kernel_size: tuple, optional
    :returns: df_img, df_loss,dict_mapping, dict_hist i.e. df_img containing additional information including the translated metadata, dataframe containing file names leading to runtime errors,dictionary of dictionaries where each dictionary correspond to a mapping between coordinates on the image and metadata labels, dictionary of histograms used to generated each dictionary in all_dict_mapping
    :rtype: class: `pandas.core.frame.DataFrame`, class: `pandas.core.frame.DataFrame`, dict, dict
    """

    # Centroids extraction
    df_img['rotated_metadata'] = df_img['trimmed_metadata'].map(lambda trimmed_meta: np.rot90(trimmed_meta,-1))
    kernel_dilation = np.ones(dilation_kernel_size,np.uint8)
    df_img['dilated_metadata'] = df_img['rotated_metadata'].map(lambda rotated_meta: cv2.dilate(rotated_meta,kernel_dilation))
    df_img['x_centroids'],df_img['y_centroids'],df_img['is_dot'] = zip(*df_img.apply(lambda row: extract_centroids_and_determine_type(row['dilated_metadata'],row['file_name']),1))
    df_loss_centroids_extraction,loss_centroids_extraction = record_loss(df_img,'metadata_translation.determine_leftside_metadata_grid_mapping.extract_centroids_and_determine_type',subdir_location)
      
    # Remove files whose centroid metadata extraction was not successful
    df_img = df_img[~loss_centroids_extraction]
    
    # Determine metadata mapping for dot-type metadata and num-type metadata
    df_dot_subset = df_img[np.array(df_img['is_dot'])]
    df_num_subset = df_img[np.invert(np.array(df_img['is_dot']))]

    list_x_dot, list_y_dot,list_x_digit,list_y_digit = [0],[0],[0],[0]
    if not df_dot_subset.empty:
        list_x_dot = list(chain(*df_dot_subset['x_centroids'].tolist()))
        list_y_dot = list(chain(*df_dot_subset['y_centroids'].tolist()))
    
    if not df_num_subset.empty:
        list_x_digit = list(chain(*df_num_subset['x_centroids'].tolist()))
        list_y_digit = list(chain(*df_num_subset['y_centroids'].tolist()))
    dict_mapping,dict_hist = get_leftside_metadata_grid_mapping(list_x_dot,list_y_dot,list_x_digit,list_y_digit,subdir_location)

    # Determine the value of metadata based on the mappings
    df_img['dict_metadata'] = df_img.apply(lambda row: 
        map_coord_to_metadata(row['x_centroids'],row['y_centroids'],dict_mapping['dict_cat_dot'], dict_mapping['dict_num_dot']) if row['is_dot'] 
        else map_coord_to_metadata(row['x_centroids'],row['y_centroids'],dict_mapping['dict_cat_digit'], dict_mapping['dict_num_digit']),1)
    df_loss_mapping,loss_mapping = record_loss(df_img,'metadata_translation.translate_leftside_metadata.map_coord_to_metadata',subdir_location)
    df_img = df_img[~loss_mapping]
    
    df_loss = pd.concat([df_loss_centroids_extraction,df_loss_mapping])
    
    return df_img, df_loss,dict_mapping, dict_hist

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
For each of the raw images in a subdirectory i.e. R014207948/1743-9/, segment it into ionogram plot and trimmed metadata while handling errors and recording loss of data
"""

# Library imports
import glob
import sys
import cv2
import pandas as pd
import numpy as np

#List of subdirectories requiring geometric transformation (rotation,reflection)

LIST_FLIP_VERTICAL = ["R014207815/3508-A19",
                      "R014207954/2201-1A",
                      "R014207907F/289"]
#R014207909F/730: 0110-0205


LIST_ROTATE_180 = ["R014207962/1502-3A",
                   "R014207962/1505-1B",
                   "R014207965/1627-4A",
                   "R014207965/1647-4A",
                   "R01420796/1655-6A"]

def trimming_metadata(raw_metadata,type_metadata,
                     median_kernel_size=5,
                     opening_kernel_size = (3,3)):
    """Trim the rectangle containing the metadata to the smallest workable area
    
    :param raw_metadata: UTF-8 grayscale 2D array of values ranging from [0,255] representing rectangular metadata part of raw scanned image
    :type raw_metadata: class: `numpy.ndarray`
    :param type_metadata: where the detected metadata is located compared to the ionogram i.e. 'left', 'right', 'top', 'bottom'
    :type type_metadata: str
    :param median_kernel_size: size of the filter for median filtering morphological operation, defaults to 5
    :type median_kernel_size: int, optional
    :param opening_kernel_size: size of the filter for opening morphological operation, defaults to (3,3)
    :type opening_kernel_size: (int,int), optional
    :return: trimmed metadata i.e.  trimmed UTF-8 2D array of values  containing metadata 
    :rtype: class: `numpy.ndarray`
    :raises Exception: returns np.nan if there is an error
        
    """
    try:

        # Median filtering to remove salt and pepper noise
        median_filtered_meta = cv2.medianBlur(raw_metadata,median_kernel_size)
        
        # Opening operation: Erosion + Dilation
        kernel_opening = np.ones(opening_kernel_size,np.uint8)
        opened_meta = cv2.morphologyEx(median_filtered_meta,cv2.MORPH_OPEN,kernel_opening)
        
        # Binarizatinon for connected component algorithm
        _,meta_binary = cv2.threshold(opened_meta, 127,255,cv2.THRESH_BINARY)    
        
        # Run connected component algorithm
        connected_meta = connected_components_metadata_location(meta_binary)
        
        if type_metadata == 'left':
            trimmed_metadata = leftside_metadata_trimming(connected_meta,meta_binary)
        else:
            trimmed_metadata =  bottomside_metadata_trimming(connected_meta,opened_meta)
    
    
        return trimmed_metadata
    except:
        return np.nan
    
def record_loss(df,function_name,subdir_location,
                columns_to_extract=['file_name'],loss_extraction=[]):
    """Generate a dataframe that records loss due to a self-imposed filter or a runtime programming error
    
    :param df: dataframe containing information of which image files did not pass a self-imposed filter or lead to runtime programming errors
    :type df: class: `pandas.core.frame.DataFrame`
    :param function_name: function or self-imposed filter leading to a loss
    :type function_name: str
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param columns_to_extract: list of columns of df to extract, defaults to ['file_name']
    :type columns_to_extract: list, optional
    :param loss_extraction: whether a custom series is to be used to extract selected rows from the dataframe, defaults to []
    :type loss_extraction: class: `pandas.core.series.Series`, optional
    :returns: df_loss_extraction,loss_extraction i.e. dataframe containing file names leading to runtime errors or that do not pass pre-established filters (metadata size, ionogram size) as well as boolean series indicating which row of data to remove (==1)
    :rtype: (class: `pandas.core.frame.DataFrame`,class: `pandas.core.series.Series`)
    """   
    if len(loss_extraction) == 0:
        # function should return NA if there an error
        loss_extraction = df.isna().any(axis=1)
    # Record the files whose extraction was not successful
    df_loss_extraction = df[loss_extraction].copy()
    df_loss_extraction = df_loss_extraction[columns_to_extract]
    df_loss_extraction['func_name'] = function_name
    df_loss_extraction[ 'subdir_name'] = subdir_location
    
    return df_loss_extraction,loss_extraction

def limits_ionogram(raw_img, row_or_col,
                    starting_index_col=15):
    """Returns the upper and lower limits of the ionogram part of the scan by row or column using mean-based thresholding
    
    :param starting_img: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned image
    :type starting_img: class: `numpy.ndarray`
    :param row_or_col: 0 for column or 1 for row
    :type row_or_col: int
    :param starting_index_col: where the ionogram starting column should be after to protect against cuts, defaults to 15
    :type starting_index_col: int, optional
    :return:  limits[0],limits[-1] i.e. the upper and lower limits of the ionogram part of the scan by row (row_or_col=1) or column (row_or_col=0)
    :rtype: int,int
            
            
    """
    
    # Mean pixel values by by row/col
    mean_values = np.mean(raw_img, row_or_col)
    
    # Normalized mean values
    normalized_mean_values = (mean_values - np.min(mean_values))/np.max(mean_values)
    
    # Threshold is the overall mean value of the entire image
    threshold = np.mean(normalized_mean_values)
    
    if row_or_col == 0:
        #Protect against scans that includes cuts from another ionogram ex:R014207956\2394-1B\51.png 
        limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold and i > starting_index_col]
    else:
        limits = [i for i, mean in enumerate(normalized_mean_values) if mean >threshold]
        
    return limits[0],limits[-1]

def extract_ionogram(raw_img_array):
    """Extract ionogram part of a raw scanned image and return coordinates delimiting its limits
    
    :param raw_img_array: UTF-8 grayscale 2D array of values ranging from [0,255] representing raw scanned image
    :type raw_img_array: class: `numpy.ndarray`
    :return: (limits, ionogram) i.e. (list of coordinates delimiting the limits of the ionogram part of a raw scanned image formatted as [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit], UTF-8 grayscale 2D array of values ranging from [0,255] representing ionogram part of scanned image)
    :rtype: (list,numpy.ndarray)
    :raises Exception: returns [],np.nan if there is an error
        
    """
    try:

        # Extract coordinate delimiting the ionogram part of the scan
        x_axis_left_limit ,x_axis_right_limit = limits_ionogram(raw_img_array, 0)
        y_axis_upper_limit, y_axis_lower_limit = limits_ionogram(raw_img_array, 1)

        # Extract ionogram part
        ionogram = raw_img_array[y_axis_upper_limit:y_axis_lower_limit,x_axis_left_limit:x_axis_right_limit]
        limits = [x_axis_left_limit ,x_axis_right_limit, y_axis_upper_limit, y_axis_lower_limit]
        return (limits, ionogram)
    
    # In case if ionogram extraction fails:
    except: 
        return ([],np.nan)

def segment_images(subdir_location, regex_img,
                  cutoff_width = 300, cutoff_height=150,
                  min_leftside_meta_width = 50, min_bottomside_meta_height=25):
    """From all the raw images in a subsubdirectory, extract the ionogram and trimmed metadata while handling errors and recording loss of data
    
    :param subdir_location: full path of the subdir
    :type subdir_location: str
    :param regex_img: regular expression to extract image
    :type regex_img: str
    :param cutoff_width: the width of an ionogram should be within cutoff_width of the median width of all the ionogram in a subdirectory, defaults to 300
    :type cutoff_width: int, optional
    :param cutoff_height: the height of an ionogram should be within cutoff_height of the median height of all the ionogram in a subdirectory, defaults to 150
    :type cutoff_height: int, optional
    :param min_leftside_meta_width: the minimum width of trimmed metadata located on the left side of ionograms, defaults to 50
    :type min_leftside_meta_width: int, optional
    :param min_bottomside_meta_height: the minimum height of trimmed metadata located on the bottom side of ionograms, defaults to 25
    :type min_bottomside_meta_height: int, optional
    :return: df_img,df_loss,df_outlier i.e. dataframe containing extracted ionograms and trimmed metadata from all the images in a directory,dataframe containing file names leading to runtime errors, dataframe containing file names that do not pass pre-established filters (metadata size, ionogram size)
    :rtype: (class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`)
    .. todo:: complete flip_vertical ie list of subdirectories requiring flipping
    """
    # List of raw image files in subdirectory
    regex_raw_image= subdir_location+regex_img
    list_images = glob.glob(regex_raw_image)
    
    # If flipping/rotating is required for all the images in the subdirectory
    path = subdir_location.replace('\\', '/')
    flip_vertical = any([subdir_dir in path for subdir_dir in LIST_FLIP_VERTICAL])
    rotate_180 = any([subdir_dir in path for subdir_dir in LIST_ROTATE_180])

    # DataFrame for processing
    df_img = pd.DataFrame(data = {'file_name': list_images})     
    
    #Read each image in an 2D UTF-8 grayscale array
    if flip_vertical == True:
        df_img['raw'] = df_img['file_name'].map(lambda file_name: cv2.flip(cv2.imread(file_name,0),1))
    else:
        df_img['raw'] = df_img['file_name'].map(lambda file_name: cv2.imread(file_name,0))
    
    # Extract ionogram and coordinates delimiting its limits
    df_img['limits'],df_img['ionogram'] = zip(*df_img['raw'].map(lambda raw_img: extract_ionogram(raw_img)))
    
    # Record the files whose ionogram extraction was not successful
    df_loss_ion_extraction, loss_ion_extraction = record_loss(df_img,'extract_ionogram',subdir_location)

    # Remove files whose ionogram extraction was not successful
    df_img = df_img[~loss_ion_extraction]
    
    if rotate_180 == True:
        df_img['ionogram'] = df_img['ionogram'].map(lambda ionogram: np.rot90(ionogram, 2))

    # Extract the shape of each ionogram in subdirectory
    df_img['height'],df_img['width'] = zip(*df_img['ionogram'].map(lambda array_pixels: array_pixels.shape))
    
    #Find median height and width of ionogram in subdirectory
    median_height = np.median(df_img['height'])
    median_width = np.median(df_img['width'])
    
    # Find and remove ionogram outliers    
    conditional_list_ionogram = [abs(df_img['height'] -median_height)  >cutoff_height,abs(df_img['width'] - median_width) > cutoff_width] 
    outlier_ionogram = np.any(conditional_list_ionogram,axis = 0)
    
    df_outlier_ionogram,_ = record_loss(df_img,'image_segmentation.segment_images_in_subdir.segment_images: iono size outlier',subdir_location,['file_name','height','width'],outlier_ionogram)
    
    # Log outlier
    if not df_outlier_ionogram.empty:
        df_outlier_ionogram[ 'details'] = df_outlier_ionogram.apply(lambda row: 'height: ' + str(row['height'])+',width: ' + str(row['width']), 1)
        df_outlier_ionogram =df_outlier_ionogram[['file_name','func_name','subdir_name','details']]
    else:
        df_outlier_ionogram =df_outlier_ionogram[['file_name','func_name','subdir_name']]
    
    # Remove outlier
    df_img = df_img[~outlier_ionogram]
    

    # Raw metadata
    df_img['metadata_type'],df_img['raw_metadata'] = zip(*df_img.apply(lambda row: extract_metadata(row['raw'], row['limits']),1))
    if rotate_180 == True:
        df_img['raw_metadata'] = df_img['raw_metadata'].map(lambda raw_metadata: np.rot90(raw_metadata, 2))
    
    # There should be no metadata on left and top, especially after flipping
    outlier_metadata_location = np.any([df_img['metadata_type'] == 'right',df_img['metadata_type']=='top'],axis=0)
    df_outlier_metadata_location ,_ =  record_loss(df_img,'image_segmentation.segment_images_in_subdir.segment_images: metadata not on left or bottom',subdir_location,
                                         ['file_name','metadata_type'],outlier_metadata_location )

    if not df_outlier_metadata_location.empty:
        df_outlier_metadata_location['details'] = df_outlier_metadata_location.apply(lambda row: str(row['metadata_type']),1)
        df_outlier_metadata_location = df_outlier_metadata_location[['file_name','func_name','subdir_name','details']]
    else:
        df_outlier_metadata_location = df_outlier_metadata_location[['file_name','func_name','subdir_name']]
    
    # Remove loss from detected metadata not being on the left or bottom
    df_img = df_img[~outlier_metadata_location]
    
    # Trimmed metadata
    df_img['trimmed_metadata'] = df_img.apply(lambda row: trimming_metadata(row['raw_metadata'],row['metadata_type']) , 1)
    df_loss_trim,loss_trim = record_loss(df_img,'image_segmentation.trim_raw_metadata.trimming_metadata',subdir_location)

    
    # Remove files whose metadata trimming was not successful
    df_img = df_img[~loss_trim]
    
    
    # Check if metadata too small
    df_img['meta_height'],df_img['meta_width'] = zip(*df_img['trimmed_metadata'].map(lambda array_pixels: array_pixels.shape))
    outlier_size_metadata = np.logical_or(np.logical_and(df_img['metadata_type'] == 'left', 
                                                      df_img['meta_width'] < min_leftside_meta_width),
                                       np.logical_and(df_img['metadata_type'] == 'bottom', 
                                                      df_img['meta_height'] < min_bottomside_meta_height))
        
    df_outlier_metadata_size, _ = record_loss(df_img,'image_segmentation.segment_images_in_subdir.segment_images: metadata size outlier',subdir_location,
                                           ['file_name','metadata_type','meta_height','meta_width'],outlier_size_metadata)

    if not df_outlier_metadata_size.empty:
        df_outlier_metadata_size['details'] = df_outlier_metadata_size.apply(lambda row: row['metadata_type'] + '_height: ' + \
                                                    str(row['meta_height'])+',width: ' + str(row['meta_width']),1)
        df_outlier_metadata_size = df_outlier_metadata_size[['file_name','func_name','subdir_name','details']]
        
    else:
        df_outlier_metadata_size = df_outlier_metadata_size[['file_name','func_name','subdir_name']]
    
    # Remove files whose metadata too small
    df_img = df_img[~outlier_size_metadata]
    
    
    # Dataframe recording loss from programming errors
    df_loss = pd.concat([df_loss_ion_extraction,df_loss_trim])
    
    # Dataframe recording loss from various filters i.e. metadata too small, ionogram too small/big
    df_outlier = pd.concat([df_outlier_ionogram,df_outlier_metadata_location,df_outlier_metadata_size])
    
    return df_img,df_loss,df_outlier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process subdirectory

# COMMAND ----------

# -*- coding: utf-8 -*-
"""
Process all the raw images in a subdirectory
- Determine ionogram grid (pixel coordinates to Hz, km mappings)
- Determine leftside metadata grid (pixel coordinates to number, category mappings)
- For each raw image in the subdirectory,
    - Segment the raw image into raw ionogram and raw metadata
    - Trim the metadata
    - Translate the leftside metadata into information
    - Extract the coordinates of the ionogram trace (black)
    - Map the (x,y) pixel coordinates to (Hz, km) values
    - Extract select parameters i.e. fmin
"""
import ntpath
import glob
import os
import cv2
import numpy as np

def process_subdirectory(subdir_path, regex_images, output_folder_if_pickle,
                         min_n_leftside_metadata=10, only_ionogram_content_extraction_on_leftside_metadata=True, to_pickle=True):
    
    """Transform raw scanned images in a subdirectory into information
    
    :param subdir_path: path of subdir_path
    :type subdir_path: str
    :param regex_img: regular expression to extract images ex: '*.png'
    :type regex_img: str
    :param output_folder_if_pickle: output folder for pickle, use None if no pickle
    :type output_folder_if_pickle: string
    :param min_n_leftside_metadata: minimum number of ionograms with metadata on the left to be able to call metadata_translation.leftside_metadata_grid_mapping, defaults to 10
    :type min_n_leftside_metadata: int, optional
    :param only_ionogram_content_extraction_on_leftside_metadata: only run the scripts of ionogran_content_extraction on ionograms with metadata on the left, defaults to True
    :type only_ionogram_content_extraction_on_leftside_metadata: boolean, optional
    :param to_pickle: whether to save result of subdirectory processing as a pickle file, defaults to True
    :type to_pickle: boolean, optional
    :returns: df_processed, df_all_loss, df_outlier: :  dataframe containing data from running the full processing pipeline,dataframe containing file names leading to runtime errors, dataframe containing file names that do not pass pre-established filters (metadata size, ionogram size)
    :rtype: class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`,class: `pandas.core.frame.DataFrame`
    """
    # Run segment_images on the subdirectory 
    df_img,df_loss,df_outlier = segment_images(subdir_path, regex_images)

    # Determine ionogram grid mappings used to map (x,y) pixel coordinates of ionogram trace to (Hz, km) values
    stack = all_stack(df_img)
    col_peaks,row_peaks,mapping_Hz, mapping_km = get_grid_mappings(stack)

    # Translate metadata located on the left
    df_img_left = df_img[df_img['metadata_type']== 'left']
    
    if len(df_img_left.index) > min_n_leftside_metadata:
        # Determine leftside metadata grid (pixel coordinates to number, category mappings)
        df_img_left, df_loss_meta,dict_mapping,dict_hist= get_leftside_metadata(df_img_left,subdir_path)
        df_all_loss = df_loss.append(df_loss_meta)
    else:
        df_all_loss = df_loss
        
    #  Extract the coordinates of the ionogram trace (black), Map the (x,y) pixel coordinates to (Hz, km) values and Extract select parameters i.e. fmin
    if only_ionogram_content_extraction_on_leftside_metadata:
        df_processed, df_loss_coord = extract_coord_subdir_and_param(df_img_left,subdir_path,col_peaks,row_peaks,mapping_Hz,mapping_km)
    else:
        df_processed, df_loss_coord = extract_coord_subdir_and_param(df_img,subdir_path,col_peaks,row_peaks,mapping_Hz,mapping_km)

    df_processed['mapping_Hz'] = [mapping_Hz] * len(df_processed.index)
    df_processed['mapping_km'] = [mapping_km] * len(df_processed.index)

    if to_pickle:
        start,subdir_name = ntpath.split(subdir_path[:-1])
        start,dir_name = ntpath.split(start)
        df_processed.to_pickle(os.pardir + '/pickle/' + str(dir_name)+'_'+str(subdir_name)+'.pkl')
        
    df_all_loss = df_all_loss.append(df_loss_coord)
    return df_processed, df_all_loss,df_outlier

def process_df_leftside_metadata(df_processed,subdir_name,is_dot):
    """Process dataframe of subdirectory containing raw scanned images with leftside metadata 
    
    :param df_processed: regular expression to extract images ex: '*.png'
    :type df_processed: class: `pandas.core.frame.DataFrame`
    :param subdir_name: name of subdirectory
    :type subdir_path: str
    :param is_dot: whether the dataframe contains metadata
    :type is_dot: bool
    :returns: df_final_data :  dataframe containing data
    :rtype: class: `pandas.core.frame.DataFrame`
    """
    
    df_final_data = df_processed[['file_name','fmin', 'max_depth','dict_metadata']]
    df_final_data['subdir_name'] = subdir_name
    if is_dot:
        labels= ['day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2',
                             'second_1', 'second_2','station_code']
    else:
        labels = ['satellite_number','year','day_1','day_2','day_3','hour_1','hour_2','minute_1','minute_2',
                            'second_1', 'second_2', 'station_number_1','station_number_2']
        
    for label in labels:
        df_final_data[label] = df_final_data['dict_metadata'].map(lambda dict_meta: sum(dict_meta[label]) if label in dict_meta.keys() else 0)
    
    del df_final_data['dict_metadata']
    
    return df_final_data

def append_to_csv(csv_path, df):
    '''Append select data from a dataframe to a master CSV
    
    :param csv_path: path of csv to append data to
    :type: str
    :param df: dataframe to extract data from to add to a csv
    :type df: class:  `pandas.core.frame.DataFrame`
    
    '''
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a',header=False)
    else:
        df.to_csv(csv_path, mode='a')

# COMMAND ----------

# MAGIC %md
# MAGIC # Pipeline

# COMMAND ----------

try:
    dbutils.fs.mount(
  source = "wasbs://dcosting@dhdemosand.blob.core.windows.net",
  mount_point = "/mnt/test",
  extra_configs = {"fs.azure.sas.datahub.dhdemosand.blob.core.windows.net":"https://dhdemosand.blob.core.windows.net/datahub?sv=2020-08-04&st=2022-03-14T17%3A01%3A17Z&se=2022-03-28T17%3A01%3A17Z&sr=c&sp=racwdxltmei&sig=jACYD0vp3cHcLJ5HuqMrXUqCcsNiN744rpql1LvRjLk%3D"})
except:
    pass
import os
process_subdirectory("/dbfs/mnt/demo/AlouetteIonograms", "*.png", "/dbfs/mnt/demo/AlouetteOutput")


# COMMAND ----------


