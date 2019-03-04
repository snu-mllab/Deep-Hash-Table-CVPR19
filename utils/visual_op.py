import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import matplotlib.pyplot as plt
import numpy as np

def image_normalizer(image_set):
    save_shape = image_set.shape

    image_set = np.reshape(image_set, [save_shape[0], -1])
    max_array = np.max(image_set, axis=0)
    min_array = np.min(image_set, axis=0)

    image_set = np.divide(image_set-min_array, max_array-min_array)
    image_set = np.reshape(image_set, save_shape)
    return image_set

def generate_query_retrieve_results(query_idx, search_result):
    '''
    Args:
        query_idx - int
        search_result - Numpy 2D array [nquery, nsearch]
    Return:
        Numpy 1D array
            [query_idx, retrieve_indices]
    '''
    return np.concatenate([np.array([query_idx]), search_result[query_idx]], axis=0)

def idx_array2matrix_image(image_set, idx_array, ncol, nrow):
    '''
    Args:
        image_set - Numpy 4D array [nimage, height, width, 3 or 1]
        idx_array - Numpy 1D array 
        ncol - int
        nrow - int
    Return:
        matrix_image - Numpy 5D array
            [nrow, ncol, height, width, 3 or 1]
    '''
    null_image = np.ones(image_set.shape[1:])

    matrix_image = [[null_image for v in range(ncol)] for v in range(nrow)]

    idx_array_length = len(idx_array)

    for i in range(min(idx_array_length, ncol*nrow)):
        r_idx = i//ncol
        c_idx = i%ncol
        image_idx = idx_array[i]
        matrix_image[r_idx][c_idx] = image_set[image_idx]
    matrix_image = np.array(matrix_image)
    return matrix_image

def idx_set2matrix_image(image_set, idx_set):
    '''
    Args:
        image_set - Numpy 4D array [nimage, height, width, 3 or 1]
        idx_set - Numpy 2D array [nrow, ncol]
    Return:
        matrix_image - Numpy 5D array 
            [nrow, ncol, height, width, 3 or 1]
    '''
    nrow, ncol = idx_set.shape

    matrix_image = list()
    for r_idx in range(nrow):
        tmp = list()
        for c_idx in range(ncol):
            image_idx = idx_set[r_idx][c_idx]
            tmp.append(image_set[image_idx])
        matrix_image.append(tmp)
    matrix_image = np.array(matrix_image)
    return matrix_image

def matrix_image2big_image(matrix_image, row_margin=5, col_margin=5):
    '''
    Args:
        matrix_image - Numpy 5D array
            [nrow, ncol, height, width, 3 or 1
        row_margin - int
            defaults to be 5
        col_margin - int
            defaults to be 5
    '''
    nrow, ncol, height, width, nch = matrix_image.shape
    big_row = nrow*height + nrow*row_margin
    big_col = ncol*width + ncol*col_margin
    big_image = np.ones([big_row, big_col, nch])

    for r_idx in range(nrow): 
        for c_idx in range(ncol):
            for h_idx in range(height):
                for w_idx in range(width):
                    big_image_h_idx = r_idx*(height+row_margin)+h_idx
                    big_image_w_idx = c_idx*(width+col_margin)+w_idx
                    for ch_idx in range(nch): big_image[big_image_h_idx][big_image_w_idx][ch_idx] = matrix_image[r_idx][c_idx][h_idx][w_idx][ch_idx]
    return np.squeeze(big_image)

