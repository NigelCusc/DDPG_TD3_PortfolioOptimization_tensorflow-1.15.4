"""
Contains a set of utility function to process data
"""

from __future__ import print_function
import numpy as np
import h5py
import statistics

exclude_set = {}

def normalize(x):
    """ Create a universal normalization function across close/open ratio

    Args:
        x: input of any shape

    Returns: normalized data

    """
    #return (x - 1) * 100
    
    w = x.shape[1]
    a = x.shape[0]
    _x = x.reshape(a,w)
    result = x.copy()
    
    for i in range(w):
        mean = sum(_x[:,i])/a 
        std = statistics.stdev(_x[:,i])
        
        for j in range(a):
            if std != 0:
                result[j,i,0] = (_x[j,i] - mean)/std
            else: 
                result[j,i,0] = (_x[j,i] - mean)
    
    return result


def write_to_h5py(history, abbreviation, date_list, filepath='datasets/stocks_history_2.h5'):
    """ Write a numpy array history and a list of string to h5py

    Args:
        history: (N, timestamp, 5)
        abbreviation: a list of stock abbreviations
        dates
    Returns:

    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('history', data=history)
        abbr_array = np.array(abbreviation, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("abbreviation", data=abbr_array, dtype=string_dt)
        date_array = np.array(date_list, dtype=object)
        f.create_dataset("dates", data=date_array, dtype=string_dt)


def read_stock_history(filepath='datasets/stocks_history.h5'):
    """ Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        abbreviation:
        dates:
    """
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        abbreviation = f['abbreviation'][:].tolist()
        if not isinstance(abbreviation[0], str):
            abbreviation = [abbr.decode('utf-8') for abbr in abbreviation]
        dates = f['dates'][:].tolist()
    return history, abbreviation, dates

