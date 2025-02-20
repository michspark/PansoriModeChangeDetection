import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from scipy.signal import medfilt 

input_dir = '/home/sangheon/Desktop/Pansori/PansoriData/F0'
output_dir = '/home/sangheon/Desktop/Pansori/PansoriData/Histogram'

def f0_to_midi(f0):
    df = pd.read.csv(f0)
    df.columns = ['Time,', 'F0', 'Confidence']
    df.loc[df]
    return 69 + 12 * np.log2(f0 / 440)

def apply_median_filter(f0_series, kernel_size = 5):
    return medfilt(f0_series, kernel_size)

def get_normal_pitch_histogram():
    pass

def stable_region():
    pass

def stable_region_duration_histogram():
    pass

def stable_region_count_histogram():
    pass