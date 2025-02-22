import numpy as np 
import pandas as pd 
import scipy.ndimage
import os 
import csv
from scipy.ndimage import median_filter

"""
Algorithm implemented from "DETECTING STABLE REGIONS IN FREQUENCY TRAJECTORIES FOR
TONAL ANALYSIS OF TRADITIONAL GEORGIAN VOCAL MUSIC" 
"""

def filter_pitch_contour(file_path):

    df = pd.read_csv(file_path, header = None)
    df.columns = ['Time', 'F0', 'Confidence']
    
    df.loc[df["Confidence"] <= 0.7, "F0"] = np.nan
    
    f0_values = df["F0"].to_numpy()
    
    time_values = df["Time"].to_numpy()

    lower_bound = np.nanpercentile(f0_values, 5)
    upper_bound = np.nanpercentile(f0_values, 95)

    filtered_f0 = np.where((f0_values < lower_bound) | (f0_values > upper_bound), np.nan, f0_values)

    return filtered_f0, time_values

def freq_to_cent(freq):
    return 1200 * np.log2(freq / 440.0)

def morphological_filter(frequency, time, window_size = 43, threshold = 90):

    """
    Applies morphological filtering to detect stable pitch regions

    Parameters:
    - frequency = pitch values in cents 
    - time = time values 
    - window_size: window size 
    - threshold: threshold in cents to define stability 
    
    Returns:
    -filtered_pitch: numpy array, stable pitch values 

    """
    gamma_max = scipy.ndimage.maximum_filter(frequency, size = window_size, mode = 'nearest')

    gamma_min = scipy.ndimage.minimum_filter(frequency, size = window_size, mode = 'nearest')

    morph_gradient = gamma_max - gamma_min

    stable_mask = np.abs(morph_gradient) <= threshold

    filtered_trajectory = np.where(stable_mask, frequency, np.nan)

    time_values = time 

    return filtered_trajectory, time_values

def masking_filter(frequency, time, R = 10, index = 5, window =43):

    """
    Applies a 2D-masking approach to detect stable pitch regions without using cv2.dilate.
    
    Parameters:
    - frequency_trajectory: numpy array, pitch values in cents (with NaN for missing values)
    - R: frequency resolution in cents
    - beta: frequency tolerance (in bins)
    - L: window size for median filtering (odd integer)
    
    Returns:
    - filtered_trajectory: numpy array, stable pitch values (NaN where unstable)
    """

    valid_mask = ~np.isnan(frequency)

    binned_freq = np.full_like(frequency,np.nan, dtype = float)

    binned_freq[valid_mask] = np.round(frequency[valid_mask] / R).astype(int)

    max_bin = np.nanmax(binned_freq) + 1

    binary_mask = np.zeros((len(frequency), int(max_bin)), dtype = np.uint8)

    for i, b in enumerate(binned_freq):
        if not np.isnan(b) and b >= 0:
            binary_mask[i, int(b)] = 1

    T, F = binary_mask.shape

    binary_mask_copy = np.copy(binary_mask)

    for t in range(T):
        for f in range(F):
            if binary_mask[t, f] == 1:
                lower = max(0, f - index)
                upper = min(F, f + index + 1)
    
    binary_mask_copy = median_filter(binary_mask_copy, size = (43, 1), mode = 'nearest')

    masked_frequency = np.full_like(frequency, np.nan, dtype = float)

    for i, b in enumerate(binned_freq):
        if not np.isnan(b) and binary_mask_copy[i, int(b)] == 1:
            masked_frequency[i] = frequency[i]

    return masked_frequency, time

def main():

    file_path ='/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/F0/김창환 춘향가-이별가.csv'

    output_file_path = "Desktop/Pansori_2025_ISMIR/PansoriData/morph_filtered_pitch.csv"

    masked_output_file_path = "Desktop/Pansori_2025_ISMIR/PansoriData/masked_filtered_pitch.csv"

    filtered_freq, time_values = filter_pitch_contour(file_path)

    filtered_cent = freq_to_cent(filtered_freq)

    morph_filtered_freq, time_values = morphological_filter(filtered_cent, time_values)

    masked_filtered_freq, masked_time_values = masking_filter(filtered_freq, time_values)

    rows = zip(time_values, morph_filtered_freq)

    with open(output_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Frequency'])
        for row in rows:
            writer.writerow(row)

    masked_rows = zip(masked_time_values, masked_filtered_freq)

    with open(masked_output_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time','Frequency'])
        for row in masked_rows:
            writer.writerow(row)

if __name__ == "__main__":
    main()