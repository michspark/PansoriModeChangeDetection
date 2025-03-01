import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import medfilt
from collections import Counter
from scipy.ndimage import median_filter

def frequency_to_midi(frequency):
    return 69 + 12 * np.log2(frequency / 440)

def get_midi_contour_from_csv(csv_fn, low_pitch=41, high_pitch=77):
    df = pd.read_csv(csv_fn)
    df.columns = ["time", "frequency", "confidence"]
    frequency = df['frequency'].values
    confidence = df['confidence'].values
    threshold = 0.7
    frequency[confidence < threshold] = np.nan
    filtered_frequency = median_filter(frequency, size = 5, mode = 'nearest')

    pitch_in_midi = [frequency_to_midi(freq) for freq in filtered_frequency]
    midi_notes = [x for x in pitch_in_midi if low_pitch <= x < high_pitch]

    return midi_notes

def get_histogram(midi_notes, bin_size = 0.25):
    edges = np.arange(41, 77 + bin_size, bin_size)
    hist, _ = np.histogram(midi_notes, bins = edges, density = True)
    return hist, edges

def main(file_path, output_path):
    os.makedirs(output_path, exist_ok = True)

    for file_name in os.listdir(file_path):
        if file_name.endswith(".csv"):
            f0_path = os.path.join(file_path, file_name)
            midi_notes = get_midi_contour_from_csv(f0_path)
            histogram, edges = get_histogram(midi_notes)
            save_file = os.path.join(output_path, f"{file_name.replace('.csv', '')}.npy")
            np.save(save_file, histogram)

if __name__ == "__main__":
    if __name__ == "__main__":
        input_path = "/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/F0/F0_Split_10/화초가_심청가_우조"
        output_path = "/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/Histogram/화초가_심청가_우조"
        main(input_path, output_path)