import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from collections import Counter

'''
Generate 5th interval Normalized based on 10 second split clip and full song clip to find tonic of the song

'''

def frequency_to_midi(frequency):
    return 69 + 12 * np.log2(frequency / 440)

def get_midi_contour_from_csv(csv_fn, low_pitch = 41, high_pitch = 77):
    df = pd.read_csv(csv_fn)
    df.columns = ["time", "frequency"]
    frequency = df['frequency'].values
    filtered_frequency = median_filter(frequency, size=5, mode='nearest')

    pitch_in_midi = [frequency_to_midi(freq) for freq in filtered_frequency]
    midi_notes = [x for x in pitch_in_midi if low_pitch <= x < high_pitch]

    return midi_notes

def tonic_get_midi_contour_from_csv(csv_fn, low_pitch = 41, high_pitch = 77):
    df = pd.read_csv(csv_fn)
    df.columns = ["time", "frequency", "confidence"]
    frequency = df['frequency'].values
    confidence = df['confidence'].values
    threshold = 0.7
    frequency[confidence < threshold] = np.nan
    filtered_frequency = median_filter(frequency, size=5, mode='nearest')

    pitch_in_midi = [frequency_to_midi(freq) for freq in filtered_frequency]
    midi_notes = [x for x in pitch_in_midi if low_pitch <= x < high_pitch]

    return midi_notes

def compensate_tuning(midi_notes):
    max_appearance = 0
    best_comp = 0
    best_midi_notes = []

    for compensate in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        adjusted_midi = [round(midi + compensate) for midi in midi_notes]
        common_pitch, num_appearance = Counter(adjusted_midi).most_common(1)[0]

        if num_appearance > max_appearance:
            max_appearance = num_appearance
            best_comp = compensate
            best_midi_notes = adjusted_midi

    return best_midi_notes, best_comp

def get_tonic(midi_notes):
    unique_notes, counts = np.unique(midi_notes, return_counts=True)
    max_sum = 0
    best_tonic = unique_notes[0]

    for note in unique_notes:
        shifted_note = note + 7
        total_count = counts[np.where(unique_notes == note)].sum()

        if shifted_note in unique_notes:
            total_count += counts[np.where(unique_notes == shifted_note)].sum()

        if total_count > max_sum:
            max_sum = total_count
            best_tonic = note

    return best_tonic

def get_histogram(midi_notes, bin_size=0.25, best_comp=0.0, best_tonic = 55):

    lowest_edge = best_tonic - 12.5 - bin_size / 2
    highest_edge = best_tonic + 12.5 + bin_size / 2

    edges = np.arange(lowest_edge, highest_edge + bin_size, bin_size)
    bins = edges[:-1] + bin_size / 2

    edges, bins

    edges = np.arange(best_tonic - 12.5, best_tonic + 12.5 + bin_size, bin_size)

    if len(midi_notes) == 0:
        return np.zeros(len(edges) - 1)

    hist = np.histogram(midi_notes, bins=edges, density=False)[0]

    return hist, edges

def main():
    tonic_file_path = '/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/F0/F0/수궁가 중 소지노화(신만엽제)_Ujo.csv'
    file_path = '/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/F0/F0_Masked_Split_10/수궁가_소지노화_우조'
    output_path = '/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/Histogram/5th_int_masked/수궁가_소지노화_우조'

    os.makedirs(output_path, exist_ok=True)

    tonic_midi_notes = tonic_get_midi_contour_from_csv(tonic_file_path)
    tonic_midi_notes, best_comp = compensate_tuning(tonic_midi_notes)
    best_tonic = get_tonic(tonic_midi_notes)
    for file_name in os.listdir(file_path):
        if file_name.endswith(".csv"):
            f0_path = os.path.join(file_path, file_name)
            midi_notes = get_midi_contour_from_csv(f0_path)
            midi_notes, best_comp = compensate_tuning(midi_notes)

        if len(midi_notes) > 0:
            histogram, edges = get_histogram(midi_notes, best_comp = best_comp, best_tonic = best_tonic)
            save_file = os.path.join(output_path, f"{file_name.replace('.csv', '')}.npy")
            np.save(save_file, histogram)

if __name__ == "__main__":
  main()