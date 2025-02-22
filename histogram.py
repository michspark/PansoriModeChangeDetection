import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from scipy.signal import medfilt 

input_dir = '/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/F0'
output_dir = '/home/sangheon/Desktop/Pansori_2025_ISMIR/PansoriData/Histogram'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def f0_to_midi(f0):
    return 69 + 12 * np.log2(f0 / 440)

def midi_to_note(midi_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    octave = int(midi_number // 12) - 1
    note_index = int(midi_number % 12) % 12

    return f"{note_names[note_index]}{octave}"

def apply_median_filter(f0_series, kernel_size=5):

    return medfilt(f0_series, kernel_size=kernel_size)

def main():
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)

            df = pd.read_csv(file_path, header = None)
            df.columns = ['Time', 'F0', 'Confidence']
            df.loc[df["Confidence"] <= 0.7, "F0"] = np.nan

            f0_values = df["F0"].dropna()

            lower_bound = np.percentile(f0_values, 6)
            upper_bound = np.percentile(f0_values, 94)
            filtered_f0 = f0_values[(f0_values >= lower_bound) & (f0_values <= upper_bound)]
            smoothed_f0 = apply_median_filter(filtered_f0.values)

            midi_notes = f0_to_midi(smoothed_f0) 
            midi_bins = np.arange(midi_notes.min(), midi_notes.max() + 1)
            note_labels = [midi_to_note(m) for m in midi_bins]

            plt.figure(figsize=(12, 6))
            plt.hist(midi_notes, bins = midi_bins, edgecolor='black', alpha=0.7)
            plt.xticks(midi_bins[::4], note_labels[::4], rotation=45)
            plt.xlabel("Pitch (Note Names)")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Filtered & Smoothed F0 Values - {file_name}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            output_path = os.path.join(output_dir, f"{file_name.replace('.csv', '')}_histogram.png")

            plt.savefig(output_path)
            plt.close()
            print(f"Processed: {file_name} -> {output_path}")

if __name__ == "__main__":
    main() 
