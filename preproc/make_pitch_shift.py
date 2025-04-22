import os
import glob
import torch
import torchaudio
from tqdm import tqdm
from torch_pitch_shift import pitch_shift

# Define the pitch shift values in semitones
PITCH_SHIFTS = [0.5, -0.5, 1, -1, 2, -2, 3, -3]

# Set paths
audio_dir = "data/Audio"
output_dir = "data/PitchShiftedAudio"

# Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all WAV files in the audio directory
audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
total_files = len(audio_files)

print(f"Found {total_files} audio files to process")

# select hash
import json
import unicodedata
label_json = 'data/label.json'
loaded_hash = []
with open(label_json, 'r', encoding='utf-8') as file: label_data = json.load(file)

for item in tqdm(label_data, desc='Load Hash'):
    hash_key = unicodedata.normalize('NFC', item["file_upload"].split("-")[0])
    loaded_hash.append(hash_key)

# Process each audio file
for i, audio_file in tqdm(enumerate(audio_files)):
    filename = os.path.basename(audio_file)
    hash_key = unicodedata.normalize('NFC', filename.split('-')[0])
    if hash_key not in loaded_hash: continue

    base_name, ext = os.path.splitext(filename)
    
    print(f"Processing file {i+1}/{total_files}: {filename}")
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    if waveform.ndim == 2:
      waveform = waveform.mean(dim=0, keepdim=True)
    
    
    # Explicitly convert sample_rate to an integer
    sample_rate_int = int(sample_rate)
    
    # Move to GPU if available
    waveform = waveform.to(device)
    
    # Apply pitch shifts
    for shift in tqdm(PITCH_SHIFTS, leave=False):
        # Create output filename with suffix indicating pitch shift
        sign = "p" if shift > 0 else "m"  # p for positive, m for negative
        abs_shift = abs(shift)
        # Format with 1 decimal place if it has a fractional part, otherwise as an integer
        if abs_shift == int(abs_shift):
            shift_str = f"{int(abs_shift)}"
        else:
            shift_str = f"{abs_shift:.1f}".replace(".", "p")
        
        output_filename = f"{base_name}_ps_{sign}{shift_str}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"  Skipping existing file: {output_filename}")
            continue
        
        print(f"  Creating pitch shift {shift} semitones: {output_filename}")
        
        # Apply pitch shift (stays on GPU) - use the integer sample rate
        shifted_waveform = pitch_shift(waveform.unsqueeze(0), shift, sample_rate_int)
        
        # Move back to CPU for saving
        shifted_waveform = shifted_waveform.cpu()
        
        # Save the pitch-shifted audio
        torchaudio.save(output_path, shifted_waveform.squeeze(0), sample_rate_int)

print("Pitch shifting complete!")
