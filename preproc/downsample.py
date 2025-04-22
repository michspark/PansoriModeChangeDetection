import os
import shutil
import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm

def downsample_audio(input_path, output_path, target_sr=16000, mono=True):
    """
    Downsample an audio file to target sample rate and convert to mono if needed
    using torchaudio
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the downsampled audio
        target_sr: Target sample rate (default: 16000)
        mono: Whether to convert to mono (default: True)
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_path)
    
    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=target_sr
        )
        waveform = resampler(waveform)
    
    # Save the processed audio
    torchaudio.save(output_path, waveform, target_sr)

def process_data_directory():
    # Define directories
    data_dir = Path("data")
    data_original_dir = Path("data_original")
    
    # Create data_original directory if it doesn't exist
    os.makedirs(data_original_dir, exist_ok=True)
    
    # Find all WAV files in data directory and its subdirectories
    wav_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(wav_files)} WAV files in data directory")
    
    # Process each WAV file
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        # Create relative path to determine output location
        rel_path = os.path.relpath(wav_file, data_dir)
        
        # Define output paths
        original_output_path = os.path.join(data_original_dir, rel_path)
        processed_output_path = wav_file  # Same as original path
        
        # Create directory structure in data_original if needed
        os.makedirs(os.path.dirname(original_output_path), exist_ok=True)
        
        # Create a temporary path for the processed file
        temp_output_path = wav_file + ".temp.wav"
        
        # Downsample and convert to mono
        downsample_audio(wav_file, temp_output_path)
        
        # Move original file to data_original directory
        shutil.move(wav_file, original_output_path)
        
        # Move temp file to original path
        shutil.move(temp_output_path, processed_output_path)
        
    print(f"Successfully processed {len(wav_files)} files")
    print(f"Original files moved to {data_original_dir}")
    print(f"Downsampled files saved in {data_dir}")

if __name__ == "__main__":
    process_data_directory()
