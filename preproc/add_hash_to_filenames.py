
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import REPO_ROOT
#!/usr/bin/env python3
"""
Script to add hash values to audio filenames based on matching prefixes.
For files like: 01-김수연-심청가_청이_밥_빌러_가는데_001_계면조.wav
Extracts prefix: 01-김수연-심청가_청이_밥_빌러_가는데_
And adds a hash to group files with the same prefix.
"""

import os
import hashlib
import re
from pathlib import Path
from collections import defaultdict


def extract_prefix(filename):
    """
    Extract the prefix from a filename.
    Pattern: artist-song_description_XXX_mode.wav
    Returns: artist-song_description_

    Example:
        01-김수연-심청가_청이_밥_빌러_가는데_001_계면조.wav
        -> 01-김수연-심청가_청이_밥_빌러_가는데_
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Pattern to match: everything up to _XXX_mode pattern
    # Where XXX is typically a 3-digit number
    match = re.match(r'(.+?)_(\d{3})_(.+)', name_without_ext)

    if match:
        return match.group(1) + '_'

    # Fallback: return full name if pattern doesn't match
    return name_without_ext


def generate_hash(prefix_text, length=8):
    """Generate a short hash from the prefix text."""
    hash_obj = hashlib.md5(prefix_text.encode('utf-8'))
    return hash_obj.hexdigest()[:length]


def rename_files_with_hash(directory_path, dry_run=True):
    """
    Rename audio files by adding hash values based on their prefix.

    Args:
        directory_path: Path to the directory containing audio files
        dry_run: If True, only print what would be done without renaming
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return

    # Group files by prefix
    prefix_groups = defaultdict(list)

    # Get all .wav files
    wav_files = sorted(directory.glob('*.wav'))

    if not wav_files:
        print("No .wav files found in directory")
        return

    print(f"Found {len(wav_files)} .wav files\n")

    # Group files by their prefix
    for file_path in wav_files:
        prefix = extract_prefix(file_path.name)
        prefix_groups[prefix].append(file_path)

    print(f"Found {len(prefix_groups)} unique prefixes\n")

    # Display groups and prepare renaming
    rename_operations = []

    for prefix, files in sorted(prefix_groups.items()):
        hash_value = generate_hash(prefix)
        print(f"Prefix: {prefix}")
        print(f"Hash: {hash_value}")
        print(f"Files in group: {len(files)}")

        for file_path in files:
            # Extract components
            original_name = file_path.name

            # Create new filename with hash at the beginning
            # Pattern: [hash]-original_name.wav
            new_name = f"{hash_value}-{original_name}"
            new_path = file_path.parent / new_name

            rename_operations.append((file_path, new_path))
            print(f"  {file_path.name} -> {new_name}")

        print()

    # Perform renaming
    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE - No files were actually renamed")
        print("Set dry_run=False to perform actual renaming")
        print("=" * 70)
    else:
        print("=" * 70)
        print("Performing rename operations...")
        print("=" * 70)

        for old_path, new_path in rename_operations:
            try:
                old_path.rename(new_path)
                print(f"✓ Renamed: {old_path.name} -> {new_path.name}")
            except Exception as e:
                print(f"✗ Error renaming {old_path.name}: {e}")

        print("\nRenaming complete!")


if __name__ == "__main__":
    # Configuration
    AUDIO_DIR = str(REPO_ROOT / 'data/audio')

    # Run in dry-run mode first to preview changes
    print("Running in DRY RUN mode to preview changes...\n")
    rename_files_with_hash(AUDIO_DIR, dry_run=False)

    # Uncomment the line below to actually perform the renaming
    # rename_files_with_hash(AUDIO_DIR, dry_run=False)
