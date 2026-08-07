"""Mode inference from a transcribed-MIDI piano roll.

Usage:
    python infer/infer_midi.py path/to/song.mid
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('midi', 'weights/midi/MIDI_Version/23-43-24').parse_args()
    run_single_representation('midi', args)
