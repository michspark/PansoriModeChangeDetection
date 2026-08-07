"""Mode inference from a chromagram.

Usage:
    python infer/infer_chroma.py path/to/song.wav

NOTE: no chroma checkpoint ships with this repo -- train one first
(`python train.py --config-name chroma_base`) and pass its run dir.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('chroma', 'weights/frame/Chroma_Version').parse_args()
    run_single_representation('chroma', args)
