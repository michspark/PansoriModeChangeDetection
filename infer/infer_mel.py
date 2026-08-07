"""Mode inference from a mel spectrogram (source-separated vocal audio).

Usage:
    python infer/infer_mel.py path/to/song.wav --out outputs/infer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('mel', 'weights/frame/Mel_Original_Version').parse_args()
    run_single_representation('mel', args)
