"""Mode inference from a CQT spectrogram.

Usage:
    python infer/infer_cqt.py path/to/song.wav

NOTE: no CQT checkpoint ships with this repo -- train one first
(`python train.py --config-name cqt_base`) and pass its run dir.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('cqt', 'weights/frame/CQT_Version').parse_args()
    run_single_representation('cqt', args)
