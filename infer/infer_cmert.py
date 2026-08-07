"""Mode inference from CultureMERT-95M features.

Downloads ntua-slp/CultureMERT-95M on first run and needs ~250 MB of weights,
so this is the heaviest representation to run.

Usage:
    python infer/infer_cmert.py path/to/song.wav
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('cmert', 'weights/cmert/layer08_10k_version/layer08/0803_0024_Audio_Original_CMERTClassifier_CMERTFrameDataset').parse_args()
    run_single_representation('cmert', args)
