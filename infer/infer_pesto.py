"""Mode inference from a PESTO f0 contour CSV.

The CSV needs `frequency` (Hz) and `confidence` columns at 100 Hz. Tonic
normalisation is applied here, so a raw PESTO output file works as-is.

Usage:
    python infer/infer_pesto.py path/to/song.f0.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('pesto', 'weights/frame/Pesto_Version').parse_args()
    run_single_representation('pesto', args)
