"""Mode inference from a CREPE f0 contour CSV.

Same input contract as infer_pesto.py: `frequency` + `confidence` at 100 Hz.

Usage:
    python infer/infer_crepe.py path/to/song.f0.csv

NOTE: no CREPE checkpoint ships with this repo -- train one first
(`python train.py --config-name crepe_best`) and pass its run dir.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, run_single_representation



if __name__ == '__main__':
    args = build_parser('crepe', 'weights/frame/Crepe_Version').parse_args()
    run_single_representation('crepe', args)
