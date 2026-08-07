"""Central path resolution for the repo.

Everything outside the repository (the audio corpus, the sibling MIDI-detection
repo) is located through environment variables so nothing is tied to one
machine. Defaults assume the conventional sibling layout:

    <parent>/
    ├── PansoriModeChangeDetection/   <- this repo
    ├── Pansori_Data/                 <- PANSORI_DATA_ROOT
    └── PansoriMIDIDetection/         <- PANSORI_MIDI_REPO

Override either with an environment variable:

    export PANSORI_DATA_ROOT=/data/Pansori_Data
    export PANSORI_MIDI_REPO=/code/PansoriMIDIDetection

Scripts under scripts/ import this after putting the repo root on sys.path:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from paths import REPO_ROOT, DATA_ROOT, MIDI_REPO
"""

import os
from pathlib import Path

#: Repository root (the directory containing this file).
REPO_ROOT = Path(__file__).resolve().parent

#: Root of the Pansori corpus: audio, extracted pitch, and fold-split indices.
#: Not distributed with this repo -- see the README.
DATA_ROOT = Path(os.environ.get("PANSORI_DATA_ROOT", REPO_ROOT.parent / "Pansori_Data"))

#: Sibling PansoriMIDIDetection checkout. Required only by the ensemble
#: scripts and the viewer, which load MIDI-modality checkpoints from it.
MIDI_REPO = Path(os.environ.get("PANSORI_MIDI_REPO", REPO_ROOT.parent / "PansoriMIDIDetection"))

#: Label directory shipped inside this repo.
LABEL_DIR = REPO_ROOT / "data" / "Label"

#: Default location for trained checkpoints (gitignored).
WEIGHTS_DIR = Path(os.environ.get("PANSORI_WEIGHTS_DIR", REPO_ROOT / "weights"))


def require(path: Path, what: str) -> Path:
    """Fail loudly and early with a fixable message instead of a late KeyError.

    Most scripts here need data that is not in the repo; a missing corpus should
    say which env var to set, not surface 200 lines later as an empty glob.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at: {path}\n"
            f"Set the corresponding environment variable "
            f"(PANSORI_DATA_ROOT / PANSORI_MIDI_REPO) or create the path.\n"
            f"See the README section 'Expected data layout'."
        )
    return path
