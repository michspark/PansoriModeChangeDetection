import pretty_midi
import os
import numpy as np

class PianoRollGenerator:
    def __init__(self, midi_path: str, fs: int = 100):
        self.midi = pretty_midi.PrettyMIDI(midi_path)
        self.fs = fs
        self.filename = os.path.basename(midi_path)
        self.roll: np.ndarray = None

    def generate_piano_roll(self) -> np.ndarray:
        if self.roll is None:
            self.roll = self.midi.get_piano_roll(fs=self.fs)
        return self.roll