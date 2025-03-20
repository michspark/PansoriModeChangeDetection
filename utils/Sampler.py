import random
from torch.utils.data import Sampler


class BaseSampler(Sampler):
    def __init__(self, segment_counts):
        self.segment_counts = segment_counts
        self.ids = list(segment_counts.keys())
        self.total_segments = sum(segment_counts.values())
    
    def __iter__(self):
        indices = []
        for idx in self.ids: indices.extend([idx] * self.segment_counts[idx])
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.total_segments

class FixedSampler(Sampler):
    def __init__(self, segment_counts):
        self.segment_counts = segment_counts
        self.indices = []
        for idx, count in self.segment_counts.items(): self.indices.extend([idx] * count)
        random.shuffle(self.indices)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)