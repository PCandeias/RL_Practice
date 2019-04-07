import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.memory = []
        self.cur_idx = 0

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]

    def sample(self, n_samples):
        idxs = np.random.randint(0, len(self.memory), n_samples)
        return [self.memory[idx] for idx in idxs], idxs

    def append(self, element):
        if self.max_len > len(self.memory):
            self.memory.append(element)
        else:
            self.memory[self.cur_idx] = element

        self.cur_idx = (self.cur_idx + 1) % self.max_len

class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, max_len):
        super(PriorityReplayBuffer, self).__init__(max_len)
        self.sumtree = np.zeros(2 * max_len - 1)
        self.max_priority = 1.0

    def append(self, element):
        idx = self.cur_idx
        super().append(element)
        s_idx = self.max_len + idx - 1
        dif = self.max_priority - self.sumtree[s_idx]
        self._propagate(s_idx, dif)

    def sample(self, n_samples):
        ps = np.random.rand(n_samples) * self.sumtree[0]
        samples = []
        idxs = []
        weights = []
        for p in ps:
            sample, idx, r_p = self._retrieve(0, p)
            samples.append(sample)
            idxs.append(idx)
            weights.append(r_p)

        return samples, idxs, self.sumtree[0] / (len(self.memory) * np.array(weights, copy=False))

    def _retrieve(self, s_idx, val):
        left = s_idx * 2 + 1
        right = left + 1

        if left >= len(self.sumtree): # Reached leaf?
            idx = s_idx - self.max_len + 1
            return self.memory[idx], idx, self.sumtree[s_idx]

        if self.sumtree[left] >= val:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val - self.sumtree[left])

    def update(self, idx, p):
        assert p > 0

        s_idx = self.max_len + idx - 1
        dif = (p - self.sumtree[s_idx])
        self._propagate(s_idx, dif)
        self.max_priority = max(self.max_priority, p)

    def _propagate(self, s_idx, dif):
        """
        s_idx: s_idx in sumtree
        """
        if s_idx < 0:
            return

        self.sumtree[s_idx] += dif 
        parent = (s_idx-1) // 2
        self._propagate(parent, dif)
        
    def total_weight(self):
        return self.sumtree[0]



