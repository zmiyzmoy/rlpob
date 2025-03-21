# replay_buffer.py
import collections
import numpy as np
import torch
import logging
from config import device, buffer_capacity


class PrioritizedExperienceBuffer:
    def __init__(self, capacity=buffer_capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.visit_counts = collections.deque(maxlen=capacity)

    def add(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.visit_counts.append(0)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32) + 1e-5
        visits = np.array(self.visit_counts, dtype=np.float32) + 1
        adjusted_priorities = priorities / visits
        probs = adjusted_priorities ** 0.6 / adjusted_priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        for idx in indices:
            self.visit_counts[idx] += 1
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights).to(device)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5

    def merge(self, other_buffer):
        for exp, pri, vis in zip(other_buffer.buffer, other_buffer.priorities, other_buffer.visit_counts):
            if len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(exp)
                self.priorities.append(pri)
                self.visit_counts.append(vis)

    def __len__(self):
        return len(self.buffer)
