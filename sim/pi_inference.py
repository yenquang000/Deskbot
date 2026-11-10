"""NumPy-only policy for the Raspberry Pi."""
import numpy as np


class DeskbotPolicy:
    def __init__(self, path):
        d = np.load(path)
        self.n = int(d["n_hidden"])
        self.obs_dim = int(d["obs_dim"])
        self.W = [d[f"W{i}"] for i in range(self.n)]
        self.b = [d[f"b{i}"] for i in range(self.n)]
        self.Wout, self.bout = d["Wout"], d["bout"]

    def logits(self, obs):
        x = np.asarray(obs, np.float32)
        for W, b in zip(self.W, self.b):
            x = np.tanh(W @ x + b)
        return self.Wout @ x + self.bout

    def act(self, obs):
        """Deterministic action: 0 forward, 1 backward, 2 turn left, 3 turn right."""
        return int(np.argmax(self.logits(obs)))
