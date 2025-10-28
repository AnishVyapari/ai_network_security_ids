"""
ML model definitions for IDS: RandomForest and DNN alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


@dataclass
class IDSModel:
    model_type: str = 'rf'  # 'rf' or 'dnn'
    n_estimators: int = 200
    max_depth: Optional[int] = None
    random_state: int = 42
    dnn_hidden: int = 64
    dnn_layers: int = 2
    dnn_lr: float = 1e-3
    dnn_epochs: int = 10
    dnn_batch_size: int = 64

    def __post_init__(self):
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == 'dnn':
            if torch is None:
                raise ImportError('PyTorch is required for DNN model. pip install torch')
            self.model = _SimpleDNN(input_dim=None, hidden=self.dnn_hidden, layers=self.dnn_layers)
            self._criterion = nn.BCEWithLogitsLoss()
            self._optimizer = optim.Adam(self.model.parameters(), lr=self.dnn_lr)
        else:
            raise ValueError("model_type must be 'rf' or 'dnn'")

    # ---------------
    # I/O
    # ---------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model_type': self.model_type, 'state': self._serialize()}, path)

    @classmethod
    def load(cls, path: str | Path) -> 'IDSModel':
        payload = joblib.load(path)
        model = cls(model_type=payload['model_type'])
        model._deserialize(payload['state'])
        return model

    def _serialize(self):
        if self.model_type == 'rf':
            return joblib.dumps(self.model)
        else:
            return {'state_dict': self.model.state_dict()}

    def _deserialize(self, state):
        if self.model_type == 'rf':
            self.model = joblib.loads(state)
        else:
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()

    # ---------------
    # Training / Prediction
    # ---------------
    def fit(self, X, y):
        if self.model_type == 'rf':
            self.model.fit(X, y)
        else:
            # Torch training loop (binary classification)
            X = _to_tensor(X).float()
            y = _to_tensor(y).float().view(-1, 1)
            if self.model.input_dim is None:
                self.model.reset(input_dim=X.shape[1], hidden=self.dnn_hidden, layers=self.dnn_layers)
                self._optimizer = optim.Adam(self.model.parameters(), lr=self.dnn_lr)
            self.model.train()
            ds = torch.utils.data.TensorDataset(X, y)
            dl = torch.utils.data.DataLoader(ds, batch_size=self.dnn_batch_size, shuffle=True)
            for epoch in range(self.dnn_epochs):
                epoch_loss = 0.0
                for xb, yb in dl:
                    self._optimizer.zero_grad()
                    logits = self.model(xb)
                    loss = self._criterion(logits, yb)
                    loss.backward()
                    self._optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
            return self

    def predict(self, X):
        if self.model_type == 'rf':
            return self.model.predict(X)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(_to_tensor(X).float())
                probs = torch.sigmoid(logits)
                return (probs.numpy().ravel() >= 0.5).astype(int)

    def predict_proba(self, X):
        if self.model_type == 'rf':
            # Ensure 2-column probabilities for binary classification
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            # Fallback: derive from decision function if needed
            scores = self.model.predict(X)
            return np.vstack([1 - scores, scores]).T
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(_to_tensor(X).float())
                probs = torch.sigmoid(logits).numpy().ravel()
                return np.vstack([1 - probs, probs]).T


class _SimpleDNN(nn.Module):
    def __init__(self, input_dim: Optional[int], hidden: int = 64, layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        if input_dim is not None:
            self._build()

    def reset(self, input_dim: int, hidden: int, layers: int):
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        self._build()

    def _build(self):
        dims = [self.input_dim] + [self.hidden] * self.layers + [1]
        mods = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(0.2)]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def _to_tensor(x):
    if torch is None:
        raise ImportError('PyTorch is required for DNN operations')
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    try:
        return torch.from_numpy(np.asarray(x))
    except Exception:
        return torch.tensor(x)
