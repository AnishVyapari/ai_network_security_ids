"""
Preprocessing utilities for the IDS project.

This module provides:
- Data loaders for CSV and PCAP
- Cleaning, encoding, imputation, scaling
- Feature engineering helpers

Designed for easy extension by students/researchers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    from scapy.all import rdpcap
except Exception:
    rdpcap = None  # Optional dependency for PCAP


# -----------------------------
# Data Loading
# -----------------------------

def load_data(path: str | Path, data_type: str = 'csv') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load dataset from CSV or PCAP.

    Args:
        path: Path to the input file
        data_type: 'csv' or 'pcap'

    Returns:
        X: feature DataFrame
        y: label Series if present (for CSV), otherwise None

    Notes:
        - For CSV: expects an optional 'label' column.
        - For PCAP: returns packet-level features without labels.
    """
    path = Path(path)
    if data_type == 'csv':
        df = pd.read_csv(path)
        y = df['label'] if 'label' in df.columns else None
        X = df.drop(columns=['label']) if 'label' in df.columns else df
        return X, y

    elif data_type == 'pcap':
        if rdpcap is None:
            raise ImportError("scapy is required to parse PCAPs. pip install scapy")
        packets = rdpcap(str(path))
        X = extract_pcap_features(packets)
        return X, None

    else:
        raise ValueError("data_type must be 'csv' or 'pcap'")


# -----------------------------
# Feature Engineering
# -----------------------------

def extract_pcap_features(packets) -> pd.DataFrame:
    """Extract basic flow/packet features from scapy packets.

    This is a simple starter; extend as needed.
    """
    records = []
    for pkt in packets:
        rec = {
            'length': int(len(pkt)),
            'has_tcp': int(pkt.haslayer('TCP')),
            'has_udp': int(pkt.haslayer('UDP')),
            'has_icmp': int(pkt.haslayer('ICMP')),
            'time': float(getattr(pkt, 'time', 0.0)),
        }
        # Example: add more protocol-specific fields here
        records.append(rec)
    df = pd.DataFrame(records)
    # Derive simple temporal deltas
    if not df.empty and 'time' in df:
        df = df.sort_values('time')
        df['delta_time'] = df['time'].diff().fillna(0.0)
    return df.drop(columns=['time'], errors='ignore')


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply standard preprocessing: impute, encode categoricals, scale numerics.

    Returns a numeric feature matrix (numpy array) suitable for ML models.
    """
    X = X.copy()

    # Identify column types
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipeline = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    categorical_pipeline = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols) if len(num_cols) == 0 else ('num',
             make_numeric_pipeline(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols) if len(cat_cols) == 0 else (
             'cat', make_categorical_pipeline(), cat_cols),
        ],
        remainder='drop'
    )

    X_processed = preprocessor.fit_transform(X)
    # Return as DataFrame if possible
    try:
        feature_names = []
        if num_cols:
            feature_names += num_cols
        if cat_cols:
            ohe = preprocessor.named_transformers_['cat']['ohe']
            feature_names += list(ohe.get_feature_names_out(cat_cols))
        return pd.DataFrame(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed,
                            columns=feature_names)
    except Exception:
        return pd.DataFrame(X_processed)


def make_numeric_pipeline():
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])


def make_categorical_pipeline():
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ])
