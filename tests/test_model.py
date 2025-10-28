"""
Pytest tests for IDS model.
"""
import pytest
import numpy as np
import pandas as pd
from src.ids.model import IDSModel
from src.ids.preprocessing import preprocess_features


@pytest.fixture
def sample_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
    })
    y = np.random.randint(0, 2, 100)
    return X, y


def test_rf_model(sample_data, tmp_path):
    """Test Random Forest model training and prediction."""
    X, y = sample_data
    Xp = preprocess_features(X)
    
    model = IDSModel(model_type='rf', n_estimators=10, random_state=42)
    model.fit(Xp, y)
    
    preds = model.predict(Xp)
    assert len(preds) == len(y)
    assert preds.min() >= 0 and preds.max() <= 1
    
    # Test save/load
    model_path = tmp_path / 'model.pkl'
    model.save(model_path)
    loaded_model = IDSModel.load(model_path)
    
    preds_loaded = loaded_model.predict(Xp)
    np.testing.assert_array_equal(preds, preds_loaded)


def test_model_predict_proba(sample_data):
    """Test probability predictions."""
    X, y = sample_data
    Xp = preprocess_features(X)
    
    model = IDSModel(model_type='rf', n_estimators=10, random_state=42)
    model.fit(Xp, y)
    
    proba = model.predict_proba(Xp)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
