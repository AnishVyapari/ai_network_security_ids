#!/usr/bin/env python3
"""
End-to-end training script for IDS models.
"""
from pathlib import Path
import argparse
import yaml
import pandas as pd

from src.ids.preprocessing import load_data, preprocess_features
from src.ids.model import IDSModel
from src.ids.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train_csv = cfg['paths']['train_csv']
    model_cfg = cfg['model']

    X, y = load_data(train_csv, data_type='csv')
    Xp = preprocess_features(X)

    model = IDSModel(model_type=model_cfg.get('type', 'rf'),
                     n_estimators=model_cfg.get('n_estimators', 200),
                     max_depth=model_cfg.get('max_depth'),
                     random_state=model_cfg.get('random_state', 42),
                     dnn_hidden=model_cfg.get('dnn_hidden', 64),
                     dnn_layers=model_cfg.get('dnn_layers', 2),
                     dnn_lr=model_cfg.get('dnn_lr', 1e-3),
                     dnn_epochs=model_cfg.get('dnn_epochs', 10),
                     dnn_batch_size=model_cfg.get('dnn_batch_size', 64))
    model.fit(Xp, y)

    out_path = cfg['paths'].get('model_path', 'models/ids_model.pkl')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    preds = model.predict(Xp)
    proba = model.predict_proba(Xp)
    metrics = evaluate_model(y, preds, proba)
    print('Training metrics:', metrics)


if __name__ == '__main__':
    main()
