#!/usr/bin/env python3
"""
Main entry point for the Intrusion Detection System (IDS).

This module provides:
- Training and evaluation of ML models for network intrusion detection
- Live packet sniffing and real-time threat detection
- Support for both CSV and PCAP data formats

Usage:
    Training mode:
        python main.py --mode train --data data/samples/train.csv --model-type rf
    
    Evaluation mode:
        python main.py --mode evaluate --data data/samples/test.csv --model models/ids_model.pkl
    
    Live detection mode:
        python main.py --mode live --interface eth0 --model models/ids_model.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import joblib
import pandas as pd

# Import local modules
from preprocessing import load_data, preprocess_features
from model import IDSModel
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}


def train_mode(args, config):
    """Train the IDS model.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    logger.info("Starting training mode...")
    
    # Load and preprocess data
    logger.info(f"Loading training data from {args.data}")
    X_train, y_train = load_data(args.data, data_type=args.data_type)
    X_train = preprocess_features(X_train)
    
    # Initialize and train model
    model = IDSModel(
        model_type=args.model_type,
        **config.get('model_params', {})
    )
    
    logger.info(f"Training {args.model_type} model...")
    model.fit(X_train, y_train)
    
    # Save trained model
    output_path = args.output or config.get('model_path', 'models/ids_model.pkl')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")
    
    # Evaluate on training data
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)
    
    metrics = evaluate_model(y_train, y_pred, y_proba)
    logger.info(f"Training metrics: {metrics}")


def evaluate_mode(args, config):
    """Evaluate a trained model on test data.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    logger.info("Starting evaluation mode...")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = IDSModel.load(args.model)
    
    # Load and preprocess test data
    logger.info(f"Loading test data from {args.data}")
    X_test, y_test = load_data(args.data, data_type=args.data_type)
    X_test = preprocess_features(X_test)
    
    # Make predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate and display metrics
    metrics = evaluate_model(y_test, y_pred, y_proba)
    logger.info("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate plots if requested
    if args.plot:
        output_dir = Path(args.output) if args.output else Path('results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(y_test, y_pred, output_dir / 'confusion_matrix.png')
        plot_roc_curve(y_test, y_proba, output_dir / 'roc_curve.png')
        logger.info(f"Plots saved to {output_dir}")


def live_mode(args, config):
    """Run live packet sniffing and intrusion detection.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
    """
    logger.info("Starting live detection mode...")
    
    try:
        from scapy.all import sniff
    except ImportError:
        logger.error("Scapy is required for live mode. Install with: pip install scapy")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = IDSModel.load(args.model)
    
    def packet_handler(packet):
        """Process each captured packet.
        
        Args:
            packet: Scapy packet object
        """
        try:
            # Extract features from packet
            # TODO: Implement packet feature extraction
            # features = extract_packet_features(packet)
            # features_df = preprocess_features(features)
            
            # Make prediction
            # prediction = model.predict(features_df)
            # confidence = model.predict_proba(features_df)
            
            # if prediction[0] == 1:  # Attack detected
            #     logger.warning(f"ATTACK DETECTED! Confidence: {confidence[0][1]:.2%}")
            #     logger.warning(f"Packet summary: {packet.summary()}")
            
            pass  # Placeholder for students to implement
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    # Start packet capture
    logger.info(f"Starting packet capture on interface {args.interface}...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        sniff(
            iface=args.interface,
            prn=packet_handler,
            store=False,
            count=args.count if args.count else 0
        )
    except KeyboardInterrupt:
        logger.info("\nStopping packet capture...")
    except Exception as e:
        logger.error(f"Error during packet capture: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Common arguments
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'live'],
        help='Operation mode: train, evaluate, or live'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Training/evaluation arguments
    parser.add_argument(
        '--data',
        type=str,
        help='Path to input data (CSV or PCAP file)'
    )
    parser.add_argument(
        '--data-type',
        type=str,
        default='csv',
        choices=['csv', 'pcap'],
        help='Type of input data'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='rf',
        choices=['rf', 'dnn'],
        help='Model type: rf (Random Forest) or dnn (Deep Neural Network)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to saved model (for evaluate/live modes)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for model or results'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate evaluation plots'
    )
    
    # Live mode arguments
    parser.add_argument(
        '--interface',
        type=str,
        default='eth0',
        help='Network interface for live capture'
    )
    parser.add_argument(
        '--count',
        type=int,
        help='Number of packets to capture (0 for infinite)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Route to appropriate mode
    try:
        if args.mode == 'train':
            if not args.data:
                parser.error("--data is required for train mode")
            train_mode(args, config)
            
        elif args.mode == 'evaluate':
            if not args.data or not args.model:
                parser.error("--data and --model are required for evaluate mode")
            evaluate_mode(args, config)
            
        elif args.mode == 'live':
            if not args.model:
                parser.error("--model is required for live mode")
            live_mode(args, config)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
