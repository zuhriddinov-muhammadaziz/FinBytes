"""
Comprehensive Model Benchmarking Script for WIUT FinTech Hackathon
Benchmarks strong tabular models with best leakage-safe feature sets

This script implements a complete benchmarking pipeline following competition rules:
1. Use identical validation folds for fair comparisons
2. Track training time
3. Record exact feature set and hyperparameters
4. Use early stopping where appropriate
5. Do not tune on the test set
6. Keep a reproducible experiment table

Models benchmarked:
- CatBoost (if available)
- LightGBM (if available)
- XGBoost (if available)
- HistGradientBoosting
- Logistic Regression (interpretable baseline)
- ExtraTrees

Hyperparameter search covers:
- depth / leaves
- learning rate
- number of estimators
- subsampling
- column sampling
- regularization
- class weighting (only if empirically helps ROC-AUC)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.features.basic_features import BasicFeatures
from src.features.window_features import WindowFeatures
from src.validation.cv import TimeAwareValidation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_comprehensive_model_configs(trainer: ModelTrainer) -> list:
    """Create comprehensive model configurations for benchmarking
    
    Returns default hyperparameters for all available models
    """
    model_configs = []
    
    # Logistic Regression (interpretable baseline)
    if trainer.available_models['logistic_regression']:
        model_configs.append({
            'model_name': 'logistic_regression',
            'hyperparameters': trainer.get_default_hyperparameters('logistic_regression'),
            'description': 'Interpretable baseline model'
        })
    
    # Random Forest
    if trainer.available_models['randomforest']:
        model_configs.append({
            'model_name': 'randomforest',
            'hyperparameters': trainer.get_default_hyperparameters('randomforest'),
            'description': 'Random forest baseline'
        })
    
    # Extra Trees
    if trainer.available_models['extratrees']:
        model_configs.append({
            'model_name': 'extratrees',
            'hyperparameters': trainer.get_default_hyperparameters('extratrees'),
            'description': 'Extra trees ensemble'
        })
    
    # HistGradientBoosting
    if trainer.available_models['histgradientboosting']:
        model_configs.append({
            'model_name': 'histgradientboosting',
            'hyperparameters': trainer.get_default_hyperparameters('histgradientboosting'),
            'description': 'Sklearn histogram-based gradient boosting'
        })
    
    # LightGBM
    if trainer.available_models['lightgbm']:
        model_configs.append({
            'model_name': 'lightgbm',
            'hyperparameters': trainer.get_default_hyperparameters('lightgbm'),
            'description': 'LightGBM gradient boosting'
        })
    
    # CatBoost
    if trainer.available_models['catboost']:
        model_configs.append({
            'model_name': 'catboost',
            'hyperparameters': trainer.get_default_hyperparameters('catboost'),
            'description': 'CatBoost gradient boosting'
        })
    
    # XGBoost
    if trainer.available_models['xgboost']:
        model_configs.append({
            'model_name': 'xgboost',
            'hyperparameters': trainer.get_default_hyperparameters('xgboost'),
            'description': 'XGBoost gradient boosting'
        })
    
    return model_configs


def perform_targeted_hyperparameter_search(trainer: ModelTrainer, model_name: str,
                                          X_train: pd.DataFrame, y_train: pd.Series,
                                          X_val: pd.DataFrame, y_val: pd.Series,
                                          n_trials: int = 27) -> dict:
    """Perform small, targeted hyperparameter search
    
    Covers key hyperparameters:
    - depth / leaves
    - learning rate
    - number of estimators
    - subsampling
    - column sampling
    - regularization
    - class weighting (only if empirically helps)
    
    Args:
        trainer: ModelTrainer instance
        model_name: Name of model to tune
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of hyperparameter combinations to try
    
    Returns:
        Search summary with best parameters
    """
    logger.info(f"Performing targeted hyperparameter search for {model_name} ({n_trials} trials)...")
    
    search_summary = trainer.perform_grid_search(
        model_name, X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    
    return search_summary


def main():
    """Main benchmarking pipeline"""
    logger.info("=" * 80)
    logger.info("WIUT FINTECH HACKATHON - COMPREHENSIVE MODEL BENCHMARKING")
    logger.info("=" * 80)
    
    # Load configuration
    config = get_config()
    logger.info(f"Project: {config.get('project', 'name')}")
    logger.info(f"Random Seed: {config.get_random_seed()}")
    
    # Initialize components
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    basic_features = BasicFeatures()
    window_features = WindowFeatures()
    
    logger.info(f"Available models: {list(trainer.available_models.keys())}")
    
    # Load data
    logger.info("Loading data...")
    data = basic_features.load_data()
    logger.info(f"Train signals: {len(data['train_signals'])}")
    logger.info(f"Test signals: {len(data['test_signals'])}")
    
    # Create feature sets
    logger.info("Creating feature sets...")
    
    # V1 features (baseline)
    v1_features = basic_features.create_features(data, temporal_filter=True)
    X_train_v1, y_train, X_test_v1 = basic_features.prepare_modeling_data(v1_features)
    logger.info(f"V1 features: {X_train_v1.shape[1]} features")
    
    # V2 features (multi-window)
    v2_features = window_features.create_window_features(data, temporal_filter=True)
    X_train_v2, _, X_test_v2 = window_features.prepare_modeling_data(v2_features)
    logger.info(f"V2 features: {X_train_v2.shape[1]} features")
    
    # Create fixed validation folds
    logger.info("Creating fixed time-aware validation folds...")
    time_aware_cv = TimeAwareValidation()
    fixed_folds = time_aware_cv.create_folds(data['train_signals'])
    time_aware_cv.save_folds('artifacts/fixed_model_benchmark_folds.json')
    logger.info(f"Created {len(fixed_folds)} fixed validation folds")
    
    # Create train/validation split for early stopping
    last_fold = fixed_folds[-1]
    train_idx = last_fold['train_indices']
    val_idx = last_fold['val_indices']
    
    X_train_v1_es, X_val_v1 = X_train_v1.iloc[train_idx], X_train_v1.iloc[val_idx]
    y_train_es, y_val_es = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    X_train_v2_es, X_val_v2 = X_train_v2.iloc[train_idx], X_train_v2.iloc[val_idx]
    
    # Create model configurations
    logger.info("Creating model configurations...")
    model_configs = create_comprehensive_model_configs(trainer)
    logger.info(f"Configured {len(model_configs)} models for benchmarking")
    
    # Benchmark V1 features
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARKING WITH V1 FEATURES (BASELINE)")
    logger.info("=" * 80)
    
    v1_benchmark = evaluator.benchmark_model_set(
        model_configs, X_train_v1_es, y_train_es, X_val_v1, y_val_es, fixed_folds,
        feature_set_name="v1_features"
    )
    
    evaluator.save_benchmark_results(v1_benchmark, 'experiments/v1_model_results.json')
    evaluator.create_results_csv(v1_benchmark, 'experiments/v1_model_results.csv')
    
    logger.info(f"V1 Best model: {v1_benchmark['best_model']} (ROC-AUC: {v1_benchmark['best_score']:.4f})")
    
    # Benchmark V2 features
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARKING WITH V2 FEATURES (MULTI-WINDOW)")
    logger.info("=" * 80)
    
    v2_benchmark = evaluator.benchmark_model_set(
        model_configs, X_train_v2_es, y_train_es, X_val_v2, y_val_es, fixed_folds,
        feature_set_name="v2_features"
    )
    
    evaluator.save_benchmark_results(v2_benchmark, 'experiments/v2_model_results.json')
    evaluator.create_results_csv(v2_benchmark, 'experiments/v2_model_results.csv')
    
    logger.info(f"V2 Best model: {v2_benchmark['best_model']} (ROC-AUC: {v2_benchmark['best_score']:.4f})")
    
    # Determine best overall configuration
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL COMPARISON")
    logger.info("=" * 80)
    
    if v2_benchmark['best_score'] > v1_benchmark['best_score']:
        best_model_name = v2_benchmark['best_model']
        best_feature_set = "v2"
        best_score = v2_benchmark['best_score']
        logger.info(f"Best configuration: {best_model_name} with V2 features (ROC-AUC: {best_score:.4f})")
    else:
        best_model_name = v1_benchmark['best_model']
        best_feature_set = "v1"
        best_score = v1_benchmark['best_score']
        logger.info(f"Best configuration: {best_model_name} with V1 features (ROC-AUC: {best_score:.4f})")
    
    # Perform targeted hyperparameter search for best model
    logger.info("\n" + "=" * 80)
    logger.info(f"TARGETED HYPERPARAMETER SEARCH FOR {best_model_name}")
    logger.info("=" * 80)
    
    if best_feature_set == "v2":
        X_train_best = X_train_v2_es
        X_val_best = X_val_v2
    else:
        X_train_best = X_train_v1_es
        X_val_best = X_val_v1
    
    search_results = perform_targeted_hyperparameter_search(
        trainer, best_model_name, X_train_best, y_train_es, X_val_best, y_val_es, n_trials=27
    )
    
    # Save hyperparameter search results
    with open('experiments/hyperparameter_search_results.json', 'w') as f:
        json.dump(search_results, f, indent=2, default=str)
    
    logger.info(f"Best hyperparameters: {search_results['best_params']}")
    logger.info(f"Best search score: {search_results['best_score']:.4f}")
    
    # Create final summary
    final_summary = {
        'benchmark_completed': datetime.now().isoformat(),
        'v1_best_model': v1_benchmark['best_model'],
        'v1_best_score': v1_benchmark['best_score'],
        'v2_best_model': v2_benchmark['best_model'],
        'v2_best_score': v2_benchmark['best_score'],
        'overall_best_model': best_model_name,
        'overall_best_feature_set': best_feature_set,
        'overall_best_score': best_score,
        'hyperparameter_search': search_results,
        'models_benchmarked': len(model_configs),
        'feature_sets_tested': ['v1', 'v2'],
        'validation_folds': len(fixed_folds),
        'random_seed': config.get_random_seed()
    }
    
    with open('experiments/final_benchmark_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARKING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final results saved to experiments/final_benchmark_summary.json")
    
    return final_summary


if __name__ == "__main__":
    main()