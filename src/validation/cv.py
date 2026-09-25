"""
Cross-Validation Module for WIUT FinTech Hackathon
Implements stratified K-Fold and time-aware validation with leakage prevention
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationStrategy:
    """Base class for validation strategies"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.validation_config = self.config.get_validation_config()
        self.folds = []
    
    def create_folds(self, data: pd.DataFrame, temporal_col: str = None) -> List[Dict[str, Any]]:
        """Create validation folds"""
        raise NotImplementedError("Subclasses must implement create_folds")
    
    def save_folds(self, output_path: str = "artifacts/validation_folds.json"):
        """Save fold definitions to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fold_data = {
            'strategy': self.__class__.__name__,
            'n_folds': len(self.folds),
            'folds': self.folds
        }
        
        with open(output_file, 'w') as f:
            json.dump(fold_data, f, indent=2, default=str)
        
        logger.info(f"Folds saved to {output_path}")
    
    def load_folds(self, input_path: str = "artifacts/validation_folds.json") -> List[Dict[str, Any]]:
        """Load fold definitions from file"""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Folds file not found: {input_path}")
        
        with open(input_file, 'r') as f:
            fold_data = json.load(f)
        
        self.folds = fold_data['folds']
        logger.info(f"Loaded {len(self.folds)} folds from {input_path}")
        return self.folds


class StratifiedKFoldValidation(ValidationStrategy):
    """Stratified K-Fold Cross-Validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self.n_splits = self.validation_config.get('n_splits', 5)
        self.random_state = self.config.get_random_seed()
    
    def create_folds(self, data: pd.DataFrame, temporal_col: str = None) -> List[Dict[str, Any]]:
        """Create stratified K-Fold folds"""
        logger.info(f"Creating {self.n_splits}-fold stratified CV splits...")
        
        target_col = self.data_config['target_column']
        signal_id_col = self.data_config['signal_id_column']
        
        # Stratification requires at least one example of each class per fold.
        class_counts = data[target_col].value_counts(dropna=False)
        n_splits = min(self.n_splits, int(class_counts.min()))
        if n_splits < 2:
            raise ValueError("Stratified validation requires at least two examples in every target class")
        if n_splits != self.n_splits:
            logger.warning("Reducing stratified folds from %s to %s due to minority class size", self.n_splits, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Create folds
        self.folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data, data[target_col])):
            fold = {
                'fold': fold_idx,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'train_signal_ids': data.iloc[train_idx][signal_id_col].tolist(),
                'val_signal_ids': data.iloc[val_idx][signal_id_col].tolist(),
                'train_target_distribution': {
                    'positive': int(data.iloc[train_idx][target_col].sum()),
                    'negative': int(len(train_idx) - data.iloc[train_idx][target_col].sum())
                },
                'val_target_distribution': {
                    'positive': int(data.iloc[val_idx][target_col].sum()),
                    'negative': int(len(val_idx) - data.iloc[val_idx][target_col].sum())
                }
            }
            self.folds.append(fold)
        
        logger.info(f"Created {len(self.folds)} stratified folds")
        return self.folds


class TimeAwareValidation(ValidationStrategy):
    """Time-Aware Validation using older signals for training, newer for validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self.n_splits = self.validation_config.get('n_splits', 5)
        self.random_state = self.config.get_random_seed()
        self.date_col = self.data_config['signal_date_column']
    
    def create_folds(self, data: pd.DataFrame, temporal_col: str = None) -> List[Dict[str, Any]]:
        """Create time-aware folds based on signal dates"""
        logger.info(f"Creating {self.n_splits}-fold time-aware CV splits...")
        
        if temporal_col is None:
            temporal_col = self.date_col
        
        target_col = self.data_config['target_column']
        signal_id_col = self.data_config['signal_id_column']
        
        # Ensure date column is datetime
        data[temporal_col] = pd.to_datetime(data[temporal_col])
        
        # Sort by date for temporal splitting
        data_sorted = data.sort_values(temporal_col).reset_index(drop=True)
        
        # Expanding-window splits: every validation block follows its training history.
        self.folds = []
        n_samples = len(data_sorted)
        if n_samples < 3:
            raise ValueError("Time-aware validation requires at least three observations")
        effective_splits = min(self.n_splits, n_samples - 1)
        validation = np.array_split(np.arange(1, n_samples), effective_splits)

        for fold_idx, val_block in enumerate(validation):
            if len(val_block) == 0:
                continue
            val_start = int(val_block[0])
            val_end = int(val_block[-1]) + 1
            val_indices = data_sorted.iloc[val_start:val_end].index.tolist()
            train_indices = data_sorted.iloc[:val_start].index.tolist()
            
            fold = {
                'fold': fold_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'train_signal_ids': data_sorted.iloc[train_indices][signal_id_col].tolist(),
                'val_signal_ids': data_sorted.iloc[val_indices][signal_id_col].tolist(),
                'train_date_range': {
                    'min': str(data_sorted.iloc[train_indices][temporal_col].min()),
                    'max': str(data_sorted.iloc[train_indices][temporal_col].max())
                },
                'val_date_range': {
                    'min': str(data_sorted.iloc[val_indices][temporal_col].min()),
                    'max': str(data_sorted.iloc[val_indices][temporal_col].max())
                },
                'train_target_distribution': {
                    'positive': int(data_sorted.iloc[train_indices][target_col].sum()),
                    'negative': int(len(train_indices) - data_sorted.iloc[train_indices][target_col].sum())
                },
                'val_target_distribution': {
                    'positive': int(data_sorted.iloc[val_indices][target_col].sum()),
                    'negative': int(len(val_indices) - data_sorted.iloc[val_indices][target_col].sum())
                }
            }
            self.folds.append(fold)
        
        logger.info(f"Created {len(self.folds)} time-aware folds")
        return self.folds


class ValidationEvaluator:
    """Evaluate models using different validation strategies"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.random_state = self.config.get_random_seed()
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                     folds: List[Dict[str, Any]], feature_transformer=None) -> Dict[str, Any]:
        """Evaluate model using specified folds
        
        Args:
            model: The model to evaluate
            X: Feature matrix
            y: Target vector
            folds: List of fold definitions with train/val indices
            feature_transformer: Optional transformer that must be fit inside training fold
                                to prevent data leakage
        """
        logger.info("Evaluating model with specified folds...")
        
        fold_scores = []
        fold_predictions = []
        
        for fold in folds:
            train_idx = fold['train_indices']
            val_idx = fold['val_indices']
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply feature transformations inside training fold if provided
            # This ensures no validation target influences feature construction
            if feature_transformer is not None:
                # Fit transformer on training data only
                feature_transformer.fit(X_train, y_train)
                # Transform both training and validation data
                X_train = feature_transformer.transform(X_train)
                X_val = feature_transformer.transform(X_val)
                logger.info(f"Fold {fold['fold']}: Feature transformations fit inside training fold")
            
            # Fit model on training data
            model.fit(X_train, y_train)
            
            # Predict on validation data
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_pred_proba = model.predict(X_val)
            
            # Calculate ROC-AUC
            fold_auc = roc_auc_score(y_val, y_pred_proba)
            fold_scores.append(fold_auc)
            
            fold_predictions.append({
                'fold': fold['fold'],
                'val_indices': val_idx,
                'y_true': y_val.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'fold_auc': fold_auc
            })
        
        results = {
            'mean_roc_auc': np.mean(fold_scores),
            'std_roc_auc': np.std(fold_scores),
            'per_fold_roc_auc': fold_scores,
            'fold_predictions': fold_predictions
        }
        
        logger.info(f"Validation complete. Mean ROC-AUC: {results['mean_roc_auc']:.4f} (+/- {results['std_roc_auc']:.4f})")
        
        return results
    
    def compare_validation_strategies(self, model, X: pd.DataFrame, y: pd.Series,
                                    data: pd.DataFrame, temporal_col: str = None, 
                                    feature_transformer=None) -> Dict[str, Any]:
        """Compare stratified vs time-aware validation
        
        Args:
            model: The model to evaluate
            X: Feature matrix
            y: Target vector
            data: Full dataset with temporal information
            temporal_col: Column name for temporal splitting
            feature_transformer: Optional transformer that must be fit inside training fold
        """
        logger.info("Comparing validation strategies...")
        
        # Create stratified folds
        stratified_cv = StratifiedKFoldValidation()
        stratified_folds = stratified_cv.create_folds(data, temporal_col)
        stratified_results = self.evaluate_model(model, X, y, stratified_folds, feature_transformer)
        
        # Create time-aware folds
        time_aware_cv = TimeAwareValidation()
        time_aware_folds = time_aware_cv.create_folds(data, temporal_col)
        time_aware_results = self.evaluate_model(model, X, y, time_aware_folds, feature_transformer)
        
        comparison = {
            'stratified_kfold': stratified_results,
            'time_aware': time_aware_results,
            'difference': {
                'mean_roc_auc': time_aware_results['mean_roc_auc'] - stratified_results['mean_roc_auc'],
                'std_roc_auc': time_aware_results['std_roc_auc'] - stratified_results['std_roc_auc']
            }
        }
        
        logger.info(f"Stratified K-Fold: {stratified_results['mean_roc_auc']:.4f} (+/- {stratified_results['std_roc_auc']:.4f})")
        logger.info(f"Time-Aware: {time_aware_results['mean_roc_auc']:.4f} (+/- {time_aware_results['std_roc_auc']:.4f})")
        logger.info(f"Difference: {comparison['difference']['mean_roc_auc']:+.4f}")
        
        return comparison


def create_validation_plan(data: pd.DataFrame, config_path: str = "config.yaml",
                         output_path: str = "artifacts/validation_plan.json") -> Dict[str, Any]:
    """Create comprehensive validation plan"""
    logger.info("Creating validation plan...")
    
    config = get_config(config_path)
    validation_config = config.get_validation_config()
    
    plan = {
        'validation_strategy': validation_config.get('strategy', 'time_aware'),
        'n_splits': validation_config.get('n_splits', 5),
        'random_seed': config.get_random_seed(),
        'temporal_column': config.get_data_config().get('signal_date_column'),
        'target_column': config.get_data_config().get('target_column'),
        'signal_id_column': config.get_data_config().get('signal_id_column'),
        'data_info': {
            'total_samples': len(data),
            'target_prevalence': float(data[config.get_data_config()['target_column']].mean()),
            'date_range': {
                'min': str(pd.to_datetime(data[config.get_data_config()['signal_date_column']]).min()),
                'max': str(pd.to_datetime(data[config.get_data_config()['signal_date_column']]).max())
            }
        },
        'strategies': {
            'stratified_kfold': {
                'description': 'Stratified K-Fold Cross-Validation',
                'advantages': ['Maintains target distribution', 'Uses all data for training', 'Standard approach'],
                'disadvantages': ['May leak temporal information', 'Not time-aware'],
                'when_to_use': 'When temporal leakage is not a concern'
            },
            'time_aware': {
                'description': 'Time-Aware Validation (train on past, validate on future)',
                'advantages': ['Prevents temporal leakage', 'More realistic evaluation', 'Time-aware'],
                'disadvantages': ['May have distribution shift', 'Less data for training'],
                'when_to_use': 'When temporal leakage is a concern'
            }
        },
        'recommended_strategy': 'time_aware',  # Default recommendation
        'reasoning': 'Time-aware validation is recommended due to temporal overlap in signal dates and to prevent leakage from future information.'
    }
    
    # Save plan
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    
    logger.info(f"Validation plan saved to {output_path}")
    
    return plan


def main():
    """Main function to create validation framework"""
    # This would typically be called with actual data
    logger.info("Validation framework initialized")
    logger.info("Use notebooks/03_validation_design.ipynb for comprehensive validation design")
    
    return {
        'status': 'Validation framework ready',
        'available_strategies': ['StratifiedKFoldValidation', 'TimeAwareValidation'],
        'evaluator': 'ValidationEvaluator'
    }


if __name__ == "__main__":
    main()
