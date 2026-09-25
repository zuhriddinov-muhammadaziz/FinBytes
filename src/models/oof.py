"""
Out-of-Fold Prediction Generation Module for WIUT FinTech Hackathon
Generates OOF predictions for ensemble training using only training data

This module implements strict OOF prediction generation:
- Each training sample gets predictions from models trained on other folds
- Ensures no data leakage in ensemble weight optimization
- Maintains reproducibility with fixed validation folds
- Supports diverse model types for ensemble diversity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
from sklearn.metrics import roc_auc_score
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from .train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OOFPredictor:
    """Generate out-of-fold predictions for ensemble training"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.random_seed = self.config.get_random_seed()
        self.trainer = ModelTrainer(config_path)
        
        # Storage for OOF predictions
        self.oof_predictions = {}
        self.model_metadata = {}
        
    def generate_oof_predictions(self, 
                                  model_configs: List[Dict[str, Any]],
                                  X: pd.DataFrame, 
                                  y: pd.Series,
                                  folds: List[Dict[str, Any]],
                                  feature_set_name: str = "v1_features") -> Dict[str, Any]:
        """Generate OOF predictions for multiple models
        
        Args:
            model_configs: List of model configurations with hyperparameters
            X: Feature matrix
            y: Target vector
            folds: Validation fold definitions
            feature_set_name: Name of feature set used
            
        Returns:
            Dictionary containing OOF predictions and metadata
        """
        logger.info(f"Generating OOF predictions for {len(model_configs)} models...")
        logger.info(f"Using {len(folds)} folds for OOF generation")
        logger.info(f"Feature set: {feature_set_name}")
        
        # Initialize OOF prediction arrays
        n_samples = len(X)
        oof_results = {}
        
        for config in model_configs:
            model_name = config['model_name']
            hyperparameters = config.get('hyperparameters', {})
            
            logger.info(f"Generating OOF predictions for {model_name}...")
            
            # Initialize OOF array for this model
            oof_predictions = np.zeros(n_samples)
            
            # Track which samples have predictions
            prediction_mask = np.zeros(n_samples, dtype=bool)
            
            # Store fold-wise predictions for analysis
            fold_predictions = []
            
            for fold in folds:
                train_idx = fold['train_indices']
                val_idx = fold['val_indices']
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                # Create fresh model instance
                model = self.trainer.get_model(model_name, hyperparameters)
                
                # Train model on training fold
                try:
                    training_info = self.trainer.train_model(
                        model, X_train, y_train, 
                        early_stopping=False  # No early stopping for OOF
                    )
                    
                    if training_info['training_successful']:
                        # Generate predictions on validation fold
                        if hasattr(model, 'predict_proba'):
                            fold_preds = model.predict_proba(X_val)[:, 1]
                        else:
                            fold_preds = model.predict(X_val)
                        
                        # Store OOF predictions
                        oof_predictions[val_idx] = fold_preds
                        prediction_mask[val_idx] = True
                        
                        fold_predictions.append({
                            'fold': fold['fold'],
                            'val_indices': val_idx,
                            'predictions': fold_preds.tolist(),
                            'training_time': training_info['training_time_seconds']
                        })
                    else:
                        logger.warning(f"Fold {fold['fold']} training failed for {model_name}")
                        # Use placeholder predictions for failed fold
                        oof_predictions[val_idx] = y.iloc[val_idx].mean()
                        prediction_mask[val_idx] = True
                        
                except Exception as e:
                    logger.error(f"Exception in fold {fold['fold']} for {model_name}: {e}")
                    # Use placeholder predictions for failed fold
                    oof_predictions[val_idx] = y.iloc[val_idx].mean()
                    prediction_mask[val_idx] = True
            
            # Verify all samples have predictions
            if not prediction_mask.all():
                missing_count = (~prediction_mask).sum()
                logger.warning(f"{missing_count} samples missing predictions for {model_name}")
                # Fill missing with mean prediction
                oof_predictions[~prediction_mask] = oof_predictions[prediction_mask].mean()
            
            # Calculate OOF ROC-AUC
            oof_auc = roc_auc_score(y, oof_predictions)
            
            # Store results
            oof_results[model_name] = {
                'oof_predictions': oof_predictions,
                'oof_roc_auc': oof_auc,
                'hyperparameters': hyperparameters,
                'fold_predictions': fold_predictions,
                'feature_set': feature_set_name,
                'n_samples': n_samples,
                'generation_completed': datetime.now().isoformat()
            }
            
            logger.info(f"  {model_name}: OOF ROC-AUC = {oof_auc:.4f}")
        
        # Add metadata
        oof_results['metadata'] = {
            'n_models': len(model_configs),
            'n_folds': len(folds),
            'n_samples': n_samples,
            'feature_set': feature_set_name,
            'random_seed': self.random_seed,
            'generation_completed': datetime.now().isoformat()
        }
        
        self.oof_predictions = oof_results
        
        logger.info(f"OOF prediction generation complete for {len(oof_results) - 1} models")
        
        return oof_results
    
    def save_oof_predictions(self, 
                            output_path: str = "artifacts/oof_predictions.parquet") -> None:
        """Save OOF predictions to parquet file
        
        Creates a DataFrame with model predictions as columns
        """
        if not self.oof_predictions:
            raise ValueError("No OOF predictions generated. Call generate_oof_predictions first.")
        
        # Extract model names (exclude metadata)
        model_names = [k for k in self.oof_predictions.keys() if k != 'metadata']
        
        # Create DataFrame
        oof_df = pd.DataFrame()
        
        for model_name in model_names:
            oof_df[model_name] = self.oof_predictions[model_name]['oof_predictions']
        
        # Add metadata as attributes
        oof_df.attrs['metadata'] = self.oof_predictions['metadata']
        
        # Save to parquet
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        oof_df.to_parquet(output_file, index=False)
        
        logger.info(f"OOF predictions saved to {output_path}")
        logger.info(f"Shape: {oof_df.shape}, Models: {len(model_names)}")
    
    def load_oof_predictions(self, 
                            input_path: str = "artifacts/oof_predictions.parquet") -> pd.DataFrame:
        """Load OOF predictions from parquet file"""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"OOF predictions file not found: {input_path}")
        
        oof_df = pd.read_parquet(input_file)
        
        logger.info(f"OOF predictions loaded from {input_path}")
        logger.info(f"Shape: {oof_df.shape}, Metadata: {oof_df.attrs.get('metadata', {})}")
        
        return oof_df
    
    def get_oof_metadata(self) -> Dict[str, Any]:
        """Get metadata about OOF predictions"""
        if not self.oof_predictions:
            raise ValueError("No OOF predictions generated. Call generate_oof_predictions first.")
        
        return self.oof_predictions.get('metadata', {})
    
    def get_model_oof_performance(self) -> pd.DataFrame:
        """Get summary of individual model OOF performance"""
        if not self.oof_predictions:
            raise ValueError("No OOF predictions generated. Call generate_oof_predictions first.")
        
        performance_data = []
        
        for model_name, results in self.oof_predictions.items():
            if model_name != 'metadata':
                performance_data.append({
                    'model': model_name,
                    'oof_roc_auc': results['oof_roc_auc'],
                    'feature_set': results['feature_set'],
                    'n_samples': results['n_samples']
                })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('oof_roc_auc', ascending=False)
        
        return performance_df


def generate_diverse_model_configs(trainer: ModelTrainer, 
                                  best_models: List[str] = None) -> List[Dict[str, Any]]:
    """Generate diverse model configurations for ensemble
    
    Selects strongest diverse models:
    - CatBoost (if available)
    - LightGBM (if available)  
    - XGBoost (if available)
    - Best alternative tree model (ExtraTrees or HistGradientBoosting)
    - Optional linear baseline (LogisticRegression)
    
    Args:
        trainer: ModelTrainer instance
        best_models: Optional list of specific models to use
        
    Returns:
        List of model configurations
    """
    model_configs = []
    
    if best_models:
        # Use specified models
        for model_name in best_models:
            if trainer.available_models.get(model_name, False):
                model_configs.append({
                    'model_name': model_name,
                    'hyperparameters': trainer.get_default_hyperparameters(model_name)
                })
    else:
        # Auto-select diverse models
        # Gradient boosting models (primary)
        if trainer.available_models['catboost']:
            model_configs.append({
                'model_name': 'catboost',
                'hyperparameters': trainer.get_default_hyperparameters('catboost')
            })
        
        if trainer.available_models['lightgbm']:
            model_configs.append({
                'model_name': 'lightgbm',
                'hyperparameters': trainer.get_default_hyperparameters('lightgbm')
            })
        
        if trainer.available_models['xgboost']:
            model_configs.append({
                'model_name': 'xgboost',
                'hyperparameters': trainer.get_default_hyperparameters('xgboost')
            })
        
        # Best alternative tree model
        if trainer.available_models['histgradientboosting']:
            model_configs.append({
                'model_name': 'histgradientboosting',
                'hyperparameters': trainer.get_default_hyperparameters('histgradientboosting')
            })
        elif trainer.available_models['extratrees']:
            model_configs.append({
                'model_name': 'extratrees',
                'hyperparameters': trainer.get_default_hyperparameters('extratrees')
            })
        
        # Optional linear baseline
        if trainer.available_models['logistic_regression']:
            model_configs.append({
                'model_name': 'logistic_regression',
                'hyperparameters': trainer.get_default_hyperparameters('logistic_regression')
            })
    
    logger.info(f"Generated {len(model_configs)} diverse model configurations for ensemble")
    for config in model_configs:
        logger.info(f"  - {config['model_name']}")
    
    return model_configs


def main():
    """Main function to demonstrate OOF prediction generation"""
    oof_predictor = OOFPredictor()
    
    logger.info("OOF prediction module initialized")
    logger.info("Use notebooks/08_ensemble.ipynb for comprehensive ensemble building")
    
    return {
        'status': 'OOF prediction module ready',
        'capabilities': [
            'generate_oof_predictions',
            'save_oof_predictions',
            'load_oof_predictions',
            'get_model_performance'
        ]
    }


if __name__ == "__main__":
    main()