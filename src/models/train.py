"""
Model Training Module for WIUT FinTech Hackathon
Model training with hyperparameter search and reproducibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

# Import lightgbm at module level for early stopping
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training with hyperparameter search and reproducibility"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.random_seed = self.config.get_random_seed()
        
        # Available models
        self.available_models = self._check_available_models()
        
        logger.info(f"Available models: {list(self.available_models.keys())}")
    
    def _check_available_models(self) -> Dict[str, bool]:
        """Check which gradient boosting libraries are available"""
        available = {}
        
        # CatBoost
        try:
            import catboost as cb
            available['catboost'] = True
            logger.info("CatBoost available")
        except ImportError:
            available['catboost'] = False
            logger.info("CatBoost not available")
        
        # LightGBM
        try:
            import lightgbm as lgb
            available['lightgbm'] = True
            logger.info("LightGBM available")
        except ImportError:
            available['lightgbm'] = False
            logger.info("LightGBM not available")
        
        # XGBoost
        try:
            import xgboost as xgb
            available['xgboost'] = True
            logger.info("XGBoost available")
        except ImportError:
            available['xgboost'] = False
            logger.info("XGBoost not available")
        
        # HistGradientBoosting (sklearn)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            available['histgradientboosting'] = True
            logger.info("HistGradientBoosting available")
        except ImportError:
            available['histgradientboosting'] = False
            logger.info("HistGradientBoosting not available")
        
        # Always available models
        available['logistic_regression'] = True
        available['extratrees'] = True
        available['randomforest'] = True
        
        return available
    
    def get_model(self, model_name: str, hyperparameters: Dict[str, Any] = None):
        """Get model instance with specified hyperparameters"""
        if hyperparameters is None:
            hyperparameters = {}
        
        if model_name == 'catboost' and self.available_models['catboost']:
            import catboost as cb
            return cb.CatBoostClassifier(
                random_seed=self.random_seed,
                verbose=False,
                **hyperparameters
            )
        
        elif model_name == 'lightgbm' and self.available_models['lightgbm']:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                random_state=self.random_seed,
                verbose=-1,
                **hyperparameters
            )
        
        elif model_name == 'xgboost' and self.available_models['xgboost']:
            import xgboost as xgb
            return xgb.XGBClassifier(
                random_state=self.random_seed,
                verbosity=0,
                **hyperparameters
            )
        
        elif model_name == 'histgradientboosting' and self.available_models['histgradientboosting']:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                random_state=self.random_seed,
                **hyperparameters
            )
        
        elif model_name == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000,
                **hyperparameters
            )
        
        elif model_name == 'extratrees':
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                random_state=self.random_seed,
                n_jobs=-1,
                **hyperparameters
            )
        
        elif model_name == 'randomforest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                random_state=self.random_seed,
                n_jobs=-1,
                **hyperparameters
            )
        
        else:
            raise ValueError(f"Model {model_name} not available or not recognized")
    
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   early_stopping: bool = False, early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """Train model with optional early stopping and time tracking"""
        
        start_time = time.time()
        
        training_info = {
            'model_name': model.__class__.__name__,
            'hyperparameters': model.get_params(),
            'training_started': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        try:
            # Train with early stopping if supported and validation data provided
            if early_stopping and X_val is not None and y_val is not None:
                if self.available_models.get('lightgbm') and 'LGBM' in model.__class__.__name__ and LIGHTGBM_AVAILABLE:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                    )
                elif self.available_models.get('xgboost') and 'XGB' in model.__class__.__name__:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                elif self.available_models.get('catboost') and 'CatBoost' in model.__class__.__name__:
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                else:
                    # Fall back to normal training
                    model.fit(X_train, y_train)
                    training_info['early_stopping_used'] = False
            else:
                model.fit(X_train, y_train)
                training_info['early_stopping_used'] = False
            
            training_time = time.time() - start_time
            training_info['training_time_seconds'] = training_time
            training_info['training_completed'] = datetime.now().isoformat()
            training_info['training_successful'] = True
            
            logger.info(f"Model trained successfully in {training_time:.2f} seconds")
            
        except Exception as e:
            training_time = time.time() - start_time
            training_info['training_time_seconds'] = training_time
            training_info['training_completed'] = datetime.now().isoformat()
            training_info['training_successful'] = False
            training_info['error'] = str(e)
            
            logger.error(f"Model training failed: {e}")
        
        return training_info
    
    def get_default_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get default hyperparameters for each model type"""
        
        defaults = {
            'catboost': {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'subsample': 0.8,
                'colsample_bylevel': 0.8
            },
            'lightgbm': {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'xgboost': {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'histgradientboosting': {
                'max_iter': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'l2_regularization': 0.1
            },
            'logistic_regression': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs'
            },
            'extratrees': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'randomforest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        }
        
        return defaults.get(model_name, {})
    
    def get_hyperparameter_search_space(self, model_name: str) -> Dict[str, List]:
        """Get hyperparameter search space for each model type
        
        Covers required parameters:
        - depth / leaves
        - learning rate
        - number of estimators
        - subsampling
        - column sampling
        - regularization
        - class weighting (optional, for empirical testing)
        """
        
        search_spaces = {
            'catboost': {
                'depth': [4, 6, 8],  # Tree depth
                'learning_rate': [0.05, 0.1, 0.15],  # Learning rate
                'iterations': [500, 1000, 2000],  # Number of estimators
                'l2_leaf_reg': [1, 3, 5],  # Regularization
                'subsample': [0.6, 0.8, 1.0],  # Subsampling
                'colsample_bylevel': [0.6, 0.8, 1.0],  # Column sampling
                'auto_class_weights': [None, 'Balanced']  # Class weighting (optional)
            },
            'lightgbm': {
                'max_depth': [4, 6, 8],  # Tree depth
                'num_leaves': [31, 63, 127],  # Number of leaves
                'learning_rate': [0.05, 0.1, 0.15],  # Learning rate
                'n_estimators': [500, 1000, 2000],  # Number of estimators
                'reg_alpha': [0.0, 0.1, 0.2],  # L1 regularization
                'reg_lambda': [0.0, 0.1, 0.2],  # L2 regularization
                'subsample': [0.6, 0.8, 1.0],  # Subsampling
                'colsample_bytree': [0.6, 0.8, 1.0],  # Column sampling
                'class_weight': [None, 'balanced']  # Class weighting (optional)
            },
            'xgboost': {
                'max_depth': [4, 6, 8],  # Tree depth
                'learning_rate': [0.05, 0.1, 0.15],  # Learning rate
                'n_estimators': [500, 1000, 2000],  # Number of estimators
                'reg_alpha': [0.0, 0.1, 0.2],  # L1 regularization
                'reg_lambda': [0.0, 0.1, 0.2],  # L2 regularization
                'subsample': [0.6, 0.8, 1.0],  # Subsampling
                'colsample_bytree': [0.6, 0.8, 1.0],  # Column sampling
                'scale_pos_weight': [1, 2, 5]  # Class weighting (optional)
            },
            'histgradientboosting': {
                'max_depth': [4, 6, 8],  # Tree depth
                'max_iter': [500, 1000, 2000],  # Number of estimators
                'learning_rate': [0.05, 0.1, 0.15],  # Learning rate
                'l2_regularization': [0.0, 0.1, 0.2],  # Regularization
                'class_weight': [None, 'balanced']  # Class weighting (optional)
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],  # Regularization strength
                'penalty': ['l2'],  # Regularization type
                'solver': ['lbfgs'],
                'class_weight': [None, 'balanced']  # Class weighting (optional)
            },
            'extratrees': {
                'n_estimators': [50, 100, 200],  # Number of estimators
                'max_depth': [5, 10, 15],  # Tree depth
                'min_samples_split': [2, 5, 10],  # Regularization
                'min_samples_leaf': [1, 2, 4],  # Regularization
                'max_features': [0.6, 0.8, 1.0],  # Column sampling
                'class_weight': [None, 'balanced']  # Class weighting (optional)
            },
            'randomforest': {
                'n_estimators': [50, 100, 200],  # Number of estimators
                'max_depth': [5, 10, 15],  # Tree depth
                'min_samples_split': [2, 5, 10],  # Regularization
                'min_samples_leaf': [1, 2, 4],  # Regularization
                'max_features': [0.6, 0.8, 1.0],  # Column sampling
                'class_weight': [None, 'balanced']  # Class weighting (optional)
            }
        }
        
        return search_spaces.get(model_name, {})
    
    def perform_grid_search(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_trials: int = 27) -> Dict[str, Any]:
        """Perform small grid search over hyperparameters"""
        
        logger.info(f"Performing grid search for {model_name} ({n_trials} trials)...")
        
        search_space = self.get_hyperparameter_search_space(model_name)
        default_params = self.get_default_hyperparameters(model_name)
        
        # Generate all combinations (small targeted search)
        from itertools import product
        
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        # Limit to reasonable number of combinations
        if len(param_names) > 0:
            all_combinations = list(product(*param_values))
            # Sample if too many combinations
            if len(all_combinations) > n_trials:
                import random
                random.seed(self.random_seed)
                all_combinations = random.sample(all_combinations, n_trials)
        else:
            all_combinations = [tuple()]
        
        best_score = 0
        best_params = None
        search_results = []
        
        for i, combination in enumerate(all_combinations):
            # Create hyperparameter dict
            trial_params = default_params.copy()
            for j, param_name in enumerate(param_names):
                trial_params[param_name] = combination[j]
            
            logger.info(f"Trial {i+1}/{len(all_combinations)}: {trial_params}")
            
            try:
                # Train model with trial parameters
                model = self.get_model(model_name, trial_params)
                training_info = self.train_model(model, X_train, y_train, X_val, y_val, early_stopping=True)
                
                if training_info['training_successful']:
                    # Evaluate on validation set
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                    else:
                        y_pred_proba = model.predict(X_val)
                    
                    from sklearn.metrics import roc_auc_score
                    val_score = roc_auc_score(y_val, y_pred_proba)
                    
                    search_results.append({
                        'trial': i,
                        'hyperparameters': trial_params,
                        'val_roc_auc': val_score,
                        'training_time': training_info['training_time_seconds']
                    })
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = trial_params.copy()
                    
                    logger.info(f"  Validation ROC-AUC: {val_score:.4f}")
                else:
                    logger.warning(f"  Training failed: {training_info.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
        
        search_summary = {
            'model_name': model_name,
            'n_trials': len(all_combinations),
            'best_score': best_score,
            'best_params': best_params,
            'search_results': search_results,
            'search_completed': datetime.now().isoformat()
        }
        
        logger.info(f"Grid search complete. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return search_summary


def main():
    """Main function to demonstrate model training functionality"""
    trainer = ModelTrainer()
    
    logger.info("Model training module initialized")
    logger.info("Use notebooks/07_model_benchmark.ipynb for comprehensive model benchmarking")
    
    return {
        'status': 'Model training module ready',
        'available_models': trainer.available_models,
        'search_capabilities': ['grid_search', 'early_stopping', 'time_tracking']
    }


if __name__ == "__main__":
    main()