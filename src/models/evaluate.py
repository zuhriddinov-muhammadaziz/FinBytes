"""
Model Evaluation Module for WIUT FinTech Hackathon
Model evaluation with comparison and reproducibility
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
from .train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation with comprehensive metrics and comparison"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.random_seed = self.config.get_random_seed()
        self.trainer = ModelTrainer(config_path)
    
    def evaluate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Evaluate model on validation data"""
        
        logger.info("Evaluating model...")
        
        start_time = time.time()
        
        evaluation_info = {
            'model_name': model.__class__.__name__,
            'model_params': model.get_params(),
            'n_samples': len(X_val),
            'n_features': X_val.shape[1],
            'evaluation_started': datetime.now().isoformat()
        }
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_val)
                y_pred_proba = y_pred  # For models without predict_proba
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
            
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            evaluation_info.update({
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'evaluation_time_seconds': time.time() - start_time,
                'evaluation_completed': datetime.now().isoformat(),
                'evaluation_successful': True
            })
            
            logger.info(f"Evaluation complete. ROC-AUC: {roc_auc:.4f}")
            
        except Exception as e:
            evaluation_info.update({
                'evaluation_time_seconds': time.time() - start_time,
                'evaluation_completed': datetime.now().isoformat(),
                'evaluation_successful': False,
                'error': str(e)
            })
            
            logger.error(f"Evaluation failed: {e}")
        
        return evaluation_info
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           folds: List[Dict[str, Any]], 
                           feature_transformer=None) -> Dict[str, Any]:
        """Cross-validate model using specified folds"""
        
        logger.info(f"Cross-validating model with {len(folds)} folds...")
        
        fold_results = []
        fold_predictions = []
        
        for fold in folds:
            train_idx = fold['train_indices']
            val_idx = fold['val_indices']
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply feature transformer if provided
            if feature_transformer is not None:
                feature_transformer.fit(X_train, y_train)
                X_train = feature_transformer.transform(X_train)
                X_val = feature_transformer.transform(X_val)
            
            # Train model
            training_info = self.trainer.train_model(model, X_train, y_train, X_val, y_val, early_stopping=True)
            
            if training_info['training_successful']:
                # Evaluate
                eval_info = self.evaluate_model(model, X_val, y_val)
                
                fold_results.append({
                    'fold': fold['fold'],
                    'train_samples': len(train_idx),
                    'val_samples': len(val_idx),
                    'training_time': training_info['training_time_seconds'],
                    'eval_info': eval_info
                })
                
                # Store predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred_proba = model.predict(X_val)
                
                fold_predictions.append({
                    'fold': fold['fold'],
                    'val_indices': val_idx,
                    'y_true': y_val.tolist(),
                    'y_pred_proba': y_pred_proba.tolist()
                })
        
        # Calculate aggregate metrics
        if fold_results:
            roc_aucs = [fold['eval_info']['roc_auc'] for fold in fold_results if fold['eval_info']['evaluation_successful']]
            
            cv_results = {
                'model_name': model.__class__.__name__,
                'model_params': model.get_params(),
                'n_folds': len(folds),
                'mean_roc_auc': np.mean(roc_aucs) if roc_aucs else 0,
                'std_roc_auc': np.std(roc_aucs) if roc_aucs else 0,
                'per_fold_roc_auc': roc_aucs if roc_aucs else [],
                'total_training_time': sum(fold['training_time'] for fold in fold_results),
                'fold_results': fold_results,
                'fold_predictions': fold_predictions,
                'cv_completed': datetime.now().isoformat()
            }
            
            logger.info(f"Cross-validation complete. Mean ROC-AUC: {cv_results['mean_roc_auc']:.4f} (+/- {cv_results['std_roc_auc']:.4f})")
        else:
            cv_results = {
                'model_name': model.__class__.__name__,
                'model_params': model.get_params(),
                'n_folds': len(folds),
                'mean_roc_auc': 0,
                'std_roc_auc': 0,
                'per_fold_roc_auc': [],
                'total_training_time': 0,
                'fold_results': [],
                'fold_predictions': [],
                'cv_completed': datetime.now().isoformat(),
                'error': 'All folds failed'
            }
        
        return cv_results
    
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                      folds: List[Dict[str, Any]], feature_transformer=None) -> Dict[str, Any]:
        """Compare multiple models using the same validation folds"""
        
        logger.info(f"Comparing {len(models)} models using fixed validation folds...")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Clone model to avoid state pollution
            import copy
            model_copy = copy.deepcopy(model)
            
            cv_results = self.cross_validate_model(model_copy, X, y, folds, feature_transformer)
            comparison_results[model_name] = cv_results
        
        # Create comparison summary
        comparison_summary = {
            'comparison_completed': datetime.now().isoformat(),
            'n_folds': len(folds),
            'random_seed': self.random_seed,
            'model_results': comparison_results,
            'best_model': None,
            'best_score': 0
        }
        
        # Find best model
        for model_name, results in comparison_results.items():
            if results['mean_roc_auc'] > comparison_summary['best_score']:
                comparison_summary['best_score'] = results['mean_roc_auc']
                comparison_summary['best_model'] = model_name
        
        logger.info(f"Comparison complete. Best model: {comparison_summary['best_model']} (ROC-AUC: {comparison_summary['best_score']:.4f})")
        
        return comparison_summary
    
    def benchmark_model_set(self, model_configs: List[Dict[str, Any]], 
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          folds: List[Dict[str, Any]],
                          feature_set_name: str = "v1_features") -> Dict[str, Any]:
        """Benchmark a set of models with hyperparameters
        
        Follows competition rules:
        1. Uses identical validation folds for fair comparisons
        2. Tracks training time
        3. Records exact feature set and hyperparameters
        4. Uses early stopping where appropriate
        5. Does not tune on the test set
        6. Keeps reproducible experiment table
        """
        
        logger.info(f"Benchmarking {len(model_configs)} models with {feature_set_name}...")
        logger.info(f"Using {len(folds)} identical validation folds for fair comparison")
        
        benchmark_results = {
            'feature_set': feature_set_name,
            'n_features': X_train.shape[1],
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_folds': len(folds),
            'random_seed': self.random_seed,
            'benchmark_started': datetime.now().isoformat(),
            'model_results': {}
        }
        
        for config in model_configs:
            model_name = config['model_name']
            hyperparameters = config.get('hyperparameters', {})
            
            logger.info(f"Benchmarking {model_name}...")
            logger.info(f"  Hyperparameters: {hyperparameters}")
            
            try:
                # Get model
                model = self.trainer.get_model(model_name, hyperparameters)
                
                # Train model with early stopping
                training_info = self.trainer.train_model(model, X_train, y_train, X_val, y_val, early_stopping=True)
                
                if training_info['training_successful']:
                    # Evaluate model
                    eval_info = self.evaluate_model(model, X_val, y_val)
                    
                    if eval_info['evaluation_successful']:
                        # Cross-validate for robust estimate using identical folds
                        cv_results = self.cross_validate_model(model, X_train, y_train, folds)
                        
                        benchmark_results['model_results'][model_name] = {
                            'hyperparameters': hyperparameters,
                            'training_time': training_info['training_time_seconds'],
                            'early_stopping_used': training_info.get('early_stopping_used', False),
                            'val_roc_auc': eval_info['roc_auc'],
                            'cv_mean_roc_auc': cv_results['mean_roc_auc'],
                            'cv_std_roc_auc': cv_results['std_roc_auc'],
                            'cv_per_fold_roc_auc': cv_results['per_fold_roc_auc'],
                            'cv_total_time': cv_results['total_training_time'],
                            'benchmark_successful': True,
                            'feature_set': feature_set_name,
                            'n_features': X_train.shape[1]
                        }
                        
                        logger.info(f"  {model_name}: CV ROC-AUC = {cv_results['mean_roc_auc']:.4f} ± {cv_results['std_roc_auc']:.4f}")
                        logger.info(f"  Training time: {training_info['training_time_seconds']:.2f}s")
                    else:
                        benchmark_results['model_results'][model_name] = {
                            'hyperparameters': hyperparameters,
                            'training_time': training_info['training_time_seconds'],
                            'benchmark_successful': False,
                            'error': 'Evaluation failed'
                        }
                        logger.warning(f"  {model_name}: Evaluation failed")
                else:
                    benchmark_results['model_results'][model_name] = {
                        'hyperparameters': hyperparameters,
                        'benchmark_successful': False,
                        'error': training_info.get('error', 'Training failed')
                    }
                    logger.warning(f"  {model_name}: Training failed")
            
            except Exception as e:
                benchmark_results['model_results'][model_name] = {
                    'hyperparameters': hyperparameters,
                    'benchmark_successful': False,
                    'error': str(e)
                }
                logger.error(f"  {model_name}: Exception - {e}")
        
        benchmark_results['benchmark_completed'] = datetime.now().isoformat()
        
        # Find best model
        successful_results = {k: v for k, v in benchmark_results['model_results'].items() 
                           if v.get('benchmark_successful', False)}
        
        if successful_results:
            best_model = max(success_results.items(), key=lambda x: x[1]['cv_mean_roc_auc'])
            benchmark_results['best_model'] = best_model[0]
            benchmark_results['best_score'] = best_model[1]['cv_mean_roc_auc']
        else:
            benchmark_results['best_model'] = None
            benchmark_results['best_score'] = 0
        
        logger.info(f"Benchmark complete. Best model: {benchmark_results['best_model']} (ROC-AUC: {benchmark_results['best_score']:.4f})")
        
        return benchmark_results
    
    def save_benchmark_results(self, results: Dict[str, Any], 
                              output_path: str = "experiments/model_results.json") -> None:
        """Save benchmark results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_path}")
    
    def create_results_csv(self, results: Dict[str, Any], 
                         output_path: str = "experiments/model_results.csv") -> None:
        """Create CSV summary of benchmark results for reproducible experiment table
        
        Includes all required information:
        - Feature set used
        - Model name and hyperparameters
        - ROC-AUC performance (mean, std, per-fold)
        - Training time metrics
        - Early stopping usage
        - Number of features
        """
        
        csv_data = []
        
        for model_name, model_results in results['model_results'].items():
            if model_results.get('benchmark_successful', False):
                csv_data.append({
                    'feature_set': results['feature_set'],
                    'model': model_name,
                    'cv_mean_roc_auc': model_results['cv_mean_roc_auc'],
                    'cv_std_roc_auc': model_results['cv_std_roc_auc'],
                    'val_roc_auc': model_results['val_roc_auc'],
                    'training_time': model_results['training_time'],
                    'cv_total_time': model_results['cv_total_time'],
                    'early_stopping_used': model_results.get('early_stopping_used', False),
                    'n_features': model_results.get('n_features', results.get('n_features', 'unknown')),
                    'n_folds': results.get('n_folds', len(model_results.get('cv_per_fold_roc_auc', []))),
                    'random_seed': results.get('random_seed', 'unknown'),
                    'hyperparameters': str(model_results['hyperparameters'])
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df = df.sort_values('cv_mean_roc_auc', ascending=False)
            df.to_csv(output_path, index=False)
            logger.info(f"CSV results saved to {output_path}")
            logger.info(f"Experiment table contains {len(df)} successful model runs")
        else:
            logger.warning("No successful benchmarks to save to CSV")


def main():
    """Main function to demonstrate model evaluation functionality"""
    evaluator = ModelEvaluator()
    
    logger.info("Model evaluation module initialized")
    logger.info("Use notebooks/07_model_benchmark.ipynb for comprehensive model benchmarking")
    
    return {
        'status': 'Model evaluation module ready',
        'available_capabilities': [
            'cross_validation',
            'model_comparison',
            'benchmark_model_set',
            'grid_search_integration'
        ]
    }


if __name__ == "__main__":
    main()