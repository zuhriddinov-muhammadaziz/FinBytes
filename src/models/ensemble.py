"""
Ensemble Methods and Weight Optimization for WIUT FinTech Hackathon
Implements various ensemble strategies using only OOF predictions

This module implements ensemble methods that strictly avoid test set usage:
- Simple probability average
- Rank average  
- Optimized weighted average (using only OOF predictions)
- Blending with small set of candidate weights
- All weight optimization uses only OOF predictions and true labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from .train import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build and optimize ensembles using only OOF predictions"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.random_seed = self.config.get_random_seed()
        self.trainer = ModelTrainer(config_path)
        
        # Store ensemble results
        self.ensemble_results = {}
        
    def simple_average(self, predictions_df: pd.DataFrame, 
                      y_true: pd.Series) -> Dict[str, Any]:
        """Simple probability average ensemble
        
        Args:
            predictions_df: DataFrame with model predictions as columns
            y_true: True labels
            
        Returns:
            Dictionary with ensemble predictions and performance
        """
        logger.info("Computing simple probability average...")
        
        # Compute average across all models
        ensemble_preds = predictions_df.mean(axis=1)
        
        # Calculate performance
        ensemble_auc = roc_auc_score(y_true, ensemble_preds)
        
        result = {
            'method': 'simple_average',
            'predictions': ensemble_preds.values,
            'roc_auc': ensemble_auc,
            'n_models': len(predictions_df.columns),
            'weights': {col: 1.0/len(predictions_df.columns) for col in predictions_df.columns},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Simple average ROC-AUC: {ensemble_auc:.4f}")
        
        return result
    
    def rank_average(self, predictions_df: pd.DataFrame,
                    y_true: pd.Series) -> Dict[str, Any]:
        """Rank average ensemble
        
        Converts predictions to ranks, then averages the ranks,
        then converts back to probabilities.
        
        Args:
            predictions_df: DataFrame with model predictions as columns
            y_true: True labels
            
        Returns:
            Dictionary with ensemble predictions and performance
        """
        logger.info("Computing rank average...")
        
        # Convert predictions to ranks (0-1 scale)
        ranked_preds = predictions_df.rank(axis=0, pct=True)
        
        # Average the ranks
        ensemble_ranks = ranked_preds.mean(axis=1)
        
        # Calculate performance
        ensemble_auc = roc_auc_score(y_true, ensemble_ranks)
        
        result = {
            'method': 'rank_average',
            'predictions': ensemble_ranks.values,
            'roc_auc': ensemble_auc,
            'n_models': len(predictions_df.columns),
            'weights': {col: 1.0/len(predictions_df.columns) for col in predictions_df.columns},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Rank average ROC-AUC: {ensemble_auc:.4f}")
        
        return result
    
    def optimized_weighted_average(self, predictions_df: pd.DataFrame,
                                   y_true: pd.Series,
                                   n_trials: int = 100) -> Dict[str, Any]:
        """Optimized weighted average using grid search on OOF predictions
        
        Searches over a small set of candidate weights using only OOF predictions.
        Uses random sampling from simplex for weight combinations.
        
        Args:
            predictions_df: DataFrame with model predictions as columns
            y_true: True labels
            n_trials: Number of weight combinations to try
            
        Returns:
            Dictionary with ensemble predictions and performance
        """
        logger.info(f"Optimizing weighted average ({n_trials} trials)...")
        
        model_names = predictions_df.columns.tolist()
        n_models = len(model_names)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        best_score = 0
        best_weights = None
        best_predictions = None
        
        # Generate candidate weight combinations
        for trial in range(n_trials):
            # Generate random weights from Dirichlet distribution (sums to 1)
            weights = np.random.dirichlet(np.ones(n_models))
            
            # Compute weighted average
            ensemble_preds = (predictions_df.values * weights).sum(axis=1)
            
            # Calculate performance
            trial_score = roc_auc_score(y_true, ensemble_preds)
            
            if trial_score > best_score:
                best_score = trial_score
                best_weights = weights
                best_predictions = ensemble_preds
        
        # Create weight dictionary
        weight_dict = {name: float(weight) for name, weight in zip(model_names, best_weights)}
        
        result = {
            'method': 'optimized_weighted_average',
            'predictions': best_predictions,
            'roc_auc': best_score,
            'n_models': n_models,
            'weights': weight_dict,
            'n_trials': n_trials,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Optimized weighted average ROC-AUC: {best_score:.4f}")
        logger.info(f"Best weights: {weight_dict}")
        
        return result
    
    def grid_search_weights(self, predictions_df: pd.DataFrame,
                          y_true: pd.Series,
                          weight_steps: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]) -> Dict[str, Any]:
        """Grid search over small set of weight combinations
        
        Searches over a small, predefined set of weight combinations.
        More systematic than random search but limited to small grid.
        
        Args:
            predictions_df: DataFrame with model predictions as columns
            y_true: True labels
            weight_steps: Possible weight values to search
            
        Returns:
            Dictionary with ensemble predictions and performance
        """
        logger.info(f"Grid search weights (steps: {weight_steps})...")
        
        model_names = predictions_df.columns.tolist()
        n_models = len(model_names)
        
        # For simplicity, use grid search for top 3 models only
        # For more models, use random search instead
        if n_models > 3:
            logger.warning(f"Too many models ({n_models}) for grid search, using random search instead")
            return self.optimized_weighted_average(predictions_df, y_true, n_trials=100)
        
        # Generate all combinations that sum to 1
        from itertools import product
        
        best_score = 0
        best_weights = None
        best_predictions = None
        
        # Generate weight combinations
        weight_combinations = product(weight_steps, repeat=n_models)
        
        for weights in weight_combinations:
            # Skip if weights don't sum to 1 (within tolerance)
            if not np.isclose(sum(weights), 1.0, atol=0.01):
                continue
            
            # Compute weighted average
            ensemble_preds = (predictions_df.values * np.array(weights)).sum(axis=1)
            
            # Calculate performance
            trial_score = roc_auc_score(y_true, ensemble_preds)
            
            if trial_score > best_score:
                best_score = trial_score
                best_weights = np.array(weights)
                best_predictions = ensemble_preds
        
        # Create weight dictionary
        weight_dict = {name: float(weight) for name, weight in zip(model_names, best_weights)}
        
        result = {
            'method': 'grid_search_weights',
            'predictions': best_predictions,
            'roc_auc': best_score,
            'n_models': n_models,
            'weights': weight_dict,
            'weight_steps': weight_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Grid search ROC-AUC: {best_score:.4f}")
        logger.info(f"Best weights: {weight_dict}")
        
        return result
    
    def weighted_by_performance(self, predictions_df: pd.DataFrame,
                               y_true: pd.Series,
                               power: float = 2.0) -> Dict[str, Any]:
        """Weight models by their individual OOF performance
        
        Uses ROC-AUC scores to determine weights, with optional power transformation.
        
        Args:
            predictions_df: DataFrame with model predictions as columns
            y_true: True labels
            power: Power to transform performance scores (higher = more extreme weights)
            
        Returns:
            Dictionary with ensemble predictions and performance
        """
        logger.info(f"Weighting by performance (power={power})...")
        
        # Calculate individual model performances
        individual_scores = {}
        for model_name in predictions_df.columns:
            score = roc_auc_score(y_true, predictions_df[model_name])
            individual_scores[model_name] = score
        
        # Transform scores to weights
        # Shift scores to be positive, then apply power
        min_score = min(individual_scores.values())
        shifted_scores = {k: (v - min_score + 0.01) for k, v in individual_scores.items()}
        powered_scores = {k: v ** power for k, v in shifted_scores.items()}
        total_power = sum(powered_scores.values())
        weights = {k: v / total_power for k, v in powered_scores.items()}
        
        # Compute weighted average
        weight_array = np.array([weights[col] for col in predictions_df.columns])
        ensemble_preds = (predictions_df.values * weight_array).sum(axis=1)
        
        # Calculate performance
        ensemble_auc = roc_auc_score(y_true, ensemble_preds)
        
        result = {
            'method': 'weighted_by_performance',
            'predictions': ensemble_preds,
            'roc_auc': ensemble_auc,
            'n_models': len(predictions_df.columns),
            'weights': weights,
            'individual_scores': individual_scores,
            'power': power,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance-weighted ROC-AUC: {ensemble_auc:.4f}")
        logger.info(f"Weights: {weights}")
        
        return result
    
    def compare_ensemble_methods(self, predictions_df: pd.DataFrame,
                                y_true: pd.Series,
                                n_trials: int = 100) -> Dict[str, Any]:
        """Compare multiple ensemble methods
        
        Evaluates:
        1. Individual model ROC-AUC
        2. Simple probability average
        3. Rank average
        4. Optimized weighted average
        5. Performance-weighted average
        
        Args:
            predictions_df: DataFrame with model predictions as columns
            y_true: True labels
            n_trials: Number of trials for optimized methods
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing ensemble methods...")
        
        comparison_results = {
            'individual_models': {},
            'ensemble_methods': {},
            'best_method': None,
            'best_score': 0,
            'comparison_completed': datetime.now().isoformat()
        }
        
        # Evaluate individual models
        logger.info("Evaluating individual models...")
        for model_name in predictions_df.columns:
            individual_auc = roc_auc_score(y_true, predictions_df[model_name])
            comparison_results['individual_models'][model_name] = individual_auc
            logger.info(f"  {model_name}: {individual_auc:.4f}")
        
        # Simple average
        simple_avg = self.simple_average(predictions_df, y_true)
        comparison_results['ensemble_methods']['simple_average'] = simple_avg
        
        # Rank average
        rank_avg = self.rank_average(predictions_df, y_true)
        comparison_results['ensemble_methods']['rank_average'] = rank_avg
        
        # Optimized weighted average
        optimized_avg = self.optimized_weighted_average(predictions_df, y_true, n_trials)
        comparison_results['ensemble_methods']['optimized_weighted_average'] = optimized_avg
        
        # Performance-weighted average
        perf_weighted = self.weighted_by_performance(predictions_df, y_true)
        comparison_results['ensemble_methods']['weighted_by_performance'] = perf_weighted
        
        # Find best method
        all_methods = {
            **comparison_results['individual_models'],
            **{k: v['roc_auc'] for k, v in comparison_results['ensemble_methods'].items()}
        }
        
        best_method = max(all_methods.items(), key=lambda x: x[1])
        comparison_results['best_method'] = best_method[0]
        comparison_results['best_score'] = best_method[1]
        
        logger.info(f"Best method: {best_method[0]} (ROC-AUC: {best_method[1]:.4f})")
        
        return comparison_results
    
    def select_final_ensemble(self, comparison_results: Dict[str, Any],
                             min_improvement: float = 0.005) -> Dict[str, Any]:
        """Select final ensemble based on OOF evidence
        
        Selects ensemble if it provides meaningful improvement over best individual model.
        
        Args:
            comparison_results: Results from compare_ensemble_methods
            min_improvement: Minimum improvement over best individual model to use ensemble
            
        Returns:
            Dictionary with selected ensemble information
        """
        logger.info("Selecting final ensemble...")
        
        # Get best individual model score
        best_individual = max(comparison_results['individual_models'].values())
        
        # Get best ensemble score
        ensemble_scores = {k: v['roc_auc'] for k, v in comparison_results['ensemble_methods'].items()}
        best_ensemble_method = max(ensemble_scores.items(), key=lambda x: x[1])
        best_ensemble_score = best_ensemble_method[1]
        
        # Calculate improvement
        improvement = best_ensemble_score - best_individual
        
        logger.info(f"Best individual: {best_individual:.4f}")
        logger.info(f"Best ensemble: {best_ensemble_method[0]} ({best_ensemble_score:.4f})")
        logger.info(f"Improvement: {improvement:+.4f}")
        
        # Select ensemble if improvement is meaningful
        if improvement >= min_improvement:
            selected_ensemble = {
                'use_ensemble': True,
                'method': best_ensemble_method[0],
                'score': best_ensemble_score,
                'improvement': improvement,
                'weights': comparison_results['ensemble_methods'][best_ensemble_method[0]]['weights'],
                'reason': f"Ensemble provides {improvement:.4f} improvement over best individual model"
            }
            logger.info(f"Selected ensemble: {best_ensemble_method[0]}")
        else:
            # Select best individual model
            best_individual_model = max(comparison_results['individual_models'].items(), key=lambda x: x[1])
            selected_ensemble = {
                'use_ensemble': False,
                'method': best_individual_model[0],
                'score': best_individual_model[1],
                'improvement': 0.0,
                'weights': {best_individual_model[0]: 1.0},
                'reason': f"Individual model {best_individual_model[0]} performs well, ensemble improvement ({improvement:.4f}) below threshold ({min_improvement})"
            }
            logger.info(f"Selected individual model: {best_individual_model[0]}")
        
        return selected_ensemble
    
    def apply_ensemble_to_test(self, test_predictions_df: pd.DataFrame,
                              ensemble_config: Dict[str, Any]) -> np.ndarray:
        """Apply selected ensemble method to test predictions
        
        Args:
            test_predictions_df: DataFrame with test predictions from each model
            ensemble_config: Configuration from select_final_ensemble
            
        Returns:
            Ensemble predictions for test set
        """
        logger.info("Applying ensemble to test predictions...")
        
        if ensemble_config['use_ensemble']:
            method = ensemble_config['method']
            weights = ensemble_config['weights']
            
            if method == 'simple_average':
                # Equal weights
                ensemble_preds = test_predictions_df.mean(axis=1).values
                
            elif method == 'rank_average':
                # Rank average
                ranked_preds = test_predictions_df.rank(axis=0, pct=True)
                ensemble_preds = ranked_preds.mean(axis=1).values
                
            elif method in ['optimized_weighted_average', 'grid_search_weights', 'weighted_by_performance']:
                # Weighted average
                weight_array = np.array([weights.get(col, 0) for col in test_predictions_df.columns])
                ensemble_preds = (test_predictions_df.values * weight_array).sum(axis=1)
                
            else:
                # Default to simple average
                logger.warning(f"Unknown method {method}, using simple average")
                ensemble_preds = test_predictions_df.mean(axis=1).values
                
        else:
            # Use single best model
            model_name = ensemble_config['method']
            ensemble_preds = test_predictions_df[model_name].values
        
        # Ensure predictions are in [0,1]
        ensemble_preds = np.clip(ensemble_preds, 0, 1)
        
        logger.info(f"Test ensemble predictions generated (method: {ensemble_config['method']})")
        
        return ensemble_preds
    
    def save_ensemble_results(self, results: Dict[str, Any],
                             output_path: str = "experiments/ensemble_results.json") -> None:
        """Save ensemble results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Ensemble results saved to {output_path}")
    
    def create_ensemble_csv(self, comparison_results: Dict[str, Any],
                           selected_ensemble: Dict[str, Any],
                           output_path: str = "experiments/ensemble_results.csv") -> None:
        """Create CSV summary of ensemble results"""
        csv_data = []
        
        # Individual models
        for model_name, score in comparison_results['individual_models'].items():
            csv_data.append({
                'method': model_name,
                'type': 'individual',
                'roc_auc': score,
                'selected': selected_ensemble['method'] == model_name and not selected_ensemble['use_ensemble']
            })
        
        # Ensemble methods
        for method_name, results in comparison_results['ensemble_methods'].items():
            csv_data.append({
                'method': method_name,
                'type': 'ensemble',
                'roc_auc': results['roc_auc'],
                'selected': selected_ensemble['method'] == method_name and selected_ensemble['use_ensemble']
            })
        
        df = pd.DataFrame(csv_data)
        df = df.sort_values('roc_auc', ascending=False)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Ensemble CSV saved to {output_path}")


def main():
    """Main function to demonstrate ensemble functionality"""
    ensemble_builder = EnsembleBuilder()
    
    logger.info("Ensemble module initialized")
    logger.info("Use notebooks/08_ensemble.ipynb for comprehensive ensemble building")
    
    return {
        'status': 'Ensemble module ready',
        'available_methods': [
            'simple_average',
            'rank_average',
            'optimized_weighted_average',
            'grid_search_weights',
            'weighted_by_performance'
        ]
    }


if __name__ == "__main__":
    main()