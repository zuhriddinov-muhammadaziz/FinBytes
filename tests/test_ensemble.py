"""
Unit Tests for Ensemble Functionality
Tests OOF prediction generation and ensemble methods
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.oof import OOFPredictor, generate_diverse_model_configs
from src.models.ensemble import EnsembleBuilder
from src.models.train import ModelTrainer


class TestOOFPredictor:
    """Test OOF prediction generation"""
    
    @pytest.fixture
    def oof_predictor(self):
        """Create OOFPredictor instance"""
        return OOFPredictor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Create sample folds
        folds = []
        fold_size = n_samples // 5
        for i in range(5):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < 4 else n_samples
            val_indices = list(range(val_start, val_end))
            train_indices = [j for j in range(n_samples) if j not in val_indices]
            
            folds.append({
                'fold': i,
                'train_indices': train_indices,
                'val_indices': val_indices
            })
        
        return X, y, folds
    
    def test_oof_predictor_initialization(self, oof_predictor):
        """Test OOFPredictor initialization"""
        assert oof_predictor is not None
        assert oof_predictor.trainer is not None
        assert hasattr(oof_predictor, 'oof_predictions')
    
    def test_generate_oof_predictions_basic(self, oof_predictor, sample_data):
        """Test basic OOF prediction generation"""
        X, y, folds = sample_data
        
        # Create simple model config
        model_configs = [
            {
                'model_name': 'logistic_regression',
                'hyperparameters': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            }
        ]
        
        results = oof_predictor.generate_oof_predictions(
            model_configs, X, y, folds, feature_set_name="test_features"
        )
        
        assert 'logistic_regression' in results
        assert 'metadata' in results
        assert results['logistic_regression']['oof_predictions'].shape == (len(X),)
        assert 'oof_roc_auc' in results['logistic_regression']
        assert results['metadata']['n_samples'] == len(X)
    
    def test_oof_predictions_all_samples_covered(self, oof_predictor, sample_data):
        """Test that all samples get OOF predictions"""
        X, y, folds = sample_data
        
        model_configs = [
            {
                'model_name': 'logistic_regression',
                'hyperparameters': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            }
        ]
        
        results = oof_predictor.generate_oof_predictions(
            model_configs, X, y, folds, feature_set_name="test_features"
        )
        
        oof_preds = results['logistic_regression']['oof_predictions']
        
        # Check no NaN values
        assert not np.any(np.isnan(oof_preds))
        
        # Check all values are in reasonable range
        assert np.all(oof_preds >= 0) and np.all(oof_preds <= 1)
    
    def test_save_and_load_oof_predictions(self, oof_predictor, sample_data, tmp_path):
        """Test saving and loading OOF predictions"""
        X, y, folds = sample_data
        
        model_configs = [
            {
                'model_name': 'logistic_regression',
                'hyperparameters': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            }
        ]
        
        oof_predictor.generate_oof_predictions(
            model_configs, X, y, folds, feature_set_name="test_features"
        )
        
        # Save
        output_path = tmp_path / "test_oof.parquet"
        oof_predictor.save_oof_predictions(str(output_path))
        
        assert output_path.exists()
        
        # Load
        loaded_df = oof_predictor.load_oof_predictions(str(output_path))
        
        assert 'logistic_regression' in loaded_df.columns
        assert len(loaded_df) == len(X)
    
    def test_get_model_oof_performance(self, oof_predictor, sample_data):
        """Test getting model OOF performance summary"""
        X, y, folds = sample_data
        
        model_configs = [
            {
                'model_name': 'logistic_regression',
                'hyperparameters': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            }
        ]
        
        oof_predictor.generate_oof_predictions(
            model_configs, X, y, folds, feature_set_name="test_features"
        )
        
        performance_df = oof_predictor.get_model_oof_performance()
        
        assert 'model' in performance_df.columns
        assert 'oof_roc_auc' in performance_df.columns
        assert len(performance_df) == 1
        assert performance_df.iloc[0]['model'] == 'logistic_regression'


class TestEnsembleBuilder:
    """Test ensemble building methods"""
    
    @pytest.fixture
    def ensemble_builder(self):
        """Create EnsembleBuilder instance"""
        return EnsembleBuilder()
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing"""
        np.random.seed(42)
        n_samples = 100
        
        predictions_df = pd.DataFrame({
            'model_a': np.random.rand(n_samples),
            'model_b': np.random.rand(n_samples),
            'model_c': np.random.rand(n_samples)
        })
        
        y_true = pd.Series(np.random.randint(0, 2, n_samples))
        
        return predictions_df, y_true
    
    def test_ensemble_builder_initialization(self, ensemble_builder):
        """Test EnsembleBuilder initialization"""
        assert ensemble_builder is not None
        assert ensemble_builder.trainer is not None
        assert hasattr(ensemble_builder, 'ensemble_results')
    
    def test_simple_average(self, ensemble_builder, sample_predictions):
        """Test simple average ensemble"""
        predictions_df, y_true = sample_predictions
        
        result = ensemble_builder.simple_average(predictions_df, y_true)
        
        assert result['method'] == 'simple_average'
        assert len(result['predictions']) == len(predictions_df)
        assert 'roc_auc' in result
        assert result['n_models'] == 3
        assert all(w == 1/3 for w in result['weights'].values())
    
    def test_rank_average(self, ensemble_builder, sample_predictions):
        """Test rank average ensemble"""
        predictions_df, y_true = sample_predictions
        
        result = ensemble_builder.rank_average(predictions_df, y_true)
        
        assert result['method'] == 'rank_average'
        assert len(result['predictions']) == len(predictions_df)
        assert 'roc_auc' in result
        assert result['n_models'] == 3
    
    def test_optimized_weighted_average(self, ensemble_builder, sample_predictions):
        """Test optimized weighted average"""
        predictions_df, y_true = sample_predictions
        
        result = ensemble_builder.optimized_weighted_average(predictions_df, y_true, n_trials=10)
        
        assert result['method'] == 'optimized_weighted_average'
        assert len(result['predictions']) == len(predictions_df)
        assert 'roc_auc' in result
        assert result['n_models'] == 3
        assert len(result['weights']) == 3
        assert np.isclose(sum(result['weights'].values()), 1.0, atol=0.01)
    
    def test_weighted_by_performance(self, ensemble_builder, sample_predictions):
        """Test performance-weighted ensemble"""
        predictions_df, y_true = sample_predictions
        
        result = ensemble_builder.weighted_by_performance(predictions_df, y_true, power=2.0)
        
        assert result['method'] == 'weighted_by_performance'
        assert len(result['predictions']) == len(predictions_df)
        assert 'roc_auc' in result
        assert 'individual_scores' in result
        assert len(result['weights']) == 3
        assert np.isclose(sum(result['weights'].values()), 1.0, atol=0.01)
    
    def test_compare_ensemble_methods(self, ensemble_builder, sample_predictions):
        """Test comparison of ensemble methods"""
        predictions_df, y_true = sample_predictions
        
        comparison = ensemble_builder.compare_ensemble_methods(
            predictions_df, y_true, n_trials=10
        )
        
        assert 'individual_models' in comparison
        assert 'ensemble_methods' in comparison
        assert 'best_method' in comparison
        assert 'best_score' in comparison
        
        # Check individual models
        assert len(comparison['individual_models']) == 3
        
        # Check ensemble methods
        assert 'simple_average' in comparison['ensemble_methods']
        assert 'rank_average' in comparison['ensemble_methods']
        assert 'optimized_weighted_average' in comparison['ensemble_methods']
    
    def test_select_final_ensemble_with_improvement(self, ensemble_builder, sample_predictions):
        """Test ensemble selection when ensemble provides improvement"""
        predictions_df, y_true = sample_predictions
        
        # Create comparison with ensemble improvement
        comparison = {
            'individual_models': {
                'model_a': 0.75,
                'model_b': 0.78,
                'model_c': 0.76
            },
            'ensemble_methods': {
                'simple_average': {
                    'roc_auc': 0.80,
                    'weights': {'model_a': 0.33, 'model_b': 0.33, 'model_c': 0.34}
                }
            }
        }
        
        selected = ensemble_builder.select_final_ensemble(comparison, min_improvement=0.01)
        
        assert selected['use_ensemble'] == True
        assert selected['method'] == 'simple_average'
        assert selected['improvement'] > 0
    
    def test_select_final_ensemble_without_improvement(self, ensemble_builder, sample_predictions):
        """Test ensemble selection when individual model is better"""
        predictions_df, y_true = sample_predictions
        
        # Create comparison without ensemble improvement
        comparison = {
            'individual_models': {
                'model_a': 0.85,
                'model_b': 0.78,
                'model_c': 0.76
            },
            'ensemble_methods': {
                'simple_average': {
                    'roc_auc': 0.80,
                    'weights': {'model_a': 0.33, 'model_b': 0.33, 'model_c': 0.34}
                }
            }
        }
        
        selected = ensemble_builder.select_final_ensemble(comparison, min_improvement=0.01)
        
        assert selected['use_ensemble'] == False
        assert selected['method'] == 'model_a'
        assert selected['improvement'] == 0.0
    
    def test_apply_ensemble_to_test_ensemble(self, ensemble_builder, sample_predictions):
        """Test applying ensemble to test predictions"""
        predictions_df, y_true = sample_predictions
        
        ensemble_config = {
            'use_ensemble': True,
            'method': 'simple_average',
            'weights': {'model_a': 0.33, 'model_b': 0.33, 'model_c': 0.34}
        }
        
        test_predictions = pd.DataFrame({
            'model_a': np.random.rand(50),
            'model_b': np.random.rand(50),
            'model_c': np.random.rand(50)
        })
        
        ensemble_preds = ensemble_builder.apply_ensemble_to_test(test_predictions, ensemble_config)
        
        assert len(ensemble_preds) == 50
        assert np.all(ensemble_preds >= 0) and np.all(ensemble_preds <= 1)
    
    def test_apply_ensemble_to_test_single_model(self, ensemble_builder, sample_predictions):
        """Test applying single model to test predictions"""
        predictions_df, y_true = sample_predictions
        
        ensemble_config = {
            'use_ensemble': False,
            'method': 'model_b',
            'weights': {'model_b': 1.0}
        }
        
        test_predictions = pd.DataFrame({
            'model_a': np.random.rand(50),
            'model_b': np.random.rand(50),
            'model_c': np.random.rand(50)
        })
        
        ensemble_preds = ensemble_builder.apply_ensemble_to_test(test_predictions, ensemble_config)
        
        assert len(ensemble_preds) == 50
        assert np.all(ensemble_preds >= 0) and np.all(ensemble_preds <= 1)
        # Should be identical to model_b predictions
        np.testing.assert_array_equal(ensemble_preds, test_predictions['model_b'].values)


class TestDiverseModelConfigs:
    """Test diverse model configuration generation"""
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance"""
        return ModelTrainer()
    
    def test_generate_diverse_model_configs_auto(self, trainer):
        """Test automatic diverse model selection"""
        configs = generate_diverse_model_configs(trainer)
        
        assert isinstance(configs, list)
        assert len(configs) > 0
        
        # Check each config has required fields
        for config in configs:
            assert 'model_name' in config
            assert 'hyperparameters' in config
    
    def test_generate_diverse_model_configs_specific(self, trainer):
        """Test specific model selection"""
        specific_models = ['logistic_regression', 'extratrees']
        configs = generate_diverse_model_configs(trainer, best_models=specific_models)
        
        assert len(configs) == 2
        model_names = [config['model_name'] for config in configs]
        assert set(model_names) == set(specific_models)


class TestEnsembleIntegration:
    """Integration tests for ensemble pipeline"""
    
    @pytest.fixture
    def ensemble_builder(self):
        """Create EnsembleBuilder instance"""
        return EnsembleBuilder()
    
    @pytest.fixture
    def oof_predictor(self):
        """Create OOFPredictor instance"""
        return OOFPredictor()
    
    def test_full_ensemble_pipeline(self, oof_predictor, ensemble_builder, tmp_path):
        """Test complete ensemble pipeline from OOF to final ensemble"""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Create folds
        folds = []
        fold_size = n_samples // 5
        for i in range(5):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < 4 else n_samples
            val_indices = list(range(val_start, val_end))
            train_indices = [j for j in range(n_samples) if j not in val_indices]
            folds.append({
                'fold': i,
                'train_indices': train_indices,
                'val_indices': val_indices
            })
        
        # Generate OOF predictions
        model_configs = [
            {
                'model_name': 'logistic_regression',
                'hyperparameters': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            }
        ]
        
        oof_results = oof_predictor.generate_oof_predictions(
            model_configs, X, y, folds, feature_set_name="test_features"
        )
        
        # Save OOF predictions
        oof_path = tmp_path / "test_oof.parquet"
        oof_predictor.save_oof_predictions(str(oof_path))
        
        # Load OOF predictions
        oof_df = oof_predictor.load_oof_predictions(str(oof_path))
        
        # Compare ensemble methods
        comparison = ensemble_builder.compare_ensemble_methods(oof_df, y, n_trials=5)
        
        # Select final ensemble
        selected = ensemble_builder.select_final_ensemble(comparison)
        
        # Verify pipeline worked
        assert 'logistic_regression' in oof_results
        assert comparison['best_method'] is not None
        assert selected['method'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])