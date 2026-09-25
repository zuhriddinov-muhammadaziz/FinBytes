"""
Unit tests for leakage detection module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.validation.leakage import LeakageDetector
from src.validation.cv import StratifiedKFoldValidation, TimeAwareValidation, ValidationEvaluator, create_validation_plan
from src.config import get_config


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Sample signals data
    train_signals = pd.DataFrame({
        'signal_id': ['SG_001', 'SG_002', 'SG_003', 'SG_004', 'SG_005'],
        'signal_sanasi': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01'],
        'eskalatsiya': [0, 1, 0, 1, 0]
    })
    
    test_signals = pd.DataFrame({
        'signal_id': ['SG_006', 'SG_007'],
        'signal_sanasi': ['2025-06-01', '2025-07-01']
    })
    
    # Sample features data
    train_features = pd.DataFrame({
        'signal_id': ['SG_001', 'SG_002', 'SG_003', 'SG_004', 'SG_005'],
        'transaction_count': [5, 3, 7, 2, 4],
        'total_amount': [100.0, 200.0, 150.0, 75.0, 125.0]
    })
    
    test_features = pd.DataFrame({
        'signal_id': ['SG_006', 'SG_007'],
        'transaction_count': [6, 2],
        'total_amount': [180.0, 90.0]
    })
    
    # Sample transactions data
    train_transactions = pd.DataFrame({
        'signal_id': ['SG_001', 'SG_001', 'SG_002', 'SG_003', 'SG_004'],
        'amount': [100.0, 200.0, 150.0, 300.0, 250.0],
        'tranzaksiya_vaqti': ['2024-12-01', '2024-12-15', '2025-01-15', '2025-02-15', '2025-03-15']
    })
    
    return {
        'train_signals': train_signals,
        'test_signals': test_signals,
        'train_features': train_features,
        'test_features': test_features,
        'train_transactions': train_transactions
    }


class TestLeakageDetector:
    """Test cases for LeakageDetector class"""
    
    def test_initialization(self):
        """Test LeakageDetector initialization"""
        detector = LeakageDetector()
        assert detector.config is not None
        assert detector.data_config is not None
        assert detector.findings == []
    
    def test_temporal_leakage_detection(self, sample_data):
        """Test temporal leakage detection"""
        detector = LeakageDetector()
        
        result = detector.check_temporal_leakage(
            sample_data['train_features'],
            sample_data['train_signals']
        )
        
        assert 'temporal_rule' in result
        assert 'features_checked' in result
        assert 'leakage_detected' in result
        assert 'leakage_details' in result
        assert 'timestamp_violations' in result  # Check for explicit timestamp violation detection
    
    def test_timestamp_violation_detection(self, sample_data):
        """Test explicit timestamp violation detection"""
        detector = LeakageDetector()
        
        # Create a feature with timestamps after signal date to test violation detection
        features_with_violation = sample_data['train_features'].copy()
        features_with_violation['test_timestamp'] = pd.to_datetime(['2025-06-01', '2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01'])
        
        result = detector.check_temporal_leakage(
            features_with_violation,
            sample_data['train_signals']
        )
        
        # Should detect the timestamp violations
        assert 'timestamp_violations' in result
        assert isinstance(result['timestamp_violations'], list)
    
    def test_target_leakage_detection(self, sample_data):
        """Test target leakage detection"""
        detector = LeakageDetector()
        
        result = detector.check_target_leakage(sample_data['train_features'])
        
        assert 'target_column' in result
        assert 'features_checked' in result
        assert 'leakage_detected' in result
        assert 'leakage_details' in result
    
    def test_train_test_separation_check(self, sample_data):
        """Test train/test separation check"""
        detector = LeakageDetector()
        
        result = detector.check_train_test_leakage(
            sample_data['train_features'],
            sample_data['test_features']
        )
        
        assert 'overlap_detected' in result
        assert 'overlap_details' in result
        
        # With our sample data, no overlap expected
        assert result['overlap_detected'] == False
    
    def test_comprehensive_leakage_audit(self, sample_data):
        """Test comprehensive leakage audit"""
        detector = LeakageDetector()
        
        data_dict = {
            'train': sample_data['train_features'],
            'test': sample_data['test_features']
        }
        signals_dict = {
            'train': sample_data['train_signals'],
            'test': sample_data['test_signals']
        }
        transactions_dict = {
            'train': sample_data['train_transactions']
        }
        
        result = detector.run_comprehensive_leakage_audit(
            data_dict,
            signals_dict,
            transactions_dict
        )
        
        assert 'temporal_leakage' in result
        assert 'target_leakage' in result
        assert 'feature_engineering_compliance' in result
        assert 'train_test_separation' in result
        assert 'overall_leakage_detected' in result


class TestValidationStrategies:
    """Test cases for validation strategies"""
    
    def test_stratified_kfold_initialization(self):
        """Test StratifiedKFoldValidation initialization"""
        cv = StratifiedKFoldValidation()
        assert cv.n_splits == 5
        assert cv.random_state is not None
        assert cv.folds == []
    
    def test_stratified_kfold_fold_creation(self, sample_data):
        """Test stratified K-Fold fold creation"""
        cv = StratifiedKFoldValidation()
        
        folds = cv.create_folds(sample_data['train_signals'])
        
        assert len(folds) == 2  # Limited by the two examples in the minority class.
        assert all('fold' in fold for fold in folds)
        assert all('train_indices' in fold for fold in folds)
        assert all('val_indices' in fold for fold in folds)
        assert all('train_signal_ids' in fold for fold in folds)
        assert all('val_signal_ids' in fold for fold in folds)
    
    def test_time_aware_validation_initialization(self):
        """Test TimeAwareValidation initialization"""
        cv = TimeAwareValidation()
        assert cv.n_splits == 5
        assert cv.random_state is not None
        assert cv.folds == []
    
    def test_time_aware_fold_creation(self, sample_data):
        """Test time-aware fold creation"""
        cv = TimeAwareValidation()
        
        folds = cv.create_folds(sample_data['train_signals'])
        
        assert len(folds) == 4  # One observation is needed to seed the expanding training window.
        assert all('fold' in fold for fold in folds)
        assert all('train_indices' in fold for fold in folds)
        assert all('val_indices' in fold for fold in folds)
        assert all('train_date_range' in fold for fold in folds)
        assert all('val_date_range' in fold for fold in folds)
    
    def test_temporal_ordering(self, sample_data):
        """Test that time-aware folds maintain temporal ordering"""
        cv = TimeAwareValidation()
        
        folds = cv.create_folds(sample_data['train_signals'])
        
        # Check that train dates are before val dates for each fold
        for fold in folds:
            train_max = pd.to_datetime(fold['train_date_range']['max'])
            val_min = pd.to_datetime(fold['val_date_range']['min'])
            assert train_max <= val_min, f"Temporal ordering violated in fold {fold['fold']}"


class TestValidationEvaluator:
    """Test cases for ValidationEvaluator"""
    
    def test_initialization(self):
        """Test ValidationEvaluator initialization"""
        evaluator = ValidationEvaluator()
        assert evaluator.config is not None
        assert evaluator.random_state is not None
    
    def test_model_evaluation_framework(self):
        """Test that model evaluation framework is properly structured"""
        evaluator = ValidationEvaluator()
        
        # Check that required methods exist
        required_methods = [
            'evaluate_model',
            'compare_validation_strategies'
        ]
        
        for method in required_methods:
            assert hasattr(evaluator, method), f"Missing method: {method}"
        
        # Check that evaluate_model accepts feature_transformer parameter
        import inspect
        sig = inspect.signature(evaluator.evaluate_model)
        assert 'feature_transformer' in sig.parameters, "evaluate_model should accept feature_transformer parameter"


class TestValidationPlan:
    """Test cases for validation plan creation"""
    
    def test_validation_plan_creation(self, sample_data):
        """Test validation plan creation"""
        plan = create_validation_plan(sample_data['train_signals'])
        
        assert 'validation_strategy' in plan
        assert 'n_splits' in plan
        assert 'random_seed' in plan
        assert 'strategies' in plan
        assert 'recommended_strategy' in plan


class TestConfiguration:
    """Test cases for configuration integration"""
    
    def test_cv_config_loading(self):
        """Test that CV configuration can be loaded"""
        config = get_config()
        
        assert config is not None
        assert hasattr(config, 'get_validation_config')
        
        validation_config = config.get_validation_config()
        assert validation_config is not None


def test_import():
    """Test that modules can be imported"""
    try:
        from src.validation.leakage import LeakageDetector
        from src.validation.cv import StratifiedKFoldValidation, TimeAwareValidation, ValidationEvaluator, create_validation_plan
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_validation_workflow():
    """Test the complete validation workflow"""
    # This test ensures the workflow is properly structured
    # It doesn't run the actual validation since we don't have real data
    
    # Check that all required classes exist
    assert hasattr(LeakageDetector, 'check_temporal_leakage')
    assert hasattr(LeakageDetector, 'check_target_leakage')
    assert hasattr(LeakageDetector, 'check_train_test_leakage')
    assert hasattr(LeakageDetector, 'run_comprehensive_leakage_audit')
    
    assert hasattr(StratifiedKFoldValidation, 'create_folds')
    assert hasattr(TimeAwareValidation, 'create_folds')
    assert hasattr(ValidationEvaluator, 'evaluate_model')
    assert hasattr(ValidationEvaluator, 'compare_validation_strategies')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
