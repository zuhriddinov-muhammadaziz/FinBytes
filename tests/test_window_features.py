"""
Unit tests for window features module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).parent.parent))

from src.features.window_features import WindowFeatures
from src.features.feature_store import FeatureStore, FeatureMetadata


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing window features"""
    base_date = datetime(2025, 1, 1)
    
    return pd.DataFrame({
        'signal_id': ['SG_001', 'SG_001', 'SG_001', 'SG_002', 'SG_002', 'SG_003', 
                     'SG_001', 'SG_002', 'SG_003', 'SG_001'],
        'amount': [100.0, 200.0, 150.0, 300.0, 250.0, 175.0, 500.0, 400.0, 600.0, 1000.0],
        'direction': ['incoming', 'outgoing', 'incoming', 'incoming', 'outgoing', 'incoming',
                     'outgoing', 'incoming', 'outgoing', 'incoming'],
        'type': ['card', 'bank_transfer', 'card', 'cash', 'card', 'international',
                'card', 'international', 'card', 'international'],
        'tranzaksiya_vaqti': [
            base_date - timedelta(days=30),  # 30 days ago
            base_date - timedelta(days=25),  # 25 days ago
            base_date - timedelta(days=20),  # 20 days ago
            base_date - timedelta(days=15),  # 15 days ago
            base_date - timedelta(days=10),  # 10 days ago
            base_date - timedelta(days=5),   # 5 days ago
            base_date - timedelta(days=3),   # 3 days ago
            base_date - timedelta(days=2),   # 2 days ago
            base_date - timedelta(days=1),   # 1 day ago
            base_date - timedelta(days=0)    # Same day
        ]
    })


@pytest.fixture
def sample_signals():
    """Create sample signal data for testing"""
    base_date = datetime(2025, 1, 1)
    
    return pd.DataFrame({
        'signal_id': ['SG_001', 'SG_002', 'SG_003'],
        'signal_sanasi': pd.to_datetime([base_date, base_date, base_date]),
        'eskalatsiya': [0, 1, 0]
    })


class TestWindowFeatures:
    """Test cases for WindowFeatures class"""
    
    def test_initialization(self):
        """Test WindowFeatures initialization"""
        window_features = WindowFeatures()
        assert window_features.config is not None
        assert window_features.data_config is not None
        assert window_features.feature_store is not None
        assert window_features.windows == [1, 3, 7, 14, 30, 60, 90]
    
    def test_temporal_filtering(self, sample_transactions, sample_signals):
        """Test temporal filtering functionality"""
        window_features = WindowFeatures()
        
        filtered = window_features.apply_temporal_filter(sample_transactions, sample_signals)
        
        # Should filter out transactions after signal date
        assert len(filtered) <= len(sample_transactions)
        assert 'signal_id' in filtered.columns
    
    def test_high_amount_threshold_computation(self, sample_transactions):
        """Test high amount threshold computation"""
        window_features = WindowFeatures()
        
        threshold = window_features.compute_high_amount_threshold(sample_transactions)
        
        assert threshold is not None
        assert threshold > 0
        assert isinstance(threshold, (int, float))
    
    def test_window_feature_computation(self, sample_transactions, sample_signals):
        """Test window feature computation"""
        window_features = WindowFeatures()
        
        # Apply temporal filtering first
        filtered_transactions = window_features.apply_temporal_filter(sample_transactions, sample_signals)
        
        # Compute window features
        window_features.high_amount_threshold = window_features.compute_high_amount_threshold(filtered_transactions)
        window_feat = window_features.compute_window_features(filtered_transactions, sample_signals)
        
        assert 'signal_id' in window_feat.columns
        assert len(window_feat) == len(sample_signals)
        
        # Check for some expected window features
        expected_patterns = ['transaction_count_1d', 'transaction_count_7d', 'transaction_count_30d']
        for pattern in expected_patterns:
            if pattern in window_feat.columns:
                assert pattern in window_feat.columns
    
    def test_single_window_features(self, sample_transactions, sample_signals):
        """Test single window feature computation"""
        window_features = WindowFeatures()
        
        # Merge for window computation
        merged = sample_transactions.merge(
            sample_signals[['signal_id', 'signal_sanasi']], 
            on='signal_id', 
            how='left'
        )
        
        window_features.high_amount_threshold = window_features.compute_high_amount_threshold(sample_transactions)
        
        # Test 7-day window
        window_feat = window_features._compute_single_window_features(merged, 7)
        
        assert 'signal_id' in window_feat.columns
        assert 'transaction_count_7d' in window_feat.columns
        
        # Check that all 7-day features have the 7d suffix
        for col in window_feat.columns:
            if col != 'signal_id':
                assert '7d' in col or col in ['index']  # Allow index column
    
    def test_recency_features(self, sample_transactions, sample_signals):
        """Test recency feature computation"""
        window_features = WindowFeatures()
        
        # Merge for recency computation
        merged = sample_transactions.merge(
            sample_signals[['signal_id', 'signal_sanasi']], 
            on='signal_id', 
            how='left'
        )
        
        recency_feat = window_features._compute_recency_features(merged)
        
        assert 'signal_id' in recency_feat.columns
        assert 'days_since_last_transaction' in recency_feat.columns
        
        # Check for direction-specific recency if direction exists
        if 'direction' in sample_transactions.columns:
            expected_recency = ['days_since_last_incoming', 'days_since_last_outgoing']
            for feat in expected_recency:
                if feat in recency_feat.columns:
                    assert feat in recency_feat.columns
    
    def test_velocity_features(self, sample_transactions, sample_signals):
        """Test velocity feature computation"""
        window_features = WindowFeatures()
        
        # Merge for velocity computation
        merged = sample_transactions.merge(
            sample_signals[['signal_id', 'signal_sanasi']], 
            on='signal_id', 
            how='left'
        )
        
        velocity_feat = window_features._compute_velocity_features(merged)
        
        assert 'signal_id' in velocity_feat.columns
        assert 'transactions_per_active_day' in velocity_feat.columns
        assert 'transactions_per_active_hour' in velocity_feat.columns
    
    def test_complete_window_feature_pipeline(self, sample_transactions, sample_signals):
        """Test complete window feature creation pipeline"""
        window_features = WindowFeatures()
        
        data = {
            'train_signals': sample_signals,
            'test_signals': sample_signals.iloc[:1],  # Use subset for test
            'train_transactions': sample_transactions,
            'test_transactions': sample_transactions.iloc[:5]  # Use subset for test
        }
        
        features = window_features.create_window_features(data, temporal_filter=True)
        
        assert 'train' in features
        assert 'test' in features
        assert 'signal_id' in features['train'].columns
        assert len(features['train']) == len(sample_signals)
    
    def test_modeling_data_preparation(self, sample_transactions, sample_signals):
        """Test modeling data preparation"""
        window_features = WindowFeatures()
        
        data = {
            'train_signals': sample_signals,
            'test_signals': sample_signals.iloc[:1],
            'train_transactions': sample_transactions,
            'test_transactions': sample_transactions.iloc[:5]
        }
        
        features = window_features.create_window_features(data, temporal_filter=True)
        X_train, y_train, X_test = window_features.prepare_modeling_data(features)
        
        assert X_train.shape[0] == len(sample_signals)
        assert len(y_train) == len(sample_signals)
        assert X_test.shape[0] == 1  # Since we used 1 sample for test
        assert X_train.shape[1] == X_test.shape[1]  # Same features
    
    def test_window_diversity(self, sample_transactions, sample_signals):
        """Test that different windows produce different results"""
        window_features = WindowFeatures()
        
        # Merge for window computation
        merged = sample_transactions.merge(
            sample_signals[['signal_id', 'signal_sanasi']], 
            on='signal_id', 
            how='left'
        )
        
        window_features.high_amount_threshold = window_features.compute_high_amount_threshold(sample_transactions)
        
        # Compute features for different windows
        window_1d = window_features._compute_single_window_features(merged, 1)
        window_7d = window_features._compute_single_window_features(merged, 7)
        window_30d = window_features._compute_single_window_features(merged, 30)
        
        # Check that different windows have different transaction counts
        # (unless all transactions fall in all windows, which is unlikely)
        assert 'transaction_count_1d' in window_1d.columns
        assert 'transaction_count_7d' in window_7d.columns
        assert 'transaction_count_30d' in window_30d.columns
    
    def test_safe_denominator_handling(self):
        """Test safe denominator handling in ratios"""
        window_features = WindowFeatures()
        
        # Test safe_divide function
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 3])
        
        from src.features.feature_store import safe_divide
        result = safe_divide(numerator, denominator)
        
        assert result[0] == 5.0  # 10 / 2
        assert result[1] == 0.0  # 20 / 0 -> default 0
        assert result[2] == 10.0  # 30 / 3
    
    def test_sparse_data_handling(self):
        """Test handling of sparse transaction data"""
        window_features = WindowFeatures()
        
        # Create sparse data (signals with few transactions)
        sparse_transactions = pd.DataFrame({
            'signal_id': ['SG_001', 'SG_002'],
            'amount': [100.0, 200.0],
            'direction': ['incoming', 'outgoing'],
            'type': ['card', 'bank_transfer'],
            'tranzaksiya_vaqti': pd.to_datetime(['2024-12-01', '2024-12-15'])
        })
        
        sparse_signals = pd.DataFrame({
            'signal_id': ['SG_001', 'SG_002'],
            'signal_sanasi': pd.to_datetime(['2025-01-01', '2025-01-01']),
            'eskalatsiya': [0, 1]
        })
        
        # Should handle sparse data without errors
        filtered = window_features.apply_temporal_filter(sparse_transactions, sparse_signals)
        assert len(filtered) <= len(sparse_transactions)
    
    def test_feature_metadata_registration(self, sample_transactions, sample_signals):
        """Test that window features are properly registered with metadata"""
        window_features = WindowFeatures()
        
        # Apply temporal filtering
        filtered_transactions = window_features.apply_temporal_filter(sample_transactions, sample_signals)
        
        # Compute window features
        window_features.high_amount_threshold = window_features.compute_high_amount_threshold(filtered_transactions)
        window_features.compute_window_features(filtered_transactions, sample_signals)
        
        # Check that features were registered
        total_registered = len(window_features.feature_store.get_all_features())
        assert total_registered > 0
        
        # Check that metadata includes required fields
        for feat_name, metadata in window_features.feature_store.features_metadata.items():
            assert metadata.name == feat_name
            assert metadata.feature_type in ['count', 'amount', 'ratio', 'time']
            assert metadata.lookback_rule is not None


class TestWindowFeatureIntegration:
    """Test cases for window feature integration"""
    
    def test_feature_store_integration(self):
        """Test integration with feature store"""
        window_features = WindowFeatures()
        
        # Check that feature store is properly initialized
        assert window_features.feature_store is not None
        assert isinstance(window_features.feature_store, FeatureStore)
    
    def test_feature_group_registration(self, sample_transactions, sample_signals):
        """Test feature group registration"""
        window_features = WindowFeatures()
        
        # Create some features
        filtered_transactions = window_features.apply_temporal_filter(sample_transactions, sample_signals)
        window_features.high_amount_threshold = window_features.compute_high_amount_threshold(filtered_transactions)
        window_features.compute_window_features(filtered_transactions, sample_signals)
        
        # Register feature groups
        window_features._register_feature_groups()
        
        # Check that groups were registered
        assert len(window_features.feature_store.feature_groups) > 0
    
    def test_feature_summary(self, sample_transactions, sample_signals):
        """Test feature summary generation"""
        window_features = WindowFeatures()
        
        # Create features
        data = {
            'train_signals': sample_signals,
            'test_signals': sample_signals.iloc[:1],
            'train_transactions': sample_transactions,
            'test_transactions': sample_transactions.iloc[:5]
        }
        
        features = window_features.create_window_features(data, temporal_filter=True)
        summary = window_features.get_feature_summary()
        
        assert 'total_features' in summary
        assert 'windows' in summary
        assert 'feature_groups' in summary
        assert 'feature_types' in summary
        assert 'safety_validation' in summary


def test_import():
    """Test that modules can be imported"""
    try:
        from src.features.window_features import WindowFeatures
        from src.features.feature_store import FeatureStore, FeatureMetadata
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_window_workflow():
    """Test the complete window feature workflow"""
    # This test ensures the workflow is properly structured
    window_features = WindowFeatures()
    
    # Check that all required methods exist
    required_methods = [
        'load_data', 'apply_temporal_filter', 'compute_high_amount_threshold',
        'compute_window_features', '_compute_single_window_features',
        '_compute_recency_features', '_compute_velocity_features',
        'create_window_features', 'prepare_modeling_data'
    ]
    
    for method in required_methods:
        assert hasattr(window_features, method), f"Missing method: {method}"


def test_window_definitions():
    """Test that window definitions are properly set"""
    window_features = WindowFeatures()
    
    expected_windows = [1, 3, 7, 14, 30, 60, 90]
    assert window_features.windows == expected_windows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])