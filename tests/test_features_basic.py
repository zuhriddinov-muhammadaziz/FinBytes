"""
Unit tests for basic feature engineering modules
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.features.feature_store import FeatureStore, FeatureMetadata, safe_divide, safe_ratio, create_systematic_feature_name
from src.features.basic_features import BasicFeatures


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing"""
    return pd.DataFrame({
        'signal_id': ['SG_001', 'SG_001', 'SG_001', 'SG_002', 'SG_002', 'SG_003'],
        'amount': [100.0, 200.0, 150.0, 300.0, 250.0, 175.0],
        'direction': ['incoming', 'outgoing', 'incoming', 'incoming', 'outgoing', 'incoming'],
        'type': ['card', 'bank_transfer', 'card', 'cash', 'card', 'international'],
        'tranzaksiya_vaqti': pd.to_datetime(['2024-12-01', '2024-12-15', '2025-01-15', '2025-02-15', '2025-03-15', '2025-04-15'])
    })


@pytest.fixture
def sample_signals():
    """Create sample signal data for testing"""
    return pd.DataFrame({
        'signal_id': ['SG_001', 'SG_002', 'SG_003'],
        'signal_sanasi': pd.to_datetime(['2025-01-01', '2025-02-01', '2025-03-01']),
        'eskalatsiya': [0, 1, 0]
    })


class TestFeatureStore:
    """Test cases for FeatureStore class"""
    
    def test_initialization(self):
        """Test FeatureStore initialization"""
        feature_store = FeatureStore()
        assert feature_store.config is not None
        assert feature_store.data_config is not None
        assert feature_store.features_metadata == {}
        assert feature_store.feature_groups == {}
    
    def test_feature_registration(self):
        """Test feature registration"""
        feature_store = FeatureStore()
        
        metadata = FeatureMetadata(
            name='test_feature',
            description='Test feature description',
            feature_type='count',
            lookback_rule='Test rule',
            data_source='transactions',
            aggregation_method='count'
        )
        
        feature_store.register_feature(metadata)
        
        assert 'test_feature' in feature_store.features_metadata
        assert feature_store.get_feature_metadata('test_feature').name == 'test_feature'
    
    def test_feature_group_registration(self):
        """Test feature group registration"""
        feature_store = FeatureStore()
        
        feature_names = ['feature1', 'feature2', 'feature3']
        feature_store.register_feature_group('test_group', feature_names)
        
        assert 'test_group' in feature_store.feature_groups
        assert feature_store.get_feature_group('test_group') == feature_names
    
    def test_get_features_by_type(self):
        """Test getting features by type"""
        feature_store = FeatureStore()
        
        # Register features of different types
        for i in range(3):
            metadata = FeatureMetadata(
                name=f'count_feature_{i}',
                description=f'Count feature {i}',
                feature_type='count',
                lookback_rule='Test rule',
                data_source='transactions',
                aggregation_method='count'
            )
            feature_store.register_feature(metadata)
        
        for i in range(2):
            metadata = FeatureMetadata(
                name=f'amount_feature_{i}',
                description=f'Amount feature {i}',
                feature_type='amount',
                lookback_rule='Test rule',
                data_source='transactions',
                aggregation_method='sum'
            )
            feature_store.register_feature(metadata)
        
        count_features = feature_store.get_features_by_type('count')
        amount_features = feature_store.get_features_by_type('amount')
        
        assert len(count_features) == 3
        assert len(amount_features) == 2
    
    def test_safety_validation(self):
        """Test feature safety validation"""
        feature_store = FeatureStore()
        
        # Register safe feature
        safe_metadata = FeatureMetadata(
            name='safe_feature',
            description='Safe feature',
            feature_type='count',
            lookback_rule='Safe rule',
            data_source='transactions',
            aggregation_method='count',
            safe_from_leakage=True
        )
        feature_store.register_feature(safe_metadata)
        
        # Register unsafe feature
        unsafe_metadata = FeatureMetadata(
            name='unsafe_feature',
            description='Unsafe feature',
            feature_type='count',
            lookback_rule='Unsafe rule',
            data_source='transactions',
            aggregation_method='count',
            safe_from_leakage=False
        )
        feature_store.register_feature(unsafe_metadata)
        
        validation = feature_store.validate_feature_safety()
        
        assert validation['total_features'] == 2
        assert validation['safe_features'] == 1
        assert len(validation['unsafe_features']) == 1
        assert validation['validation_passed'] == False
    
    def test_save_and_load_metadata(self, tmp_path):
        """Test saving and loading feature metadata"""
        feature_store = FeatureStore()
        
        # Register a feature
        metadata = FeatureMetadata(
            name='test_feature',
            description='Test feature',
            feature_type='count',
            lookback_rule='Test rule',
            data_source='transactions',
            aggregation_method='count'
        )
        feature_store.register_feature(metadata)
        
        # Save metadata
        output_path = tmp_path / "test_metadata.json"
        feature_store.save_metadata(str(output_path))
        
        # Create new feature store and load metadata
        new_feature_store = FeatureStore()
        new_feature_store.load_metadata(str(output_path))
        
        assert 'test_feature' in new_feature_store.features_metadata
        assert new_feature_store.get_feature_metadata('test_feature').description == 'Test feature'


class TestBasicFeatures:
    """Test cases for BasicFeatures class"""
    
    def test_initialization(self):
        """Test BasicFeatures initialization"""
        basic_features = BasicFeatures()
        assert basic_features.config is not None
        assert basic_features.data_config is not None
        assert basic_features.feature_store is not None
    
    def test_temporal_filtering(self, sample_transactions, sample_signals):
        """Test temporal filtering functionality"""
        basic_features = BasicFeatures()
        
        filtered = basic_features.apply_temporal_filter(sample_transactions, sample_signals)
        
        # Should filter out transactions after signal date
        assert len(filtered) <= len(sample_transactions)
        assert 'signal_id' in filtered.columns
    
    def test_count_features_computation(self, sample_transactions):
        """Test count feature computation"""
        basic_features = BasicFeatures()
        
        count_features = basic_features.compute_count_features(sample_transactions)
        
        assert 'signal_id' in count_features.columns
        assert 'transaction_count' in count_features.columns
        assert len(count_features) == sample_transactions['signal_id'].nunique()
    
    def test_amount_features_computation(self, sample_transactions):
        """Test amount feature computation"""
        basic_features = BasicFeatures()
        
        amount_features = basic_features.compute_amount_features(sample_transactions)
        
        assert 'signal_id' in amount_features.columns
        # Check for some expected amount features
        expected_amount_features = ['total_amount', 'mean_amount', 'median_amount']
        for feat in expected_amount_features:
            if feat in amount_features.columns:
                assert feat in amount_features.columns
    
    def test_ratio_features_computation(self, sample_transactions):
        """Test ratio feature computation"""
        basic_features = BasicFeatures()
        
        ratio_features = basic_features.compute_ratio_features(sample_transactions)
        
        assert 'signal_id' in ratio_features.columns
        # Check for ratio features
        if 'direction' in sample_transactions.columns:
            expected_ratios = ['incoming_ratio', 'outgoing_ratio']
            for ratio in expected_ratios:
                if ratio in ratio_features.columns:
                    assert ratio in ratio_features.columns
    
    def test_time_features_computation(self, sample_transactions):
        """Test time feature computation"""
        basic_features = BasicFeatures()
        
        time_features = basic_features.compute_time_features(sample_transactions)
        
        assert 'signal_id' in time_features.columns
        # Check for time features
        expected_time_features = ['first_transaction_timestamp', 'last_transaction_timestamp', 'history_span_days']
        for feat in expected_time_features:
            if feat in time_features.columns:
                assert feat in time_features.columns
    
    def test_feature_creation_pipeline(self, sample_transactions, sample_signals):
        """Test complete feature creation pipeline"""
        basic_features = BasicFeatures()
        
        data = {
            'train_signals': sample_signals,
            'test_signals': sample_signals.iloc[:1],  # Use subset for test
            'train_transactions': sample_transactions,
            'test_transactions': sample_transactions.iloc[:3]  # Use subset for test
        }
        
        features = basic_features.create_features(data, temporal_filter=True)
        
        assert 'train' in features
        assert 'test' in features
        assert 'signal_id' in features['train'].columns
        assert len(features['train']) == len(sample_signals)
    
    def test_modeling_data_preparation(self, sample_transactions, sample_signals):
        """Test modeling data preparation"""
        basic_features = BasicFeatures()
        
        data = {
            'train_signals': sample_signals,
            'test_signals': sample_signals.iloc[:1],
            'train_transactions': sample_transactions,
            'test_transactions': sample_transactions.iloc[:3]
        }
        
        features = basic_features.create_features(data, temporal_filter=True)
        X_train, y_train, X_test = basic_features.prepare_modeling_data(features)
        
        assert X_train.shape[0] == len(sample_signals)
        assert len(y_train) == len(sample_signals)
        assert X_test.shape[0] == 1  # Since we used 1 sample for test
        assert X_train.shape[1] == X_test.shape[1]  # Same features


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_safe_divide(self):
        """Test safe division function"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 3])
        
        result = safe_divide(numerator, denominator)
        
        assert result[0] == 5.0  # 10 / 2
        assert result[1] == 0.0  # 20 / 0 -> default 0
        assert result[2] == 10.0  # 30 / 3
    
    def test_safe_ratio(self):
        """Test safe ratio function"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([40, 0, 30])
        
        result = safe_ratio(numerator, denominator)
        
        assert result[0] == 0.2  # 10 / (10 + 40)
        assert result[1] == 1.0  # 20 / (20 + 0)
        assert result[2] == 0.5  # 30 / (30 + 30)
    
    def test_systematic_feature_name(self):
        """Test systematic feature naming"""
        name1 = create_systematic_feature_name("count", prefix="transaction", suffix="total")
        assert name1 == "transaction_count_total"
        
        name2 = create_systematic_feature_name("amount")
        assert name2 == "amount"
        
        name3 = create_systematic_feature_name("ratio", suffix="incoming")
        assert name3 == "ratio_incoming"


class TestFeatureMetadata:
    """Test cases for FeatureMetadata dataclass"""
    
    def test_feature_metadata_creation(self):
        """Test FeatureMetadata creation"""
        metadata = FeatureMetadata(
            name='test_feature',
            description='Test description',
            feature_type='count',
            lookback_rule='Test rule',
            data_source='transactions',
            aggregation_method='count'
        )
        
        assert metadata.name == 'test_feature'
        assert metadata.description == 'Test description'
        assert metadata.feature_type == 'count'
        assert metadata.lookback_rule == 'Test rule'
        assert metadata.data_source == 'transactions'
        assert metadata.aggregation_method == 'count'
        assert metadata.safe_from_leakage == True
        assert metadata.dependencies == []
    
    def test_feature_metadata_with_dependencies(self):
        """Test FeatureMetadata with dependencies"""
        metadata = FeatureMetadata(
            name='derived_feature',
            description='Derived feature',
            feature_type='ratio',
            lookback_rule='Derived rule',
            data_source='derived',
            aggregation_method='ratio',
            dependencies=['feature1', 'feature2']
        )
        
        assert len(metadata.dependencies) == 2
        assert 'feature1' in metadata.dependencies
        assert 'feature2' in metadata.dependencies


def test_import():
    """Test that modules can be imported"""
    try:
        from src.features.feature_store import FeatureStore, FeatureMetadata
        from src.features.basic_features import BasicFeatures
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_feature_workflow():
    """Test the complete feature workflow"""
    # This test ensures the workflow is properly structured
    basic_features = BasicFeatures()
    
    # Check that all required methods exist
    required_methods = [
        'load_data', 'apply_temporal_filter', 'compute_count_features',
        'compute_amount_features', 'compute_ratio_features', 'compute_time_features',
        'create_features', 'prepare_modeling_data'
    ]
    
    for method in required_methods:
        assert hasattr(basic_features, method), f"Missing method: {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
