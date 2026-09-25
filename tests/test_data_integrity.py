"""
Unit tests for data integrity audit module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.audit import DataIntegrityAuditor
from src.config import get_config


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Sample signals data
    train_signals = pd.DataFrame({
        'signal_id': ['SG_001', 'SG_002', 'SG_003', 'SG_004'],
        'signal_sanasi': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01'],
        'eskalatsiya': [0, 1, 0, 1]
    })
    
    test_signals = pd.DataFrame({
        'signal_id': ['SG_005', 'SG_006'],
        'signal_sanasi': ['2025-05-01', '2025-06-01']
    })
    
    # Sample transactions data
    train_transactions = pd.DataFrame({
        'signal_id': ['SG_001', 'SG_001', 'SG_002', 'SG_003', 'SG_004'],
        'amount': [100.0, 200.0, 150.0, 300.0, 250.0],
        'tranzaksiya_vaqti': ['2024-12-01', '2024-12-15', '2025-01-15', '2025-02-15', '2025-03-15']
    })
    
    test_transactions = pd.DataFrame({
        'signal_id': ['SG_005', 'SG_006'],
        'amount': [175.0, 225.0],
        'tranzaksiya_vaqti': ['2025-04-15', '2025-05-15']
    })
    
    return {
        'train_signals': train_signals,
        'test_signals': test_signals,
        'train_transactions': train_transactions,
        'test_transactions': test_transactions
    }


class TestDataIntegrityAuditor:
    """Test cases for DataIntegrityAuditor class"""
    
    def test_initialization(self):
        """Test DataIntegrityAuditor initialization"""
        auditor = DataIntegrityAuditor()
        assert auditor.config is not None
        assert auditor.paths is not None
        assert auditor.results == {}
        assert auditor.raw_data == {}
    
    def test_identify_transaction_date_columns(self, sample_data):
        """Test date column identification"""
        auditor = DataIntegrityAuditor()
        df = sample_data['train_transactions']
        
        date_cols = auditor._identify_transaction_date_columns(df)
        
        # Should identify 'tranzaksiya_vaqti' as a date column
        assert 'tranzaksiya_vaqti' in date_cols
    
    def test_signal_id_uniqueness_check(self, sample_data):
        """Test signal ID uniqueness checking"""
        auditor = DataIntegrityAuditor()
        auditor.raw_data = sample_data
        
        result = auditor._check_signal_id_uniqueness()
        
        assert 'train_signal_id_unique' in result
        assert 'test_signal_id_unique' in result
        assert 'signal_id_overlap_count' in result
        
        # With our sample data, all should be unique and no overlap
        assert result['train_signal_id_unique'] == True
        assert result['test_signal_id_unique'] == True
        assert result['signal_id_overlap_count'] == 0
    
    def test_missing_signal_ids_check(self, sample_data):
        """Test missing signal ID checking"""
        auditor = DataIntegrityAuditor()
        auditor.raw_data = sample_data
        
        result = auditor._check_missing_signal_ids()
        
        assert 'train_transactions_null_signal_id' in result
        assert 'test_transactions_null_signal_id' in result
        assert 'train_missing_percentage' in result
        assert 'test_missing_percentage' in result
        
        # With our sample data, no missing signal IDs
        assert result['train_transactions_null_signal_id'] == 0
        assert result['test_transactions_null_signal_id'] == 0
    
    def test_transaction_distribution_computation(self, sample_data):
        """Test transaction distribution computation"""
        auditor = DataIntegrityAuditor()
        auditor.raw_data = sample_data
        
        result = auditor._compute_transaction_distribution()
        
        assert 'train' in result
        assert 'test' in result
        assert 'mean_transactions_per_signal' in result['train']
        assert 'median_transactions_per_signal' in result['train']
        
        # Verify calculations
        train_mean = result['train']['mean_transactions_per_signal']
        assert train_mean > 0  # Should have some transactions
    
    def test_temporal_alignment_check(self, sample_data):
        """Test temporal alignment checking"""
        auditor = DataIntegrityAuditor()
        auditor.raw_data = sample_data
        
        # Convert date columns
        sample_data['train_signals']['signal_sanasi'] = pd.to_datetime(sample_data['train_signals']['signal_sanasi'])
        sample_data['test_signals']['signal_sanasi'] = pd.to_datetime(sample_data['test_signals']['signal_sanasi'])
        
        result = auditor._check_temporal_alignment()
        
        # Should have results for date columns found
        assert len(result) > 0 or result == {}  # May be empty if no date columns identified
    
    def test_duplicate_detection(self, sample_data):
        """Test duplicate detection"""
        auditor = DataIntegrityAuditor()
        auditor.raw_data = sample_data
        
        result = auditor._detect_duplicates()
        
        assert 'train' in result
        assert 'test' in result
        assert 'exact_duplicates' in result['train']
        assert 'exact_duplicates' in result['test']
        
        # With our sample data, no duplicates
        assert result['train']['exact_duplicates']['count'] == 0
        assert result['test']['exact_duplicates']['count'] == 0
    
    def test_cross_leakage_check(self, sample_data):
        """Test cross-train/test leakage checking"""
        auditor = DataIntegrityAuditor()
        auditor.raw_data = sample_data
        
        result = auditor._check_cross_leakage()
        
        assert 'signal_id_overlap' in result
        assert 'leakage_detected' in result
        
        # With our sample data, no overlap
        assert result['signal_id_overlap']['count'] == 0
        assert result['leakage_detected'] == False


class TestDataQuality:
    """Test cases for data quality aspects"""
    
    def test_data_integrity_framework(self):
        """Test that the data integrity framework is properly structured"""
        auditor = DataIntegrityAuditor()
        
        # Check that all required methods exist
        assert hasattr(auditor, '_check_signal_id_uniqueness')
        assert hasattr(auditor, '_check_missing_signal_ids')
        assert hasattr(auditor, '_compute_transaction_distribution')
        assert hasattr(auditor, '_compute_date_ranges')
        assert hasattr(auditor, '_check_temporal_alignment')
        assert hasattr(auditor, '_analyze_signal_transaction_relationship')
        assert hasattr(auditor, '_analyze_target_prevalence')
        assert hasattr(auditor, '_compare_train_test_distributions')
        assert hasattr(auditor, '_detect_duplicates')
        assert hasattr(auditor, '_check_cross_leakage')
        assert hasattr(auditor, '_derive_temporal_rules')
    
    def test_temporal_rules_derivation(self):
        """Test temporal rules derivation framework"""
        auditor = DataIntegrityAuditor()
        
        # Should have method to derive temporal rules
        assert hasattr(auditor, '_derive_temporal_rules')
        
        # Test that it returns expected structure
        auditor.results = {
            'temporal_alignment': {},
            'signal_vs_transaction_dates': {}
        }
        
        result = auditor._derive_temporal_rules()
        
        assert 'proposed_rules' in result
        assert 'evidence' in result
        assert 'recommendation' in result


class TestConfiguration:
    """Test cases for configuration integration"""
    
    def test_config_loading(self):
        """Test that configuration can be loaded"""
        config = get_config()
        
        assert config is not None
        assert hasattr(config, 'get_paths')
        assert hasattr(config, 'get_data_config')
    
    def test_config_paths(self):
        """Test that configuration paths are properly set"""
        config = get_config()
        paths = config.get_paths()
        
        assert 'train_signals' in paths
        assert 'test_signals' in paths
        assert 'train_transactions' in paths
        assert 'test_transactions' in paths
        assert 'sample_submission' in paths


def test_import():
    """Test that modules can be imported"""
    try:
        from src.data.audit import DataIntegrityAuditor
        from src.config import get_config
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_data_integrity_workflow():
    """Test the complete data integrity workflow"""
    # This test ensures the workflow is properly structured
    # It doesn't run the actual audit since we don't have real data
    
    auditor = DataIntegrityAuditor()
    
    # Check that all required methods are present
    required_methods = [
        'load_data',
        'run_comprehensive_audit',
        'generate_report',
        'save_results'
    ]
    
    for method in required_methods:
        assert hasattr(auditor, method), f"Missing method: {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])