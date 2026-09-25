"""
Comprehensive Data Integrity Audit for WIUT FinTech Hackathon
Performs detailed temporal analysis, data quality checks, and leakage detection
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrityAuditor:
    """Comprehensive data integrity and temporal analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.paths = self.config.get_paths()
        self.results = {}
        self.raw_data = {}
    
    def load_data(self):
        """Load all data files for comprehensive analysis"""
        logger.info("Loading data files for comprehensive audit...")
        
        # Load signals data
        self.raw_data['train_signals'] = pd.read_csv(self.paths['train_signals'])
        self.raw_data['test_signals'] = pd.read_csv(self.paths['test_signals'])
        
        # Load transactions data
        self.raw_data['train_transactions'] = pd.read_parquet(self.paths['train_transactions'])
        self.raw_data['test_transactions'] = pd.read_parquet(self.paths['test_transactions'])
        
        # Load sample submission
        self.raw_data['sample_submission'] = pd.read_csv(self.paths['sample_submission'])
        
        # Convert date columns
        self.raw_data['train_signals']['signal_sanasi'] = pd.to_datetime(self.raw_data['train_signals']['signal_sanasi'])
        self.raw_data['test_signals']['signal_sanasi'] = pd.to_datetime(self.raw_data['test_signals']['signal_sanasi'])
        
        logger.info("All data files loaded successfully")
        return self.raw_data
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run all data integrity checks"""
        logger.info("Starting comprehensive data integrity audit...")
        
        # Load data first
        self.load_data()
        
        # Run all checks
        self.results['signal_id_uniqueness'] = self._check_signal_id_uniqueness()
        self.results['missing_signal_ids'] = self._check_missing_signal_ids()
        self.results['transaction_distribution'] = self._compute_transaction_distribution()
        self.results['date_ranges'] = self._compute_date_ranges()
        self.results['temporal_alignment'] = self._check_temporal_alignment()
        self.results['signal_vs_transaction_dates'] = self._analyze_signal_transaction_relationship()
        self.results['target_prevalence'] = self._analyze_target_prevalence()
        self.results['train_test_distribution'] = self._compare_train_test_distributions()
        self.results['duplicate_detection'] = self._detect_duplicates()
        self.results['cross_leakage'] = self._check_cross_leakage()
        
        # Generate temporal rules
        self.results['temporal_rules'] = self._derive_temporal_rules()
        
        logger.info("Comprehensive audit completed")
        return self.results
    
    def _check_signal_id_uniqueness(self) -> Dict[str, Any]:
        """Check whether signal_id is unique in train_signals and test_signals"""
        logger.info("Checking signal_id uniqueness...")
        
        train_signals = self.raw_data['train_signals']
        test_signals = self.raw_data['test_signals']
        
        # Check uniqueness within each dataset
        train_unique = train_signals['signal_id'].nunique() == len(train_signals)
        test_unique = test_signals['signal_id'].nunique() == len(test_signals)
        
        # Check for overlap between train and test
        train_ids = set(train_signals['signal_id'])
        test_ids = set(test_signals['signal_id'])
        overlap = train_ids.intersection(test_ids)
        
        result = {
            'train_signal_id_unique': train_unique,
            'test_signal_id_unique': test_unique,
            'train_signal_id_count': len(train_signals),
            'train_unique_count': train_signals['signal_id'].nunique(),
            'test_signal_id_count': len(test_signals),
            'test_unique_count': test_signals['signal_id'].nunique(),
            'signal_id_overlap_count': len(overlap),
            'signal_id_overlap_list': list(overlap)[:10] if overlap else [],  # First 10 for inspection
            'signal_id_overlap_exists': len(overlap) > 0
        }
        
        status = "✅" if (train_unique and test_unique and len(overlap) == 0) else "⚠️"
        logger.info(f"{status} Signal ID uniqueness: Train={train_unique}, Test={test_unique}, Overlap={len(overlap)}")
        
        return result
    
    def _check_missing_signal_ids(self) -> Dict[str, Any]:
        """Check whether transaction rows have missing or unknown signal_id values"""
        logger.info("Checking for missing signal_id in transactions...")
        
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        # Check for null signal_id
        train_null = train_trans['signal_id'].isnull().sum()
        test_null = test_trans['signal_id'].isnull().sum()
        
        # Check for empty string signal_id
        train_empty = (train_trans['signal_id'] == '').sum()
        test_empty = (test_trans['signal_id'] == '').sum()
        
        # Check for unknown values (if any standard unknown indicators)
        unknown_indicators = ['unknown', 'NULL', 'NA', 'N/A', 'null']
        train_unknown = 0
        test_unknown = 0
        
        for indicator in unknown_indicators:
            train_unknown += (train_trans['signal_id'].astype(str).str.lower() == indicator.lower()).sum()
            test_unknown += (test_trans['signal_id'].astype(str).str.lower() == indicator.lower()).sum()
        
        result = {
            'train_transactions_null_signal_id': int(train_null),
            'test_transactions_null_signal_id': int(test_null),
            'train_transactions_empty_signal_id': int(train_empty),
            'test_transactions_empty_signal_id': int(test_empty),
            'train_transactions_unknown_signal_id': int(train_unknown),
            'test_transactions_unknown_signal_id': int(test_unknown),
            'total_train_transactions': len(train_trans),
            'total_test_transactions': len(test_trans),
            'train_missing_percentage': (train_null + train_empty + train_unknown) / len(train_trans) * 100,
            'test_missing_percentage': (test_null + test_empty + test_unknown) / len(test_trans) * 100
        }
        
        total_missing = train_null + train_empty + train_unknown + test_null + test_empty + test_unknown
        status = "✅" if total_missing == 0 else "⚠️"
        logger.info(f"{status} Missing signal_id: Train={train_null + train_empty + train_unknown}, Test={test_null + test_empty + test_unknown}")
        
        return result
    
    def _compute_transaction_distribution(self) -> Dict[str, Any]:
        """Compute transaction count distribution per signal"""
        logger.info("Computing transaction distribution per signal...")
        
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        # Count transactions per signal
        train_counts = train_trans.groupby('signal_id').size()
        test_counts = test_trans.groupby('signal_id').size()
        
        # Statistics
        result = {
            'train': {
                'total_signals_with_transactions': len(train_counts),
                'total_transactions': len(train_trans),
                'mean_transactions_per_signal': float(train_counts.mean()),
                'median_transactions_per_signal': float(train_counts.median()),
                'std_transactions_per_signal': float(train_counts.std()),
                'min_transactions_per_signal': int(train_counts.min()),
                'max_transactions_per_signal': int(train_counts.max()),
                'percentiles': {
                    '25': float(train_counts.quantile(0.25)),
                    '50': float(train_counts.quantile(0.50)),
                    '75': float(train_counts.quantile(0.75)),
                    '90': float(train_counts.quantile(0.90)),
                    '95': float(train_counts.quantile(0.95)),
                    '99': float(train_counts.quantile(0.99))
                }
            },
            'test': {
                'total_signals_with_transactions': len(test_counts),
                'total_transactions': len(test_trans),
                'mean_transactions_per_signal': float(test_counts.mean()),
                'median_transactions_per_signal': float(test_counts.median()),
                'std_transactions_per_signal': float(test_counts.std()),
                'min_transactions_per_signal': int(test_counts.min()),
                'max_transactions_per_signal': int(test_counts.max()),
                'percentiles': {
                    '25': float(test_counts.quantile(0.25)),
                    '50': float(test_counts.quantile(0.50)),
                    '75': float(test_counts.quantile(0.75)),
                    '90': float(test_counts.quantile(0.90)),
                    '95': float(test_counts.quantile(0.95)),
                    '99': float(test_counts.quantile(0.99))
                }
            }
        }
        
        logger.info(f"Transaction distribution: Train mean={result['train']['mean_transactions_per_signal']:.1f}, "
                   f"Test mean={result['test']['mean_transactions_per_signal']:.1f}")
        
        return result
    
    def _compute_date_ranges(self) -> Dict[str, Any]:
        """Compute signal date range and transaction timestamp range"""
        logger.info("Computing date ranges...")
        
        train_signals = self.raw_data['train_signals']
        test_signals = self.raw_data['test_signals']
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        # Signal date ranges
        result = {
            'signal_dates': {
                'train': {
                    'min': str(train_signals['signal_sanasi'].min()),
                    'max': str(train_signals['signal_sanasi'].max()),
                    'range_days': (train_signals['signal_sanasi'].max() - train_signals['signal_sanasi'].min()).days
                },
                'test': {
                    'min': str(test_signals['signal_sanasi'].min()),
                    'max': str(test_signals['signal_sanasi'].max()),
                    'range_days': (test_signals['signal_sanasi'].max() - test_signals['signal_sanasi'].min()).days
                }
            }
        }
        
        # Transaction date ranges - identify date columns first
        date_cols = self._identify_transaction_date_columns(train_trans)
        
        result['transaction_dates'] = {}
        for date_col in date_cols:
            # Convert to datetime
            train_trans[date_col] = pd.to_datetime(train_trans[date_col], errors='coerce')
            test_trans[date_col] = pd.to_datetime(test_trans[date_col], errors='coerce')
            
            valid_train_dates = train_trans[date_col].dropna()
            valid_test_dates = test_trans[date_col].dropna()
            
            result['transaction_dates'][date_col] = {
                'train': {
                    'min': str(valid_train_dates.min()) if len(valid_train_dates) > 0 else None,
                    'max': str(valid_train_dates.max()) if len(valid_train_dates) > 0 else None,
                    'range_days': (valid_train_dates.max() - valid_train_dates.min()).days if len(valid_train_dates) > 0 else None,
                    'valid_count': len(valid_train_dates),
                    'null_count': train_trans[date_col].isnull().sum()
                },
                'test': {
                    'min': str(valid_test_dates.min()) if len(valid_test_dates) > 0 else None,
                    'max': str(valid_test_dates.max()) if len(valid_test_dates) > 0 else None,
                    'range_days': (valid_test_dates.max() - valid_test_dates.min()).days if len(valid_test_dates) > 0 else None,
                    'valid_count': len(valid_test_dates),
                    'null_count': test_trans[date_col].isnull().sum()
                }
            }
        
        logger.info(f"Date ranges computed for {len(date_cols)} transaction date columns")
        
        return result
    
    def _identify_transaction_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify date columns in transaction data"""
        date_cols = []
        
        # Check for common date column names
        date_keywords = ['date', 'time', 'vaqti', 'sanasi', 'timestamp', 'day', 'month', 'year']
        
        for col in df.columns:
            # Check column name
            if any(keyword in col.lower() for keyword in date_keywords):
                date_cols.append(col)
                continue
            
            # Try to convert sample to datetime
            try:
                sample_conversion = pd.to_datetime(df[col].head(100), errors='coerce')
                if sample_conversion.notna().sum() > 50:  # At least 50% valid
                    date_cols.append(col)
            except:
                continue
        
        return date_cols
    
    def _check_temporal_alignment(self) -> Dict[str, Any]:
        """Check whether all transaction timestamps belong to corresponding signal's historical period"""
        logger.info("Checking temporal alignment between transactions and signals...")
        
        train_signals = self.raw_data['train_signals']
        test_signals = self.raw_data['test_signals']
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        # Identify transaction date columns
        date_cols = self._identify_transaction_date_columns(train_trans)
        
        result = {}
        
        for date_col in date_cols:
            # Convert to datetime
            train_trans[date_col] = pd.to_datetime(train_trans[date_col], errors='coerce')
            test_trans[date_col] = pd.to_datetime(test_trans[date_col], errors='coerce')
            
            # Merge transactions with signals
            train_merged = train_trans.merge(train_signals[['signal_id', 'signal_sanasi']], on='signal_id', how='left')
            test_merged = test_trans.merge(test_signals[['signal_id', 'signal_sanasi']], on='signal_id', how='left')
            
            # Check temporal alignment
            train_merged['is_future'] = train_merged[date_col] > train_merged['signal_sanasi']
            test_merged['is_future'] = test_merged[date_col] > test_merged['signal_sanasi']
            
            train_future_count = train_merged['is_future'].sum()
            test_future_count = test_merged['is_future'].sum()
            
            result[date_col] = {
                'train': {
                    'total_transactions': len(train_merged),
                    'future_transactions': int(train_future_count),
                    'future_percentage': float(train_future_count / len(train_merged) * 100),
                    'past_transactions': int(len(train_merged) - train_future_count),
                    'past_percentage': float((len(train_merged) - train_future_count) / len(train_merged) * 100)
                },
                'test': {
                    'total_transactions': len(test_merged),
                    'future_transactions': int(test_future_count),
                    'future_percentage': float(test_future_count / len(test_merged) * 100),
                    'past_transactions': int(len(test_merged) - test_future_count),
                    'past_percentage': float((len(test_merged) - test_future_count) / len(test_merged) * 100)
                }
            }
            
            status = "⚠️" if (train_future_count > 0 or test_future_count > 0) else "✅"
            logger.info(f"{status} {date_col}: Train future={train_future_count}, Test future={test_future_count}")
        
        return result
    
    def _analyze_signal_transaction_relationship(self) -> Dict[str, Any]:
        """Explicitly investigate relationship between signal_sanasi and tranzaksiya_vaqti"""
        logger.info("Analyzing signal_sanasi vs tranzaksiya_vaqti relationship...")
        
        train_signals = self.raw_data['train_signals']
        train_trans = self.raw_data['train_transactions']
        
        result = {}
        
        # Check if tranzaksiya_vaqti exists
        if 'tranzaksiya_vaqti' in train_trans.columns:
            # Convert to datetime
            train_trans['tranzaksiya_vaqti'] = pd.to_datetime(train_trans['tranzaksiya_vaqti'], errors='coerce')
            
            # Merge with signals
            merged = train_trans.merge(train_signals[['signal_id', 'signal_sanasi']], on='signal_id', how='left')
            
            # Calculate time differences
            merged['time_difference'] = merged['tranzaksiya_vaqti'] - merged['signal_sanasi']
            merged['time_difference_days'] = merged['time_difference'].dt.days
            
            # Analyze distribution
            valid_diffs = merged['time_difference_days'].dropna()
            
            result['tranzaksiya_vaqti_analysis'] = {
                'exists': True,
                'sample_size': len(valid_diffs),
                'mean_difference_days': float(valid_diffs.mean()),
                'median_difference_days': float(valid_diffs.median()),
                'std_difference_days': float(valid_diffs.std()),
                'min_difference_days': int(valid_diffs.min()),
                'max_difference_days': int(valid_diffs.max()),
                'percent_negative': float((valid_diffs < 0).sum() / len(valid_diffs) * 100),
                'percent_positive': float((valid_diffs > 0).sum() / len(valid_diffs) * 100),
                'percent_zero': float((valid_diffs == 0).sum() / len(valid_diffs) * 100),
                'date_ranges': {
                    'signal_dates': {
                        'min': str(merged['signal_sanasi'].min()),
                        'max': str(merged['signal_sanasi'].max())
                    },
                    'transaction_dates': {
                        'min': str(merged['tranzaksiya_vaqti'].min()),
                        'max': str(merged['tranzaksiya_vaqti'].max())
                    }
                }
            }
            
            # Time window analysis
            result['tranzaksiya_vaqti_analysis']['time_windows'] = {
                'transactions_7_days_before_signal': int((valid_diffs >= -7).sum()),
                'transactions_30_days_before_signal': int((valid_diffs >= -30).sum()),
                'transactions_90_days_before_signal': int((valid_diffs >= -90).sum()),
                'transactions_7_days_after_signal': int((valid_diffs <= 7).sum()),
                'transactions_30_days_after_signal': int((valid_diffs <= 30).sum()),
                'transactions_90_days_after_signal': int((valid_diffs <= 90).sum())
            }
            
        else:
            result['tranzaksiya_vaqti_analysis'] = {
                'exists': False,
                'message': 'tranzaksiya_vaqti column not found in transaction data'
            }
        
        logger.info(f"Signal-transaction relationship analysis completed")
        
        return result
    
    def _derive_temporal_rules(self) -> Dict[str, Any]:
        """Derive plausible temporal rules for historical transactions"""
        logger.info("Deriving temporal rules for historical transactions...")
        
        # This will be populated based on the actual analysis
        # For now, we'll create a framework that gets populated after analysis
        
        result = {
            'proposed_rules': [],
            'evidence': {},
            'recommendation': None
        }
        
        # Get temporal alignment results
        if 'temporal_alignment' in self.results:
            for date_col, alignment_data in self.results['temporal_alignment'].items():
                train_future_pct = alignment_data['train']['future_percentage']
                test_future_pct = alignment_data['test']['future_percentage']
                
                rule = {
                    'date_column': date_col,
                    'rule_type': 'strict_no_future' if (train_future_pct == 0 and test_future_pct == 0) else 'allow_some_future',
                    'train_future_percentage': train_future_pct,
                    'test_future_percentage': test_future_pct,
                    'evidence': f"Train has {train_future_pct:.2f}% future transactions, Test has {test_future_pct:.2f}%"
                }
                result['proposed_rules'].append(rule)
        
        # Get signal-transaction relationship
        if 'signal_vs_transaction_dates' in self.results:
            rel_analysis = self.results['signal_vs_transaction_dates'].get('tranzaksiya_vaqti_analysis', {})
            if rel_analysis.get('exists'):
                result['evidence']['time_difference'] = {
                    'mean_days': rel_analysis.get('mean_difference_days'),
                    'median_days': rel_analysis.get('median_difference_days'),
                    'percent_negative': rel_analysis.get('percent_negative')
                }
        
        # Derive recommendation based on evidence
        if result['proposed_rules']:
            has_future = any(rule['train_future_percentage'] > 0 or rule['test_future_percentage'] > 0 
                           for rule in result['proposed_rules'])
            
            if has_future:
                result['recommendation'] = "REQUIRES INVESTIGATION - Future transactions detected"
            else:
                result['recommendation'] = "SAFE - No future transactions detected"
        
        logger.info(f"Temporal rules derived: {result['recommendation']}")
        
        return result
    
    def _analyze_target_prevalence(self) -> Dict[str, Any]:
        """Check target prevalence and whether class balance changes over time"""
        logger.info("Analyzing target prevalence over time...")
        
        train_signals = self.raw_data['train_signals']
        
        # Overall prevalence
        overall_prevalence = train_signals['eskalatsiya'].mean()
        
        # Prevalence over time (by month)
        train_signals['year_month'] = train_signals['signal_sanasi'].dt.to_period('M')
        monthly_prevalence = train_signals.groupby('year_month')['eskalatsiya'].agg(['mean', 'count'])
        
        result = {
            'overall_prevalence': float(overall_prevalence),
            'positive_class_count': int(train_signals['eskalatsiya'].sum()),
            'negative_class_count': int(len(train_signals) - train_signals['eskalatsiya'].sum()),
            'class_ratio': float(train_signals['eskalatsiya'].sum() / (len(train_signals) - train_signals['eskalatsiya'].sum())),
            'monthly_prevalence': {
                str(period): {
                    'prevalence': float(row['mean']),
                    'count': int(row['count'])
                }
                for period, row in monthly_prevalence.iterrows()
            }
        }
        
        # Check for significant changes
        prevalences = monthly_prevalence['mean'].values
        if len(prevalences) > 1:
            prevalence_std = float(np.std(prevalences))
            result['prevalence_stability'] = {
                'std': prevalence_std,
                'min': float(np.min(prevalences)),
                'max': float(np.max(prevalences)),
                'range': float(np.max(prevalences) - np.min(prevalences))
            }
        
        logger.info(f"Target prevalence: {overall_prevalence:.3f} ({result['positive_class_count']} positive)")
        
        return result
    
    def _compare_train_test_distributions(self) -> Dict[str, Any]:
        """Check train/test distribution differences"""
        logger.info("Comparing train/test distributions...")
        
        train_signals = self.raw_data['train_signals']
        test_signals = self.raw_data['test_signals']
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        result = {}
        
        # Signal date distribution
        result['signal_date_distribution'] = {
            'train': {
                'min': str(train_signals['signal_sanasi'].min()),
                'max': str(train_signals['signal_sanasi'].max()),
                'mean_date': str(train_signals['signal_sanasi'].mean())
            },
            'test': {
                'min': str(test_signals['signal_sanasi'].min()),
                'max': str(test_signals['signal_sanasi'].max()),
                'mean_date': str(test_signals['signal_sanasi'].mean())
            }
        }
        
        # Transaction count distribution
        train_counts = train_trans.groupby('signal_id').size()
        test_counts = test_trans.groupby('signal_id').size()
        
        result['transaction_count_distribution'] = {
            'train': {
                'mean': float(train_counts.mean()),
                'median': float(train_counts.median()),
                'std': float(train_counts.std())
            },
            'test': {
                'mean': float(test_counts.mean()),
                'median': float(test_counts.median()),
                'std': float(test_counts.std())
            }
        }
        
        # Transaction type distribution (if type column exists)
        if 'type' in train_trans.columns:
            train_type_dist = train_trans['type'].value_counts(normalize=True).to_dict()
            test_type_dist = test_trans['type'].value_counts(normalize=True).to_dict()
            
            result['transaction_type_distribution'] = {
                'train': train_type_dist,
                'test': test_type_dist
            }
        
        # Transaction direction distribution (if direction column exists)
        if 'direction' in train_trans.columns:
            train_dir_dist = train_trans['direction'].value_counts(normalize=True).to_dict()
            test_dir_dist = test_trans['direction'].value_counts(normalize=True).to_dict()
            
            result['transaction_direction_distribution'] = {
                'train': train_dir_dist,
                'test': test_dir_dist
            }
        
        # Amount indicator distribution (if amount column exists)
        if 'amount' in train_trans.columns:
            train_amount_stats = train_trans['amount'].describe()
            test_amount_stats = test_trans['amount'].describe()
            
            result['amount_distribution'] = {
                'train': {
                    'mean': float(train_amount_stats['mean']),
                    'std': float(train_amount_stats['std']),
                    'min': float(train_amount_stats['min']),
                    'max': float(train_amount_stats['max'])
                },
                'test': {
                    'mean': float(test_amount_stats['mean']),
                    'std': float(test_amount_stats['std']),
                    'min': float(test_amount_stats['min']),
                    'max': float(test_amount_stats['max'])
                }
            }
        
        logger.info("Train/test distribution comparison completed")
        
        return result
    
    def _detect_duplicates(self) -> Dict[str, Any]:
        """Detect exact duplicate transactions and suspicious duplicate groups"""
        logger.info("Detecting duplicate transactions...")
        
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        result = {
            'train': {},
            'test': {}
        }
        
        # Exact duplicates (all columns same)
        train_exact_dups = train_trans.duplicated()
        test_exact_dups = test_trans.duplicated()
        
        result['train']['exact_duplicates'] = {
            'count': int(train_exact_dups.sum()),
            'percentage': float(train_exact_dups.sum() / len(train_trans) * 100)
        }
        
        result['test']['exact_duplicates'] = {
            'count': int(test_exact_dups.sum()),
            'percentage': float(test_exact_dups.sum() / len(test_trans) * 100)
        }
        
        # Suspicious duplicates (same signal_id, amount, and date)
        if 'amount' in train_trans.columns:
            date_cols = self._identify_transaction_date_columns(train_trans)
            if date_cols:
                date_col = date_cols[0]  # Use first date column found
                
                # Create fingerprint for suspicious duplicates
                train_trans['fingerprint'] = (
                    train_trans['signal_id'].astype(str) + '_' +
                    train_trans['amount'].astype(str) + '_' +
                    train_trans[date_col].astype(str)
                )
                
                test_trans['fingerprint'] = (
                    test_trans['signal_id'].astype(str) + '_' +
                    test_trans['amount'].astype(str) + '_' +
                    test_trans[date_col].astype(str)
                )
                
                train_suspicious_dups = train_trans.duplicated(subset=['fingerprint'])
                test_suspicious_dups = test_trans.duplicated(subset=['fingerprint'])
                
                result['train']['suspicious_duplicates'] = {
                    'count': int(train_suspicious_dups.sum()),
                    'percentage': float(train_suspicious_dups.sum() / len(train_trans) * 100),
                    'fingerprint_columns': ['signal_id', 'amount', date_col]
                }
                
                result['test']['suspicious_duplicates'] = {
                    'count': int(test_suspicious_dups.sum()),
                    'percentage': float(test_suspicious_dups.sum() / len(test_trans) * 100),
                    'fingerprint_columns': ['signal_id', 'amount', date_col]
                }
        
        logger.info(f"Duplicate detection: Train exact={result['train']['exact_duplicates']['count']}, "
                   f"Test exact={result['test']['exact_duplicates']['count']}")
        
        return result
    
    def _check_cross_leakage(self) -> Dict[str, Any]:
        """Check whether same transactions appear across train and test"""
        logger.info("Checking for cross-train/test leakage...")
        
        train_trans = self.raw_data['train_transactions']
        test_trans = self.raw_data['test_transactions']
        
        result = {}
        
        # Check for overlapping signal_ids in transactions
        train_signal_ids = set(train_trans['signal_id'])
        test_signal_ids = set(test_trans['signal_id'])
        overlap_signal_ids = train_signal_ids.intersection(test_signal_ids)
        
        result['signal_id_overlap'] = {
            'count': len(overlap_signal_ids),
            'train_unique': len(train_signal_ids),
            'test_unique': len(test_signal_ids),
            'overlap_percentage': float(len(overlap_signal_ids) / min(len(train_signal_ids), len(test_signal_ids)) * 100) if min(len(train_signal_ids), len(test_signal_ids)) else 0.0
        }
        
        # Check for identical transaction rows (if we can create a unique identifier)
        date_cols = self._identify_transaction_date_columns(train_trans)
        if date_cols and 'amount' in train_trans.columns:
            date_col = date_cols[0]
            
            # Create transaction fingerprints
            train_trans = train_trans.copy()
            test_trans = test_trans.copy()
            train_trans['trans_fingerprint'] = (
                train_trans['signal_id'].astype(str) + '_' +
                train_trans['amount'].astype(str) + '_' +
                train_trans[date_col].astype(str)
            )
            
            test_trans['trans_fingerprint'] = (
                test_trans['signal_id'].astype(str) + '_' +
                test_trans['amount'].astype(str) + '_' +
                test_trans[date_col].astype(str)
            )
            
            train_fingerprints = set(train_trans['trans_fingerprint'])
            test_fingerprints = set(test_trans['trans_fingerprint'])
            overlap_fingerprints = train_fingerprints.intersection(test_fingerprints)
            
            result['transaction_fingerprint_overlap'] = {
                'count': len(overlap_fingerprints),
                'train_unique': len(train_fingerprints),
                'test_unique': len(test_fingerprints),
                'overlap_percentage': float(len(overlap_fingerprints) / min(len(train_fingerprints), len(test_fingerprints)) * 100) if min(len(train_fingerprints), len(test_fingerprints)) else 0.0,
                'fingerprint_columns': ['signal_id', 'amount', date_col]
            }
        
        leakage_detected = (result['signal_id_overlap']['count'] > 0 or 
                          ('transaction_fingerprint_overlap' in result and result['transaction_fingerprint_overlap']['count'] > 0))
        
        status = "⚠️ LEAKAGE RISK" if leakage_detected else "✅ NO LEAKAGE"
        logger.info(f"{status} Cross-dataset leakage: Signal ID overlap={result['signal_id_overlap']['count']}")
        
        result['leakage_detected'] = leakage_detected
        
        return result
    
    def generate_report(self) -> str:
        """Generate comprehensive audit report"""
        report = []
        report.append("=" * 80)
        report.append("WIUT FINTECH HACKATHON - COMPREHENSIVE DATA INTEGRITY AUDIT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 40)
        
        # Signal ID uniqueness
        if 'signal_id_uniqueness' in self.results:
            uniqueness = self.results['signal_id_uniqueness']
            report.append(f"Signal ID Uniqueness: Train={uniqueness['train_signal_id_unique']}, "
                         f"Test={uniqueness['test_signal_id_unique']}, "
                         f"Overlap={uniqueness['signal_id_overlap_count']}")
        
        # Missing signal IDs
        if 'missing_signal_ids' in self.results:
            missing = self.results['missing_signal_ids']
            report.append(f"Missing Signal IDs: Train={missing['train_missing_percentage']:.2f}%, "
                         f"Test={missing['test_missing_percentage']:.2f}%")
        
        # Temporal alignment
        if 'temporal_alignment' in self.results:
            has_future = any(alignment['train']['future_percentage'] > 0 or alignment['test']['future_percentage'] > 0
                           for alignment in self.results['temporal_alignment'].values())
            report.append(f"Future Transactions Detected: {has_future}")
        
        # Cross leakage
        if 'cross_leakage' in self.results:
            leakage = self.results['cross_leakage']['leakage_detected']
            report.append(f"Cross-Train/Test Leakage: {leakage}")
        
        report.append("")
        
        # Detailed sections
        for section_name, section_data in self.results.items():
            if section_name == 'temporal_rules':
                continue  # Handle separately
                
            report.append(f"{section_name.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            report.append(json.dumps(section_data, indent=2, default=str))
            report.append("")
        
        # Temporal rules summary
        if 'temporal_rules' in self.results:
            report.append("TEMPORAL RULES ANALYSIS:")
            report.append("-" * 40)
            report.append(f"Recommendation: {self.results['temporal_rules']['recommendation']}")
            report.append("")
            
            if self.results['temporal_rules']['proposed_rules']:
                report.append("Proposed Rules:")
                for rule in self.results['temporal_rules']['proposed_rules']:
                    report.append(f"  - {rule['date_column']}: {rule['rule_type']}")
                    report.append(f"    Evidence: {rule['evidence']}")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str = "artifacts/data_audit.json"):
        """Save audit results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Audit results saved to {output_path}")


def main():
    """Main function to run comprehensive audit"""
    auditor = DataIntegrityAuditor()
    results = auditor.run_comprehensive_audit()
    
    # Generate and print report
    report = auditor.generate_report()
    print(report)
    
    # Save results
    auditor.save_results()
    
    return results


if __name__ == "__main__":
    main()
