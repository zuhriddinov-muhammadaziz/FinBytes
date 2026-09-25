"""
Basic Features for WIUT FinTech Hackathon
Leakage-safe historical transaction aggregates with systematic naming
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from .feature_store import FeatureStore, FeatureMetadata, safe_divide, safe_ratio, create_systematic_feature_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicFeatures:
    """Leakage-safe basic feature engineering for transaction data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.feature_store = FeatureStore(config_path)
        
        # Column mappings based on configuration
        self.signal_id_col = self.data_config['signal_id_column']
        self.signal_date_col = self.data_config['signal_date_column']
        self.target_col = self.data_config['target_column']
        self.transaction_date_col = self.data_config.get('transaction_date_column', 'tranzaksiya_vaqti')
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        logger.info("Loading data for feature engineering...")
        
        paths = self.config.get_paths()
        data = {}
        
        # Load signals
        data['train_signals'] = pd.read_csv(paths['train_signals'])
        data['test_signals'] = pd.read_csv(paths['test_signals'])
        
        # Load transactions
        data['train_transactions'] = pd.read_parquet(paths['train_transactions'])
        data['test_transactions'] = pd.read_parquet(paths['test_transactions'])
        
        # Convert date columns
        for split in ['train', 'test']:
            data[f'{split}_signals'][self.signal_date_col] = pd.to_datetime(
                data[f'{split}_signals'][self.signal_date_col]
            )
            if self.transaction_date_col in data[f'{split}_transactions'].columns:
                data[f'{split}_transactions'][self.transaction_date_col] = pd.to_datetime(
                    data[f'{split}_transactions'][self.transaction_date_col], errors='coerce'
                )
        
        logger.info("Data loaded successfully")
        return data
    
    def apply_temporal_filter(self, transactions: pd.DataFrame, 
                            signals: pd.DataFrame) -> pd.DataFrame:
        """Apply temporal filtering to prevent future information leakage"""
        
        # Merge transactions with signal dates
        merged = transactions.merge(
            signals[[self.signal_id_col, self.signal_date_col]], 
            on=self.signal_id_col, 
            how='left'
        )
        
        # Keep only transactions <= signal date
        if self.transaction_date_col in merged.columns:
            filtered = merged[merged[self.transaction_date_col] <= merged[self.signal_date_col]].copy()
            logger.info(f"Temporal filtering: {len(transactions)} -> {len(filtered)} transactions "
                       f"({len(filtered)/len(transactions)*100:.1f}% retained)")
        else:
            filtered = merged.copy()
            logger.warning("Transaction date column not found, skipping temporal filter")
        
        return filtered
    
    def compute_count_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute count-based features"""
        logger.info("Computing count features...")
        
        features = {}
        
        # Total transaction count
        total_count = transactions.groupby(self.signal_id_col).size()
        features['transaction_count'] = total_count
        
        self.feature_store.register_feature(FeatureMetadata(
            name='transaction_count',
            description='Total number of transactions per signal',
            feature_type='count',
            lookback_rule='All transactions <= signal date',
            data_source='transactions',
            aggregation_method='count'
        ))
        
        # Direction counts (if direction column exists)
        if 'direction' in transactions.columns:
            direction_counts = transactions.groupby(self.signal_id_col)['direction'].value_counts().unstack(fill_value=0)
            
            features['incoming_count'] = direction_counts.get('incoming', 0)
            features['outgoing_count'] = direction_counts.get('outgoing', 0)
            
            self.feature_store.register_feature(FeatureMetadata(
                name='incoming_count',
                description='Number of incoming transactions',
                feature_type='count',
                lookback_rule='Incoming transactions <= signal date',
                data_source='transactions',
                aggregation_method='count'
            ))
            
            self.feature_store.register_feature(FeatureMetadata(
                name='outgoing_count',
                description='Number of outgoing transactions',
                feature_type='count',
                lookback_rule='Outgoing transactions <= signal date',
                data_source='transactions',
                aggregation_method='count'
            ))
        
        # Transaction type counts (if type column exists)
        if 'type' in transactions.columns:
            type_counts = transactions.groupby(self.signal_id_col)['type'].value_counts().unstack(fill_value=0)
            
            # Common transaction types
            common_types = ['card', 'bank_transfer', 'cash', 'international']
            for trans_type in common_types:
                if trans_type in type_counts.columns:
                    feature_name = f'type_{trans_type}_count'
                    features[feature_name] = type_counts[trans_type]
                    
                    self.feature_store.register_feature(FeatureMetadata(
                        name=feature_name,
                        description=f'Number of {trans_type} transactions',
                        feature_type='count',
                        lookback_rule=f'{trans_type} transactions <= signal date',
                        data_source='transactions',
                        aggregation_method='count'
                    ))
        
        # Active days (unique dates with transactions)
        if self.transaction_date_col in transactions.columns:
            transactions['transaction_date'] = transactions[self.transaction_date_col].dt.date
            active_days = transactions.groupby(self.signal_id_col)['transaction_date'].nunique()
            features['active_days'] = active_days
            
            self.feature_store.register_feature(FeatureMetadata(
                name='active_days',
                description='Number of unique days with transactions',
                feature_type='count',
                lookback_rule='Unique transaction dates <= signal date',
                data_source='transactions',
                aggregation_method='nunique'
            ))
        
        feature_df = pd.DataFrame(features).reset_index()
        logger.info(f"Computed {len(features)} count features")
        return feature_df
    
    def compute_amount_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute amount-based features with statistics"""
        logger.info("Computing amount features...")
        
        features = {}
        
        # Identify amount columns
        amount_cols = [col for col in transactions.columns if 'amount' in col.lower()]
        
        if not amount_cols:
            logger.warning("No amount columns found in transactions")
            return pd.DataFrame({self.signal_id_col: transactions[self.signal_id_col].unique()})
        
        amount_col = amount_cols[0]  # Use first amount column found
        
        # Overall amount statistics
        amount_stats = transactions.groupby(self.signal_id_col)[amount_col].agg([
            'sum', 'mean', 'median', 'std', 'min', 'max'
        ])
        
        # Rename columns systematically
        amount_stats.columns = [f'total_amount', 'mean_amount', 'median_amount', 
                              'std_amount', 'min_amount', 'max_amount']
        
        # Add quantiles
        quantiles = transactions.groupby(self.signal_id_col)[amount_col].agg([
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            lambda x: x.quantile(0.90),
            lambda x: x.quantile(0.95)
        ])
        quantiles.columns = ['amount_q25', 'amount_q75', 'amount_q90', 'amount_q95']
        
        features = pd.concat([amount_stats, quantiles], axis=1)
        
        # Register amount features
        amount_feature_names = {
            'total_amount': 'Total sum of transaction amounts',
            'mean_amount': 'Mean transaction amount',
            'median_amount': 'Median transaction amount',
            'std_amount': 'Standard deviation of transaction amounts',
            'min_amount': 'Minimum transaction amount',
            'max_amount': 'Maximum transaction amount',
            'amount_q25': '25th percentile of transaction amounts',
            'amount_q75': '75th percentile of transaction amounts',
            'amount_q90': '90th percentile of transaction amounts',
            'amount_q95': '95th percentile of transaction amounts'
        }
        
        for feat_name, description in amount_feature_names.items():
            if feat_name in features.columns:
                self.feature_store.register_feature(FeatureMetadata(
                    name=feat_name,
                    description=description,
                    feature_type='amount',
                    lookback_rule='Amount statistics from transactions <= signal date',
                    data_source='transactions',
                    aggregation_method='sum' if feat_name == 'total_amount' else feat_name.split('_')[0] if not feat_name.startswith('amount_q') else 'quantile'
                ))
        
        # Separate incoming/outgoing amount statistics if direction exists
        if 'direction' in transactions.columns:
            for direction in ['incoming', 'outgoing']:
                direction_trans = transactions[transactions['direction'] == direction]
                if len(direction_trans) > 0:
                    dir_amount_stats = direction_trans.groupby(self.signal_id_col)[amount_col].agg([
                        'sum', 'mean', 'median'
                    ])
                    dir_amount_stats.columns = [f'{direction}_amount_sum', f'{direction}_amount_mean', f'{direction}_amount_median']
                    
                    # Merge with existing features
                    features = features.merge(dir_amount_stats, left_index=True, right_index=True, how='left')
                    
                    # Register direction-specific amount features
                    for stat in ['sum', 'mean', 'median']:
                        feat_name = f'{direction}_amount_{stat}'
                        self.feature_store.register_feature(FeatureMetadata(
                            name=feat_name,
                            description=f'{stat.capitalize()} of {direction} transaction amounts',
                            feature_type='amount',
                            lookback_rule=f'{direction.capitalize()} transaction amounts <= signal date',
                            data_source='transactions',
                            aggregation_method=stat
                        ))
        
        feature_df = features.reset_index()
        logger.info(f"Computed {len(features.columns)} amount features")
        return feature_df
    
    def compute_ratio_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute ratio-based features with safe denominator handling"""
        logger.info("Computing ratio features...")
        
        features = {}
        
        # Direction ratios
        if 'direction' in transactions.columns:
            direction_counts = transactions.groupby(self.signal_id_col)['direction'].value_counts().unstack(fill_value=0)
            
            incoming_count = direction_counts.get('incoming', 0)
            outgoing_count = direction_counts.get('outgoing', 0)
            total_direction = incoming_count + outgoing_count
            
            features['incoming_ratio'] = safe_divide(incoming_count, total_direction)
            features['outgoing_ratio'] = safe_divide(outgoing_count, total_direction)
            
            self.feature_store.register_feature(FeatureMetadata(
                name='incoming_ratio',
                description='Ratio of incoming transactions to total transactions',
                feature_type='ratio',
                lookback_rule='Ratio computed from transactions <= signal date',
                data_source='transactions',
                aggregation_method='ratio'
            ))
            
            self.feature_store.register_feature(FeatureMetadata(
                name='outgoing_ratio',
                description='Ratio of outgoing transactions to total transactions',
                feature_type='ratio',
                lookback_rule='Ratio computed from transactions <= signal date',
                data_source='transactions',
                aggregation_method='ratio'
            ))
        
        # Amount share ratios (if amount and direction exist)
        if 'direction' in transactions.columns:
            amount_cols = [col for col in transactions.columns if 'amount' in col.lower()]
            if amount_cols:
                amount_col = amount_cols[0]
                
                for direction in ['incoming', 'outgoing']:
                    direction_trans = transactions[transactions['direction'] == direction]
                    direction_amount = direction_trans.groupby(self.signal_id_col)[amount_col].sum()
                    total_amount = transactions.groupby(self.signal_id_col)[amount_col].sum()
                    
                    feat_name = f'{direction}_amount_share'
                    features[feat_name] = safe_divide(direction_amount, total_amount)
                    
                    self.feature_store.register_feature(FeatureMetadata(
                        name=feat_name,
                        description=f'Share of total amount from {direction} transactions',
                        feature_type='ratio',
                        lookback_rule=f'Amount share from {direction} transactions <= signal date',
                        data_source='transactions',
                        aggregation_method='ratio'
                    ))
        
        # Transaction type ratios
        if 'type' in transactions.columns:
            type_counts = transactions.groupby(self.signal_id_col)['type'].value_counts().unstack(fill_value=0)
            total_transactions = type_counts.sum(axis=1)
            
            common_types = ['card', 'bank_transfer', 'cash', 'international']
            for trans_type in common_types:
                if trans_type in type_counts.columns:
                    feat_name = f'type_{trans_type}_ratio'
                    features[feat_name] = safe_divide(type_counts[trans_type], total_transactions)
                    
                    self.feature_store.register_feature(FeatureMetadata(
                        name=feat_name,
                        description=f'Ratio of {trans_type} transactions to total transactions',
                        feature_type='ratio',
                        lookback_rule=f'Ratio computed from {trans_type} transactions <= signal date',
                        data_source='transactions',
                        aggregation_method='ratio'
                    ))
        
        feature_df = pd.DataFrame(features).reset_index()
        logger.info(f"Computed {len(features)} ratio features")
        return feature_df
    
    def compute_time_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features"""
        logger.info("Computing time features...")
        
        features = {}
        
        if self.transaction_date_col not in transactions.columns:
            logger.warning("Transaction date column not found, skipping time features")
            return pd.DataFrame({self.signal_id_col: transactions[self.signal_id_col].unique()})
        
        # First and last transaction timestamps
        time_stats = transactions.groupby(self.signal_id_col)[self.transaction_date_col].agg(['min', 'max'])
        time_stats.columns = ['first_transaction_timestamp', 'last_transaction_timestamp']
        
        # History span in days
        time_stats['history_span_days'] = (time_stats['last_transaction_timestamp'] - 
                                          time_stats['first_transaction_timestamp']).dt.days
        
        # Transaction gaps (time between consecutive transactions)
        def compute_transaction_gaps(group):
            """Compute gaps between consecutive transactions"""
            sorted_times = group.sort_values()
            if len(sorted_times) <= 1:
                return pd.Series({
                    'avg_transaction_gap_days': np.nan,
                    'median_transaction_gap_days': np.nan,
                    'min_transaction_gap_days': np.nan
                })
            
            gaps = sorted_times.diff().dt.days.dropna()
            return pd.Series({
                'avg_transaction_gap_days': gaps.mean(),
                'median_transaction_gap_days': gaps.median(),
                'min_transaction_gap_days': gaps.min()
            })
        
        gap_stats = pd.DataFrame.from_dict({
            signal_id: compute_transaction_gaps(group[self.transaction_date_col])
            for signal_id, group in transactions.groupby(self.signal_id_col)
        }, orient='index')
        
        # Combine all time features
        features = pd.concat([time_stats, gap_stats], axis=1)
        features.index.name = self.signal_id_col
        
        # Register time features
        time_feature_names = {
            'first_transaction_timestamp': 'Timestamp of first transaction',
            'last_transaction_timestamp': 'Timestamp of last transaction',
            'history_span_days': 'Time span between first and last transaction in days',
            'avg_transaction_gap_days': 'Average time gap between consecutive transactions in days',
            'median_transaction_gap_days': 'Median time gap between consecutive transactions in days',
            'min_transaction_gap_days': 'Minimum time gap between consecutive transactions in days'
        }
        
        for feat_name, description in time_feature_names.items():
            if feat_name in features.columns:
                self.feature_store.register_feature(FeatureMetadata(
                    name=feat_name,
                    description=description,
                    feature_type='time',
                    lookback_rule='Time statistics from transactions <= signal date',
                    data_source='transactions',
                    aggregation_method=feat_name.split('_')[0] if '_' in feat_name else 'min'
                ))
        
        feature_df = features.reset_index()
        logger.info(f"Computed {len(features.columns)} time features")
        return feature_df
    
    def create_features(self, data: Dict[str, pd.DataFrame], 
                       temporal_filter: bool = True) -> Dict[str, pd.DataFrame]:
        """Create all features for train and test datasets"""
        logger.info("Creating comprehensive feature set...")
        
        features = {}
        
        for split in ['train', 'test']:
            logger.info(f"Processing {split} data...")
            
            signals = data[f'{split}_signals']
            transactions = data[f'{split}_transactions']
            
            # Apply temporal filtering
            if temporal_filter:
                transactions = self.apply_temporal_filter(transactions, signals)
            
            # Compute feature groups
            count_features = self.compute_count_features(transactions)
            amount_features = self.compute_amount_features(transactions)
            ratio_features = self.compute_ratio_features(transactions)
            time_features = self.compute_time_features(transactions)
            
            # Merge all features
            split_features = signals[[self.signal_id_col]].copy()
            split_features = split_features.merge(count_features, on=self.signal_id_col, how='left')
            split_features = split_features.merge(amount_features, on=self.signal_id_col, how='left')
            split_features = split_features.merge(ratio_features, on=self.signal_id_col, how='left')
            split_features = split_features.merge(time_features, on=self.signal_id_col, how='left')
            
            # Merge with original signal data
            split_features = split_features.merge(signals, on=self.signal_id_col, how='left')
            
            features[split] = split_features
            
            logger.info(f"Created {len(split_features.columns)} features for {split} data")
        
        # Register feature groups
        self._register_feature_groups()
        
        # Save feature metadata
        self.feature_store.save_metadata()
        
        logger.info("Feature creation complete")
        return features
    
    def _register_feature_groups(self):
        """Register feature groups for organized feature management"""
        
        # Count features
        count_features = self.feature_store.get_features_by_type('count')
        self.feature_store.register_feature_group('count_features', count_features)
        
        # Amount features
        amount_features = self.feature_store.get_features_by_type('amount')
        self.feature_store.register_feature_group('amount_features', amount_features)
        
        # Ratio features
        ratio_features = self.feature_store.get_features_by_type('ratio')
        self.feature_store.register_feature_group('ratio_features', ratio_features)
        
        # Time features
        time_features = self.feature_store.get_features_by_type('time')
        self.feature_store.register_feature_group('time_features', time_features)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of created features"""
        return {
            'total_features': len(self.feature_store.get_all_features()),
            'feature_groups': {
                group_name: len(features) 
                for group_name, features in self.feature_store.feature_groups.items()
            },
            'feature_types': {
                feature_type: len(self.feature_store.get_features_by_type(feature_type))
                for feature_type in ['count', 'amount', 'ratio', 'time']
            },
            'safety_validation': self.feature_store.validate_feature_safety()
        }
    
    def prepare_modeling_data(self, features: Dict[str, pd.DataFrame], 
                            target_col: str = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare data for modeling with proper handling of missing values"""
        
        if target_col is None:
            target_col = self.target_col
        
        # Prepare training data
        train_features = features['train'].copy()
        
        # Select numeric features only
        numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns from features
        feature_cols = [col for col in numeric_cols 
                       if col not in [self.signal_id_col, target_col, self.signal_date_col]]
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features[target_col]
        
        # Prepare test data
        test_features = features['test'].copy()
        X_test = test_features[feature_cols].fillna(0)
        
        logger.info(f"Prepared modeling data: {X_train.shape[1]} features")
        
        return X_train, y_train, X_test


def main():
    """Main function to demonstrate basic features functionality"""
    basic_features = BasicFeatures()
    
    # This would typically be called with actual data
    logger.info("Basic features module initialized")
    logger.info("Use notebooks/04_features_v1.ipynb for comprehensive feature engineering")
    
    return {
        'status': 'Basic features module ready',
        'feature_types': ['count', 'amount', 'ratio', 'time'],
        'safety_features': ['temporal_filtering', 'safe_denominator_handling', 'systematic_naming']
    }


if __name__ == "__main__":
    main()
