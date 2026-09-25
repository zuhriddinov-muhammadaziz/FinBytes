"""
Multi-Window Behavioral Features for WIUT FinTech Hackathon
Time-windowed transaction aggregates with recency and velocity features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import timedelta
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from .feature_store import FeatureStore, FeatureMetadata, safe_divide, safe_ratio, create_systematic_feature_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WindowFeatures:
    """Multi-window behavioral features with temporal safety"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.feature_store = FeatureStore(config_path)
        
        # Column mappings
        self.signal_id_col = self.data_config['signal_id_column']
        self.signal_date_col = self.data_config['signal_date_column']
        self.target_col = self.data_config['target_column']
        self.transaction_date_col = self.data_config.get('transaction_date_column', 'tranzaksiya_vaqti')
        
        # Window definitions (in days)
        self.windows = [1, 3, 7, 14, 30, 60, 90]
        
        # High amount threshold (75th percentile of amounts)
        self.high_amount_threshold = None
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        logger.info("Loading data for window features...")
        
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
        transactions = transactions.drop(columns=[self.signal_date_col], errors='ignore')
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
    
    def compute_high_amount_threshold(self, transactions: pd.DataFrame) -> float:
        """Compute high amount threshold (75th percentile)"""
        amount_cols = [col for col in transactions.columns if 'amount' in col.lower()]
        
        if not amount_cols:
            logger.warning("No amount columns found, using default threshold")
            return 1000.0  # Default threshold
        
        amount_col = amount_cols[0]
        threshold = transactions[amount_col].quantile(0.75)
        logger.info(f"High amount threshold: {threshold:.2f}")
        return threshold
    
    def compute_window_features(self, transactions: pd.DataFrame, 
                              signals: pd.DataFrame) -> pd.DataFrame:
        """Compute multi-window behavioral features"""
        logger.info("Computing multi-window behavioral features...")
        
        # Compute high amount threshold
        self.high_amount_threshold = self.compute_high_amount_threshold(transactions)
        
        # Merge transactions with signal dates for window calculations
        transactions = transactions.drop(columns=[self.signal_date_col], errors='ignore')
        merged = transactions.merge(
            signals[[self.signal_id_col, self.signal_date_col]], 
            on=self.signal_id_col, 
            how='left'
        )
        
        # Initialize features dataframe with signal IDs
        features = signals[[self.signal_id_col]].copy()
        
        # Compute features for each window
        for window_days in self.windows:
            logger.info(f"Computing {window_days}-day window features...")
            window_features = self._compute_single_window_features(merged, window_days)
            features = features.merge(window_features, on=self.signal_id_col, how='left')
        
        # Compute recency features
        logger.info("Computing recency features...")
        recency_features = self._compute_recency_features(merged)
        features = features.merge(recency_features, on=self.signal_id_col, how='left')
        
        # Compute velocity features
        logger.info("Computing velocity features...")
        velocity_features = self._compute_velocity_features(merged)
        features = features.merge(velocity_features, on=self.signal_id_col, how='left')
        
        logger.info(f"Computed {len(features.columns) - 1} window-based features")
        return features
    
    def _compute_single_window_features(self, merged: pd.DataFrame, 
                                       window_days: int) -> pd.DataFrame:
        """Compute features for a single time window"""
        
        window_features = {}
        window_suffix = f"{window_days}d"
        
        # Filter transactions within window
        window_start = merged[self.signal_date_col] - pd.Timedelta(days=window_days)
        in_window = merged[
            (merged[self.transaction_date_col] >= window_start) & 
            (merged[self.transaction_date_col] <= merged[self.signal_date_col])
        ]
        
        # Transaction counts
        total_count = in_window.groupby(self.signal_id_col).size()
        window_features[f'transaction_count_{window_suffix}'] = total_count
        
        self.feature_store.register_feature(FeatureMetadata(
            name=f'transaction_count_{window_suffix}',
            description=f'Total transactions in {window_days}-day window',
            feature_type='count',
            lookback_rule=f'Transactions in {window_days} days before signal date',
            data_source='transactions',
            aggregation_method='count'
        ))
        
        # Direction counts
        if 'direction' in in_window.columns:
            direction_counts = in_window.groupby(self.signal_id_col)['direction'].value_counts().unstack(fill_value=0)
            
            window_features[f'incoming_count_{window_suffix}'] = direction_counts.get('incoming', 0)
            window_features[f'outgoing_count_{window_suffix}'] = direction_counts.get('outgoing', 0)
            
            self.feature_store.register_feature(FeatureMetadata(
                name=f'incoming_count_{window_suffix}',
                description=f'Incoming transactions in {window_days}-day window',
                feature_type='count',
                lookback_rule=f'Incoming transactions in {window_days} days before signal date',
                data_source='transactions',
                aggregation_method='count'
            ))
            
            self.feature_store.register_feature(FeatureMetadata(
                name=f'outgoing_count_{window_suffix}',
                description=f'Outgoing transactions in {window_days}-day window',
                feature_type='count',
                lookback_rule=f'Outgoing transactions in {window_days} days before signal date',
                data_source='transactions',
                aggregation_method='count'
            ))
        
        # Type-specific counts
        if 'type' in in_window.columns:
            type_counts = in_window.groupby(self.signal_id_col)['type'].value_counts().unstack(fill_value=0)
            
            common_types = ['card', 'bank_transfer', 'cash', 'international']
            for trans_type in common_types:
                if trans_type in type_counts.columns:
                    feat_name = f'type_{trans_type}_count_{window_suffix}'
                    window_features[feat_name] = type_counts[trans_type]
                    
                    self.feature_store.register_feature(FeatureMetadata(
                        name=feat_name,
                        description=f'{trans_type} transactions in {window_days}-day window',
                        feature_type='count',
                        lookback_rule=f'{trans_type} transactions in {window_days} days before signal date',
                        data_source='transactions',
                        aggregation_method='count'
                    ))
        
        # Amount statistics
        amount_cols = [col for col in in_window.columns if 'amount' in col.lower()]
        if amount_cols:
            amount_col = amount_cols[0]
            
            amount_stats = in_window.groupby(self.signal_id_col)[amount_col].agg(['sum', 'mean'])
            amount_stats.columns = [f'amount_sum_{window_suffix}', f'amount_mean_{window_suffix}']
            
            for feat_name in amount_stats.columns:
                window_features[feat_name] = amount_stats[feat_name]
                
                self.feature_store.register_feature(FeatureMetadata(
                    name=feat_name,
                    description=f'{feat_name.replace(f"_{window_suffix}", "")} in {window_days}-day window',
                    feature_type='amount',
                    lookback_rule=f'Amount statistics in {window_days} days before signal date',
                    data_source='transactions',
                    aggregation_method=feat_name.split('_')[1]
                ))
        
        # Ratios (with safe denominator handling)
        if 'direction' in in_window.columns:
            total_dir = window_features.get(f'incoming_count_{window_suffix}', 0) + \
                       window_features.get(f'outgoing_count_{window_suffix}', 0)
            
            window_features[f'outgoing_ratio_{window_suffix}'] = safe_divide(
                window_features.get(f'outgoing_count_{window_suffix}', 0), total_dir
            )
            
            self.feature_store.register_feature(FeatureMetadata(
                name=f'outgoing_ratio_{window_suffix}',
                description=f'Ratio of outgoing transactions in {window_days}-day window',
                feature_type='ratio',
                lookback_rule=f'Outgoing ratio in {window_days} days before signal date',
                data_source='transactions',
                aggregation_method='ratio'
            ))
        
        if 'type' in in_window.columns:
            total_count = window_features.get(f'transaction_count_{window_suffix}', 0)
            
            if 'international' in type_counts.columns:
                window_features[f'international_ratio_{window_suffix}'] = safe_divide(
                    type_counts['international'], total_count
                )
                
                self.feature_store.register_feature(FeatureMetadata(
                    name=f'international_ratio_{window_suffix}',
                    description=f'Ratio of international transactions in {window_days}-day window',
                    feature_type='ratio',
                    lookback_rule=f'International ratio in {window_days} days before signal date',
                    data_source='transactions',
                    aggregation_method='ratio'
                ))
        
        # Active days
        if self.transaction_date_col in in_window.columns:
            in_window['transaction_date'] = in_window[self.transaction_date_col].dt.date
            active_days = in_window.groupby(self.signal_id_col)['transaction_date'].nunique()
            window_features[f'active_days_{window_suffix}'] = active_days
            
            self.feature_store.register_feature(FeatureMetadata(
                name=f'active_days_{window_suffix}',
                description=f'Active days in {window_days}-day window',
                feature_type='count',
                lookback_rule=f'Unique transaction dates in {window_days} days before signal date',
                data_source='transactions',
                aggregation_method='nunique'
            ))
        
        # High-amount transaction counts
        if amount_cols:
            amount_col = amount_cols[0]
            high_amount_trans = in_window[in_window[amount_col] > self.high_amount_threshold]
            high_count = high_amount_trans.groupby(self.signal_id_col).size()
            window_features[f'high_amount_count_{window_suffix}'] = high_count
            
            self.feature_store.register_feature(FeatureMetadata(
                name=f'high_amount_count_{window_suffix}',
                description=f'High-amount transactions in {window_days}-day window',
                feature_type='count',
                lookback_rule=f'Transactions > 75th percentile amount in {window_days} days before signal date',
                data_source='transactions',
                aggregation_method='count'
            ))
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(window_features)
        feature_df.index.name = self.signal_id_col
        feature_df = feature_df.reset_index()
        return feature_df
    
    def _compute_recency_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Compute recency features (time since last transaction types)"""
        
        recency_features = {}
        
        if self.transaction_date_col not in merged.columns:
            logger.warning("Transaction date column not found, skipping recency features")
            return pd.DataFrame({self.signal_id_col: merged[self.signal_id_col].unique()})
        
        # Overall last transaction
        last_transaction = merged.groupby(self.signal_id_col)[self.transaction_date_col].max()
        recency_features['days_since_last_transaction'] = (
            merged.groupby(self.signal_id_col)[self.signal_date_col].first() - last_transaction
        ).dt.days.abs()
        
        self.feature_store.register_feature(FeatureMetadata(
            name='days_since_last_transaction',
            description='Days since last transaction',
            feature_type='time',
            lookback_rule='Time difference between signal date and last transaction',
            data_source='transactions',
            aggregation_method='max'
        ))
        
        # Direction-specific recency
        if 'direction' in merged.columns:
            for direction in ['incoming', 'outgoing']:
                direction_trans = merged[merged['direction'] == direction]
                last_dir_transaction = direction_trans.groupby(self.signal_id_col)[self.transaction_date_col].max()
                
                signal_dates = merged.groupby(self.signal_id_col)[self.signal_date_col].first()
                recency_features[f'days_since_last_{direction}'] = (
                    signal_dates - last_dir_transaction
                ).dt.days.fillna(999)  # Large value for missing
                
                self.feature_store.register_feature(FeatureMetadata(
                    name=f'days_since_last_{direction}',
                    description=f'Days since last {direction} transaction',
                    feature_type='time',
                    lookback_rule=f'Time difference between signal date and last {direction} transaction',
                    data_source='transactions',
                    aggregation_method='max'
                ))
        
        # Type-specific recency
        if 'type' in merged.columns:
            for trans_type in ['international']:
                type_trans = merged[merged['type'] == trans_type]
                last_type_transaction = type_trans.groupby(self.signal_id_col)[self.transaction_date_col].max()
                
                signal_dates = merged.groupby(self.signal_id_col)[self.signal_date_col].first()
                recency_features[f'days_since_last_{trans_type}'] = (
                    signal_dates - last_type_transaction
                ).dt.days.fillna(999)
                
                self.feature_store.register_feature(FeatureMetadata(
                    name=f'days_since_last_{trans_type}',
                    description=f'Days since last {trans_type} transaction',
                    feature_type='time',
                    lookback_rule=f'Time difference between signal date and last {trans_type} transaction',
                    data_source='transactions',
                    aggregation_method='max'
                ))
        
        feature_df = pd.DataFrame(recency_features).reset_index()
        return feature_df
    
    def _compute_velocity_features(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Compute velocity features (transaction rates and ratios)"""
        if self.transaction_date_col not in merged.columns or self.signal_date_col not in merged.columns:
            return pd.DataFrame({self.signal_id_col: merged[self.signal_id_col].drop_duplicates()})

        work = merged.copy()
        dates = pd.to_datetime(work[self.transaction_date_col], errors='coerce')
        signal_dates = pd.to_datetime(work[self.signal_date_col], errors='coerce')
        work['_date'] = dates.dt.floor('D')
        grouped = work.groupby(self.signal_id_col, sort=False)
        active_days = grouped['_date'].nunique()
        total_counts = grouped.size()
        result = pd.DataFrame(index=total_counts.index)
        result['transactions_per_active_day'] = safe_divide(total_counts, active_days)
        result['transactions_per_active_hour'] = safe_divide(total_counts, active_days * 12)

        days_before = (signal_dates - dates).dt.total_seconds() / 86400
        work['_recent'] = days_before.ge(0) & days_before.le(7)
        work['_history'] = days_before.ge(0) & days_before.le(30)
        recent_count = work[work['_recent']].groupby(self.signal_id_col).size()
        history_count = work[work['_history']].groupby(self.signal_id_col).size()
        result['recent_activity_ratio'] = safe_divide(recent_count, history_count).reindex(result.index).fillna(0)

        if 'direction' in work.columns:
            recent_out = work[work['_recent'] & work.direction.eq('outgoing')].groupby(self.signal_id_col).size()
            history_out = work[work['_history'] & work.direction.eq('outgoing')].groupby(self.signal_id_col).size()
            result['recent_outgoing_ratio'] = safe_divide(recent_out, history_out).reindex(result.index).fillna(0)
        amount_cols = [col for col in work.columns if 'amount' in col.lower()]
        if amount_cols:
            amount_col = amount_cols[0]
            recent_amount = work[work['_recent']].groupby(self.signal_id_col)[amount_col].sum()
            history_amount = work[work['_history']].groupby(self.signal_id_col)[amount_col].sum()
            result['recent_amount_ratio'] = safe_divide(recent_amount, history_amount).reindex(result.index).fillna(0)

        for name in result.columns:
            self.feature_store.register_feature(FeatureMetadata(
                name=name, description=name.replace('_', ' ').capitalize(), feature_type='ratio',
                lookback_rule='Computed using transactions on or before the signal date',
                data_source='transactions', aggregation_method='ratio'))
        return result.rename_axis(self.signal_id_col).reset_index()
    
    def create_window_features(self, data: Dict[str, pd.DataFrame], 
                              temporal_filter: bool = True) -> Dict[str, pd.DataFrame]:
        """Create window features for train and test datasets"""
        logger.info("Creating multi-window behavioral features...")
        
        features = {}
        
        for split in ['train', 'test']:
            logger.info(f"Processing {split} data...")
            
            signals = data[f'{split}_signals']
            transactions = data[f'{split}_transactions']
            
            # Apply temporal filtering
            if temporal_filter:
                transactions = self.apply_temporal_filter(transactions, signals)
            
            # Compute window features
            split_features = self.compute_window_features(transactions, signals)
            
            # Merge with original signal data
            split_features = split_features.merge(signals, on=self.signal_id_col, how='left')
            
            features[split] = split_features
            
            logger.info(f"Created {len(split_features.columns)} window features for {split} data")
        
        # Register feature groups
        self._register_feature_groups()
        
        # Save feature metadata
        self.feature_store.save_metadata('artifacts/window_feature_metadata.json')
        
        logger.info("Window feature creation complete")
        return features
    
    def _register_feature_groups(self):
        """Register feature groups for organized feature management"""
        
        # Window-based features
        window_features = [name for name in self.feature_store.get_features_by_type('count')
                           if any(f'{days}d' in name for days in self.windows)]
        
        self.feature_store.register_feature_group('window_count_features', window_features)
        
        # Recency features
        recency_features = self.feature_store.get_features_by_type('time')
        recency_features = [f for f in recency_features if 'days_since' in f]
        self.feature_store.register_feature_group('recency_features', recency_features)
        
        # Velocity features
        velocity_features = [f for f in self.feature_store.get_all_features() if 'ratio' in f and 'recent' in f]
        self.feature_store.register_feature_group('velocity_features', velocity_features)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of created window features"""
        return {
            'total_features': len(self.feature_store.get_all_features()),
            'windows': self.windows,
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
        X_test = test_features.reindex(columns=feature_cols, fill_value=0).fillna(0)
        
        logger.info(f"Prepared modeling data: {X_train.shape[1]} features")
        
        return X_train, y_train, X_test


def main():
    """Main function to demonstrate window features functionality"""
    window_features = WindowFeatures()
    
    logger.info("Window features module initialized")
    logger.info("Use notebooks/05_features_v2.ipynb for comprehensive window feature engineering")
    
    return {
        'status': 'Window features module ready',
        'windows': [1, 3, 7, 14, 30, 60, 90],
        'feature_types': ['count', 'amount', 'ratio', 'time'],
        'recency_features': ['days_since_last_transaction', 'days_since_last_incoming', 'days_since_last_outgoing'],
        'velocity_features': ['transactions_per_active_day', 'recent_activity_ratio']
    }


if __name__ == "__main__":
    main()
