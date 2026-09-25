"""
Basic Signal-Level Aggregations for WIUT FinTech Hackathon
Leakage-safe signal-level feature extraction without using target information
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicSignalAggregations:
    """Leakage-safe basic signal-level aggregations for EDA"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.paths = self.config.get_paths()
        self.data_config = self.config.get_data_config()
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        logger.info("Loading data for basic aggregations...")
        
        data = {}
        data['train_signals'] = pd.read_csv(self.paths['train_signals'])
        data['test_signals'] = pd.read_csv(self.paths['test_signals'])
        data['train_transactions'] = pd.read_parquet(self.paths['train_transactions'])
        data['test_transactions'] = pd.read_parquet(self.paths['test_transactions'])
        
        # Convert date columns
        data['train_signals']['signal_sanasi'] = pd.to_datetime(data['train_signals']['signal_sanasi'])
        data['test_signals']['signal_sanasi'] = pd.to_datetime(data['test_signals']['signal_sanasi'])
        
        # Identify and convert transaction date columns
        date_cols = self._identify_date_columns(data['train_transactions'])
        for date_col in date_cols:
            data['train_transactions'][date_col] = pd.to_datetime(data['train_transactions'][date_col], errors='coerce')
            data['test_transactions'][date_col] = pd.to_datetime(data['test_transactions'][date_col], errors='coerce')
        
        logger.info("Data loaded successfully")
        return data
    
    def _identify_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify date columns in transaction data"""
        date_cols = []
        date_keywords = ['date', 'time', 'vaqti', 'sanasi', 'timestamp', 'day', 'month', 'year']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                date_cols.append(col)
                continue
            
            try:
                sample_conversion = pd.to_datetime(df[col].head(100), errors='coerce')
                if sample_conversion.notna().sum() > 50:
                    date_cols.append(col)
            except:
                continue
        
        return date_cols
    
    def create_signal_level_features(self, data: Dict[str, pd.DataFrame], 
                                    temporal_filter: bool = True) -> Dict[str, pd.DataFrame]:
        """Create basic signal-level features for EDA (leakage-safe)"""
        logger.info("Creating signal-level features for EDA...")
        
        features = {}
        
        # Process train data
        features['train'] = self._create_single_dataset_features(
            data['train_signals'], 
            data['train_transactions'],
            temporal_filter=temporal_filter
        )
        
        # Process test data
        features['test'] = self._create_single_dataset_features(
            data['test_signals'],
            data['test_transactions'], 
            temporal_filter=temporal_filter
        )
        
        logger.info("Signal-level features created")
        return features
    
    def _create_single_dataset_features(self, signals: pd.DataFrame, 
                                       transactions: pd.DataFrame,
                                       temporal_filter: bool = True) -> pd.DataFrame:
        """Create features for a single dataset"""
        
        # Start with signal data
        features = signals.copy()
        
        # Get transaction date column
        date_cols = self._identify_date_columns(transactions)
        date_col = date_cols[0] if date_cols else None
        
        # Apply temporal filtering if requested
        if temporal_filter and date_col:
            transactions = self._apply_temporal_filter(transactions, signals, date_col)
        
        # Basic transaction aggregations
        agg_features = self._compute_basic_aggregations(transactions, date_col)
        
        # Merge with signal features
        features = features.merge(agg_features, on='signal_id', how='left')
        
        return features
    
    def _apply_temporal_filter(self, transactions: pd.DataFrame, 
                              signals: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Apply temporal filtering to prevent future information leakage"""
        
        # Merge transactions with signal dates
        merged = transactions.merge(signals[['signal_id', 'signal_sanasi']], on='signal_id', how='left')
        
        # Keep only transactions <= signal date
        filtered = merged[merged[date_col] <= merged['signal_sanasi']].copy()
        
        logger.info(f"Temporal filtering: {len(transactions)} -> {len(filtered)} transactions "
                   f"({len(filtered)/len(transactions)*100:.1f}% retained)")
        
        return filtered
    
    def _compute_basic_aggregations(self, transactions: pd.DataFrame, 
                                    date_col: str = None) -> pd.DataFrame:
        """Compute basic transaction aggregations per signal"""
        
        aggregations = {}
        
        # Transaction count
        aggregations['transaction_count'] = transactions.groupby('signal_id').size()
        
        # Amount aggregations (if amount column exists)
        if 'amount' in transactions.columns:
            aggregations['total_amount'] = transactions.groupby('signal_id')['amount'].sum()
            aggregations['mean_amount'] = transactions.groupby('signal_id')['amount'].mean()
            aggregations['std_amount'] = transactions.groupby('signal_id')['amount'].std()
            aggregations['min_amount'] = transactions.groupby('signal_id')['amount'].min()
            aggregations['max_amount'] = transactions.groupby('signal_id')['amount'].max()
            aggregations['median_amount'] = transactions.groupby('signal_id')['amount'].median()
        
        # Direction aggregations (if direction column exists)
        if 'direction' in transactions.columns:
            direction_counts = transactions.groupby('signal_id')['direction'].value_counts().unstack(fill_value=0)
            aggregations['incoming_count'] = direction_counts.get('incoming', 0)
            aggregations['outgoing_count'] = direction_counts.get('outgoing', 0)
            total_dir = aggregations['incoming_count'] + aggregations['outgoing_count']
            aggregations['incoming_proportion'] = aggregations['incoming_count'] / total_dir.replace(0, np.nan)
            aggregations['outgoing_proportion'] = aggregations['outgoing_count'] / total_dir.replace(0, np.nan)
        
        # Type aggregations (if type column exists)
        if 'type' in transactions.columns:
            type_counts = transactions.groupby('signal_id')['type'].value_counts().unstack(fill_value=0)
            # Add top transaction types as features
            for col in type_counts.columns:
                aggregations[f'type_{col}_count'] = type_counts[col]
        
        # Amount index aggregations (if amount_index column exists)
        if 'amount_index' in transactions.columns:
            aggregations['mean_amount_index'] = transactions.groupby('signal_id')['amount_index'].mean()
            aggregations['std_amount_index'] = transactions.groupby('signal_id')['amount_index'].std()
            aggregations['min_amount_index'] = transactions.groupby('signal_id')['amount_index'].min()
            aggregations['max_amount_index'] = transactions.groupby('signal_id')['amount_index'].max()
        
        # Temporal aggregations (if date column exists)
        if date_col:
            # Time span of transactions
            date_span = transactions.groupby('signal_id')[date_col].agg(['min', 'max'])
            aggregations['transaction_date_span_days'] = (date_span['max'] - date_span['min']).dt.days
            
            # Days since first/last transaction (requires signal date - computed separately)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(aggregations).reset_index()
        
        return features_df
    
    def analyze_target_distribution(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Analyze target distribution without using it for feature engineering"""
        
        if 'eskalatsiya' not in signals.columns:
            return {"error": "Target column not found"}
        
        target_col = self.data_config['target_column']
        
        analysis = {
            'total_signals': len(signals),
            'positive_count': int(signals[target_col].sum()),
            'negative_count': int(len(signals) - signals[target_col].sum()),
            'prevalence': float(signals[target_col].mean()),
            'class_ratio': float(signals[target_col].sum() / (len(signals) - signals[target_col].sum()))
        }
        
        # Temporal analysis of target
        signals['year_month'] = signals['signal_sanasi'].dt.to_period('M')
        monthly_analysis = signals.groupby('year_month')[target_col].agg(['mean', 'count'])
        
        analysis['monthly_prevalence'] = {
            str(period): {
                'prevalence': float(row['mean']),
                'count': int(row['count'])
            }
            for period, row in monthly_analysis.iterrows()
        }
        
        return analysis
    
    def analyze_signal_date_distribution(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Analyze signal date distribution"""
        
        analysis = {
            'min_date': str(signals['signal_sanasi'].min()),
            'max_date': str(signals['signal_sanasi'].max()),
            'date_range_days': (signals['signal_sanasi'].max() - signals['signal_sanasi'].min()).days,
            'mean_date': str(signals['signal_sanasi'].mean()),
            'median_date': str(signals['signal_sanasi'].median())
        }
        
        # Monthly distribution
        signals['year_month'] = signals['signal_sanasi'].dt.to_period('M')
        monthly_counts = signals.groupby('year_month').size()
        
        analysis['monthly_distribution'] = {
            str(period): int(count) for period, count in monthly_counts.items()
        }
        
        # Day of week distribution
        signals['day_of_week'] = signals['signal_sanasi'].dt.day_name()
        dow_counts = signals['day_of_week'].value_counts()
        
        analysis['day_of_week_distribution'] = {
            day: int(count) for day, count in dow_counts.items()
        }
        
        return analysis
    
    def analyze_transaction_counts(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction count distribution"""
        
        if 'transaction_count' not in features.columns:
            return {"error": "transaction_count not found in features"}
        
        analysis = {
            'mean': float(features['transaction_count'].mean()),
            'median': float(features['transaction_count'].median()),
            'std': float(features['transaction_count'].std()),
            'min': int(features['transaction_count'].min()),
            'max': int(features['transaction_count'].max()),
            'percentiles': {
                '25': float(features['transaction_count'].quantile(0.25)),
                '50': float(features['transaction_count'].quantile(0.50)),
                '75': float(features['transaction_count'].quantile(0.75)),
                '90': float(features['transaction_count'].quantile(0.90)),
                '95': float(features['transaction_count'].quantile(0.95)),
                '99': float(features['transaction_count'].quantile(0.99))
            }
        }
        
        # Distribution buckets
        analysis['distribution_buckets'] = {
            '0_transactions': int((features['transaction_count'] == 0).sum()),
            '1_transaction': int((features['transaction_count'] == 1).sum()),
            '2-5_transactions': int(((features['transaction_count'] >= 2) & (features['transaction_count'] <= 5)).sum()),
            '6-10_transactions': int(((features['transaction_count'] >= 6) & (features['transaction_count'] <= 10)).sum()),
            '11-20_transactions': int(((features['transaction_count'] >= 11) & (features['transaction_count'] <= 20)).sum()),
            '20+_transactions': int((features['transaction_count'] > 20).sum())
        }
        
        return analysis
    
    def analyze_direction_proportions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze incoming/outgoing proportions"""
        
        analysis = {}
        
        if 'incoming_proportion' in features.columns:
            valid_props = features['incoming_proportion'].dropna()
            analysis['incoming_proportion'] = {
                'mean': float(valid_props.mean()),
                'median': float(valid_props.median()),
                'std': float(valid_props.std()),
                'min': float(valid_props.min()),
                'max': float(valid_props.max())
            }
        
        if 'outgoing_proportion' in features.columns:
            valid_props = features['outgoing_proportion'].dropna()
            analysis['outgoing_proportion'] = {
                'mean': float(valid_props.mean()),
                'median': float(valid_props.median()),
                'std': float(valid_props.std()),
                'min': float(valid_props.min()),
                'max': float(valid_props.max())
            }
        
        # Direction counts
        if 'incoming_count' in features.columns and 'outgoing_count' in features.columns:
            analysis['direction_counts'] = {
                'total_incoming': int(features['incoming_count'].sum()),
                'total_outgoing': int(features['outgoing_count'].sum()),
                'signals_with_incoming_only': int((features['outgoing_count'] == 0) & (features['incoming_count'] > 0)).sum(),
                'signals_with_outgoing_only': int((features['incoming_count'] == 0) & (features['outgoing_count'] > 0)).sum(),
                'signals_with_both': int((features['incoming_count'] > 0) & (features['outgoing_count'] > 0)).sum()
            }
        
        return analysis
    
    def analyze_type_proportions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction type proportions"""
        
        analysis = {}
        
        # Find all type columns
        type_cols = [col for col in features.columns if col.startswith('type_') and col.endswith('_count')]
        
        if type_cols:
            type_totals = {col: int(features[col].sum()) for col in type_cols}
            total_type_transactions = sum(type_totals.values())
            
            analysis['type_distribution'] = {
                col.replace('type_', '').replace('_count', ''): {
                    'count': count,
                    'proportion': float(count / total_type_transactions) if total_type_transactions > 0 else 0
                }
                for col, count in type_totals.items()
            }
            
            analysis['total_type_transactions'] = total_type_transactions
        
        return analysis
    
    def analyze_amount_index_distribution(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze amount index distributions"""
        
        analysis = {}
        
        if 'mean_amount_index' in features.columns:
            valid_indices = features['mean_amount_index'].dropna()
            analysis['mean_amount_index'] = {
                'mean': float(valid_indices.mean()),
                'median': float(valid_indices.median()),
                'std': float(valid_indices.std()),
                'min': float(valid_indices.min()),
                'max': float(valid_indices.max())
            }
        
        if 'total_amount' in features.columns:
            valid_amounts = features['total_amount'].dropna()
            analysis['total_amount'] = {
                'mean': float(valid_amounts.mean()),
                'median': float(valid_amounts.median()),
                'std': float(valid_amounts.std()),
                'min': float(valid_amounts.min()),
                'max': float(valid_amounts.max())
            }
        
        return analysis
    
    def analyze_transaction_timing(self, features: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction timing and activity concentration"""
        
        analysis = {}
        
        # Transaction date span analysis
        if 'transaction_date_span_days' in features.columns:
            valid_spans = features['transaction_date_span_days'].dropna()
            analysis['date_span'] = {
                'mean_days': float(valid_spans.mean()),
                'median_days': float(valid_spans.median()),
                'std_days': float(valid_spans.std()),
                'min_days': int(valid_spans.min()),
                'max_days': int(valid_spans.max())
            }
        
        # Signal date distribution
        signals['year_month'] = signals['signal_sanasi'].dt.to_period('M')
        monthly_counts = signals.groupby('year_month').size()
        
        analysis['signal_timing'] = {
            'monthly_distribution': {
                str(period): int(count) for period, count in monthly_counts.items()
            }
        }
        
        return analysis
    
    def compare_escalated_vs_dismissed(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Compare escalated vs dismissed signals (descriptive only, not for feature engineering)"""
        
        if 'eskalatsiya' not in features.columns:
            return {"error": "Target column not found - comparison only available for training data"}
        
        analysis = {}
        target_col = self.data_config['target_column']
        
        # Split by target
        escalated = features[features[target_col] == 1]
        dismissed = features[features[target_col] == 0]
        
        analysis['sample_sizes'] = {
            'escalated_count': len(escalated),
            'dismissed_count': len(dismissed),
            'total_count': len(features)
        }
        
        # Compare transaction counts
        if 'transaction_count' in features.columns:
            analysis['transaction_count_comparison'] = {
                'escalated': {
                    'mean': float(escalated['transaction_count'].mean()),
                    'median': float(escalated['transaction_count'].median()),
                    'std': float(escalated['transaction_count'].std())
                },
                'dismissed': {
                    'mean': float(dismissed['transaction_count'].mean()),
                    'median': float(dismissed['transaction_count'].median()),
                    'std': float(dismissed['transaction_count'].std())
                }
            }
        
        # Compare amounts
        if 'total_amount' in features.columns:
            analysis['amount_comparison'] = {
                'escalated': {
                    'mean': float(escalated['total_amount'].mean()),
                    'median': float(escalated['total_amount'].median())
                },
                'dismissed': {
                    'mean': float(dismissed['total_amount'].mean()),
                    'median': float(dismissed['total_amount'].median())
                }
            }
        
        # Compare direction proportions
        if 'incoming_proportion' in features.columns:
            analysis['direction_comparison'] = {
                'escalated': {
                    'mean_incoming_proportion': float(escalated['incoming_proportion'].mean())
                },
                'dismissed': {
                    'mean_incoming_proportion': float(dismissed['incoming_proportion'].mean())
                }
            }
        
        return analysis
    
    def generate_eda_tables(self, features: Dict[str, pd.DataFrame], 
                           analyses: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate EDA tables for export"""
        
        tables = {}
        
        # Summary statistics table
        summary_data = []
        for dataset_name, df in features.items():
            summary_data.append({
                'dataset': dataset_name,
                'total_signals': len(df),
                'signals_with_transactions': int(df['transaction_count'].notna().sum()),
                'mean_transaction_count': float(df['transaction_count'].mean()) if 'transaction_count' in df.columns else None
            })
        
        tables['summary'] = pd.DataFrame(summary_data)
        
        # Target distribution table (train only)
        if 'train' in features and 'eskalatsiya' in features['train'].columns:
            target_dist = analyses.get('target_distribution', {})
            if target_dist and 'error' not in target_dist:
                target_data = [{
                    'metric': 'total_signals',
                    'value': target_dist['total_signals']
                }, {
                    'metric': 'positive_count',
                    'value': target_dist['positive_count']
                }, {
                    'metric': 'negative_count', 
                    'value': target_dist['negative_count']
                }, {
                    'metric': 'prevalence',
                    'value': target_dist['prevalence']
                }]
                tables['target_distribution'] = pd.DataFrame(target_data)
        
        return tables
    
    def save_eda_tables(self, tables: Dict[str, pd.DataFrame], output_dir: str = "artifacts/eda_tables"):
        """Save EDA tables to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for table_name, df in tables.items():
            file_path = output_path / f"{table_name}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {table_name} to {file_path}")


def main():
    """Main function to run basic aggregations for EDA"""
    aggregator = BasicSignalAggregations()
    
    # Load data
    data = aggregator.load_data()
    
    # Create signal-level features
    features = aggregator.create_signal_level_features(data, temporal_filter=True)
    
    # Run analyses
    analyses = {}
    analyses['target_distribution'] = aggregator.analyze_target_distribution(data['train_signals'])
    analyses['signal_date_distribution'] = aggregator.analyze_signal_date_distribution(data['train_signals'])
    analyses['transaction_counts'] = aggregator.analyze_transaction_counts(features['train'])
    analyses['direction_proportions'] = aggregator.analyze_direction_proportions(features['train'])
    analyses['type_proportions'] = aggregator.analyze_type_proportions(features['train'])
    analyses['amount_index'] = aggregator.analyze_amount_index_distribution(features['train'])
    analyses['transaction_timing'] = aggregator.analyze_transaction_timing(features['train'], data['train_signals'])
    analyses['escalated_vs_dismissed'] = aggregator.compare_escalated_vs_dismissed(features['train'])
    
    # Generate and save tables
    tables = aggregator.generate_eda_tables(features, analyses)
    aggregator.save_eda_tables(tables)
    
    logger.info("Basic aggregations and EDA analysis completed")
    
    return features, analyses, tables


if __name__ == "__main__":
    main()