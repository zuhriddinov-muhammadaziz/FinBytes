"""
Leakage Detection Module for WIUT FinTech Hackathon
Detects temporal and target leakage in features and models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeakageDetector:
    """Detect various types of leakage in features and model predictions"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.findings = []
    
    def check_temporal_leakage(self, features: pd.DataFrame, signals: pd.DataFrame,
                             temporal_rule: str = "transactions <= signal_date") -> Dict[str, Any]:
        """Check if features use timestamps or records after the allowed cutoff"""
        logger.info("Checking for temporal leakage in features...")
        
        signal_date_col = self.data_config['signal_date_column']
        transaction_date_col = self.data_config.get('transaction_date_column', 'tranzaksiya_vaqti')
        
        # Merge features with signal dates
        features_with_dates = features.merge(
            signals[['signal_id', signal_date_col]], 
            on='signal_id', 
            how='left'
        )
        
        leakage_report = {
            'temporal_rule': temporal_rule,
            'features_checked': [],
            'leakage_detected': False,
            'leakage_details': {},
            'timestamp_violations': []
        }
        
        # Check if features contain date columns that could indicate future information
        date_features = [col for col in features.columns if any(keyword in col.lower() 
                      for keyword in ['date', 'time', 'vaqti', 'timestamp'])]
        
        for date_feature in date_features:
            if date_feature in features.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(features[date_feature]):
                    features[date_feature] = pd.to_datetime(features[date_feature], errors='coerce')
                
                # Check if any feature values are after signal date (explicit cutoff check)
                if date_feature in features_with_dates.columns:
                    # Create a mask for dates after signal date
                    date_after_signal = (
                        features_with_dates[date_feature] > features_with_dates[signal_date_col]
                    ) & features_with_dates[date_feature].notna()
                    
                    violations_count = date_after_signal.sum()
                    
                    leakage_report['features_checked'].append(date_feature)
                    leakage_report['leakage_details'][date_feature] = {
                        'has_date_values': features[date_feature].notna().sum(),
                        'date_range': {
                            'min': str(features[date_feature].min()) if features[date_feature].notna().any() else None,
                            'max': str(features[date_feature].max()) if features[date_feature].notna().any() else None
                        },
                        'timestamp_violations': {
                            'count': int(violations_count),
                            'percentage': float(violations_count / len(features_with_dates) * 100) if len(features_with_dates) > 0 else 0
                        }
                    }
                    
                    if violations_count > 0:
                        leakage_report['timestamp_violations'].append({
                            'feature': date_feature,
                            'violations_count': int(violations_count),
                            'violation_percentage': float(violations_count / len(features_with_dates) * 100) if len(features_with_dates) > 0 else 0
                        })
                        leakage_report['leakage_detected'] = True
        
        # Check for potential temporal leakage in aggregation features
        # Features like "last_transaction_date" could indicate future information
        temporal_aggregation_features = [col for col in features.columns 
                                         if any(keyword in col.lower() 
                                               for keyword in ['last', 'recent', 'final', 'end'])]
        
        for temp_feature in temporal_aggregation_features:
            leakage_report['features_checked'].append(temp_feature)
            leakage_report['leakage_details'][temp_feature] = {
                'type': 'temporal_aggregation',
                'requires_manual_review': True,
                'reason': 'Temporal aggregation features may use future information'
            }
        
        if leakage_report['timestamp_violations']:
            leakage_report['leakage_detected'] = True
            logger.warning(f"⚠️  EXPLICIT TIMESTAMP VIOLATIONS DETECTED: {len(leakage_report['timestamp_violations'])} features use timestamps after allowed cutoff")
        elif leakage_report['features_checked']:
            logger.warning(f"⚠️  Potential temporal leakage detected in {len(leakage_report['features_checked'])} features (requires manual review)")
        else:
            logger.info("✅ No temporal leakage detected in features")
        
        self.findings.append({
            'check_type': 'temporal_leakage',
            'result': leakage_report
        })
        
        return leakage_report
    
    def check_target_leakage(self, features: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Check if features contain target information or are derived from target"""
        logger.info("Checking for target leakage in features...")
        
        if target_col is None:
            target_col = self.data_config['target_column']
        
        leakage_report = {
            'target_column': target_col,
            'features_checked': [],
            'leakage_detected': False,
            'leakage_details': {}
        }
        
        # Check for exact target column
        if target_col in features.columns:
            leakage_report['features_checked'].append(target_col)
            leakage_report['leakage_details'][target_col] = {
                'type': 'direct_target',
                'severity': 'HIGH',
                'action': 'Remove this feature before training'
            }
            leakage_report['leakage_detected'] = True
        
        # Check for target-derived features (similar names)
        target_keywords = ['target', 'eskalatsiya', 'escalated', 'dismissed', 'class']
        target_like_features = [col for col in features.columns 
                               if any(keyword in col.lower() for keyword in target_keywords)]
        
        for feature in target_like_features:
            if feature != target_col:
                leakage_report['features_checked'].append(feature)
                leakage_report['leakage_details'][feature] = {
                    'type': 'target_like',
                    'severity': 'MEDIUM',
                    'action': 'Manual review required'
                }
        
        # Check for unusually high correlation with target (would need target data)
        # This is a placeholder - actual correlation check would require target data
        
        if leakage_report['features_checked']:
            leakage_report['leakage_detected'] = True
            logger.warning(f"⚠️  Potential target leakage detected in {len(leakage_report['features_checked'])} features")
        else:
            logger.info("✅ No target leakage detected in features")
        
        self.findings.append({
            'check_type': 'target_leakage',
            'result': leakage_report
        })
        
        return leakage_report
    
    def check_feature_leakage_from_transactions(self, features: pd.DataFrame, 
                                            transactions: pd.DataFrame, signals: pd.DataFrame,
                                            temporal_rule: str = "transactions <= signal_date") -> Dict[str, Any]:
        """Check if feature engineering respects temporal availability rules"""
        logger.info("Checking if feature engineering respects temporal availability rules...")
        
        signal_date_col = self.data_config['signal_date_column']
        transaction_date_col = self.data_config.get('transaction_date_column', 'tranzaksiya_vaqti')
        
        leakage_report = {
            'temporal_rule': temporal_rule,
            'compliance_check': [],
            'leakage_detected': False,
            'leakage_details': {}
        }
        
        # Check if transactions have been properly filtered temporally
        # This would require analyzing the actual feature engineering process
        # For now, we provide a framework for the check
        
        # Simulated check: verify that transaction aggregations respect temporal boundaries
        # In a real implementation, this would trace back to the actual feature engineering
        
        aggregation_features = [col for col in features.columns 
                              if any(keyword in col.lower() 
                                     for keyword in ['count', 'sum', 'mean', 'std', 'amount', 'transaction'])]
        
        for feature in aggregation_features:
            leakage_report['compliance_check'].append(feature)
            leakage_report['leakage_details'][feature] = {
                'type': 'aggregation_feature',
                'compliance_status': 'requires_manual_verification',
                'reason': 'Aggregation features must respect temporal boundaries'
            }
        
        # Check for date span features that might include future information
        date_span_features = [col for col in features.columns 
                            if 'span' in col.lower() or 'range' in col.lower()]
        
        for feature in date_span_features:
            leakage_report['compliance_check'].append(feature)
            leakage_report['leakage_details'][feature] = {
                'type': 'date_span_feature',
                'compliance_status': 'requires_manual_verification',
                'reason': 'Date span features must be computed from historical transactions only'
            }
        
        if leakage_report['compliance_check']:
            logger.info(f"⚠️  {len(leakage_report['compliance_check'])} aggregation features require temporal compliance verification")
        else:
            logger.info("✅ No aggregation features found")
        
        self.findings.append({
            'check_type': 'feature_engineering_compliance',
            'result': leakage_report
        })
        
        return leakage_report
    
    def check_train_test_leakage(self, train_features: pd.DataFrame, 
                                 test_features: pd.DataFrame) -> Dict[str, Any]:
        """Check if train/test data separation is maintained"""
        logger.info("Checking train/test data separation...")
        
        leakage_report = {
            'overlap_detected': False,
            'overlap_details': {}
        }
        
        # Check for overlapping signal_ids
        train_ids = set(train_features['signal_id']) if 'signal_id' in train_features.columns else set()
        test_ids = set(test_features['signal_id']) if 'signal_id' in test_features.columns else set()
        
        overlap = train_ids.intersection(test_ids)
        
        if overlap:
            leakage_report['overlap_detected'] = True
            leakage_report['overlap_details']['signal_id_overlap'] = {
                'count': len(overlap),
                'sample_ids': list(overlap)[:10]  # First 10 for inspection
            }
            logger.warning(f"⚠️  Signal ID overlap detected: {len(overlap)} shared signal_ids")
        else:
            logger.info("✅ No signal ID overlap between train and test")
        
        # Check for feature value overlap (potential fingerprinting)
        # This is a simplified check - more sophisticated checks may be needed
        common_cols = set(train_features.columns) & set(test_features.columns)
        
        leakage_report['overlap_details']['common_columns'] = {
            'count': len(common_cols),
            'columns': list(common_cols)
        }
        
        self.findings.append({
            'check_type': 'train_test_separation',
            'result': leakage_report
        })
        
        return leakage_report
    
    def run_comprehensive_leakage_audit(self, features: Dict[str, pd.DataFrame],
                                        signals: Dict[str, pd.DataFrame],
                                        transactions: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """Run comprehensive leakage audit"""
        logger.info("Running comprehensive leakage audit...")
        
        audit_results = {
            'temporal_leakage': {},
            'target_leakage': {},
            'feature_engineering_compliance': {},
            'train_test_separation': {},
            'overall_leakage_detected': False
        }
        
        # Temporal leakage check
        if 'train' in features and 'train' in signals:
            audit_results['temporal_leakage'] = self.check_temporal_leakage(
                features['train'], signals['train']
            )
        
        # Target leakage check
        if 'train' in features:
            audit_results['target_leakage'] = self.check_target_leakage(features['train'])
        
        # Feature engineering compliance check
        if transactions and 'train' in features and 'train' in signals and 'train' in transactions:
            audit_results['feature_engineering_compliance'] = self.check_feature_leakage_from_transactions(
                features['train'], transactions['train'], signals['train']
            )
        
        # Train/test separation check
        if 'train' in features and 'test' in features:
            audit_results['train_test_separation'] = self.check_train_test_leakage(
                features['train'], features['test']
            )
        
        # Overall assessment
        audit_results['overall_leakage_detected'] = any(
            result.get('leakage_detected', False) or result.get('overlap_detected', False)
            for result in audit_results.values()
            if isinstance(result, dict)
        )
        
        logger.info(f"Comprehensive leakage audit complete. Overall leakage detected: {audit_results['overall_leakage_detected']}")
        
        return audit_results
    
    def generate_leakage_report(self) -> str:
        """Generate comprehensive leakage report"""
        report = []
        report.append("=" * 80)
        report.append("WIUT FINTECH HACKATHON - LEAKAGE AUDIT REPORT")
        report.append("=" * 80)
        report.append("")
        
        for finding in self.findings:
            check_type = finding['check_type']
            result = finding['result']
            
            report.append(f"{check_type.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            report.append(json.dumps(result, indent=2, default=str))
            report.append("")
        
        return "\n".join(report)
    
    def save_leakage_report(self, output_path: str = "artifacts/leakage_audit_report.json"):
        """Save leakage audit report to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        audit_data = {
            'findings': self.findings,
            'overall_leakage_detected': any(
                finding['result'].get('leakage_detected', False) or finding['result'].get('overlap_detected', False)
                for finding in self.findings
            )
        }
        
        with open(output_file, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        logger.info(f"Leakage audit report saved to {output_path}")


def main():
    """Main function to run leakage detection"""
    detector = LeakageDetector()
    
    # This would typically be called with actual data
    logger.info("Leakage detection framework initialized")
    logger.info("Use notebooks/03_validation_design.ipynb for comprehensive leakage testing")
    
    return {
        'status': 'Leakage detection framework ready',
        'available_checks': [
            'temporal_leakage',
            'target_leakage', 
            'feature_engineering_compliance',
            'train_test_separation'
        ]
    }


if __name__ == "__main__":
    main()