"""
Feature Store for WIUT FinTech Hackathon
Systematic feature management with metadata and lookback rules
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a single feature"""
    name: str
    description: str
    feature_type: str  # 'count', 'amount', 'ratio', 'time', 'categorical'
    lookback_rule: str  # Description of temporal lookback rule
    data_source: str  # 'transactions', 'signals', 'derived'
    aggregation_method: str  # 'sum', 'mean', 'count', etc.
    safe_from_leakage: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dependencies: List[str] = field(default_factory=list)


class FeatureStore:
    """Systematic feature management with metadata tracking"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.data_config = self.config.get_data_config()
        self.features_metadata: Dict[str, FeatureMetadata] = {}
        self.feature_groups: Dict[str, List[str]] = {}
        
    def register_feature(self, metadata: FeatureMetadata) -> None:
        """Register a feature with its metadata"""
        self.features_metadata[metadata.name] = metadata
        logger.info(f"Registered feature: {metadata.name} ({metadata.feature_type})")
    
    def register_feature_group(self, group_name: str, feature_names: List[str]) -> None:
        """Register a group of related features"""
        self.feature_groups[group_name] = feature_names
        logger.info(f"Registered feature group: {group_name} with {len(feature_names)} features")
    
    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a specific feature"""
        return self.features_metadata.get(feature_name)
    
    def get_feature_group(self, group_name: str) -> List[str]:
        """Get all features in a group"""
        return self.feature_groups.get(group_name, [])
    
    def get_all_features(self) -> List[str]:
        """Get all registered feature names"""
        return list(self.features_metadata.keys())
    
    def get_features_by_type(self, feature_type: str) -> List[str]:
        """Get all features of a specific type"""
        return [
            name for name, metadata in self.features_metadata.items()
            if metadata.feature_type == feature_type
        ]
    
    def validate_feature_safety(self) -> Dict[str, Any]:
        """Validate that all registered features are safe from leakage"""
        unsafe_features = [
            name for name, metadata in self.features_metadata.items()
            if not metadata.safe_from_leakage
        ]
        
        validation_report = {
            'total_features': len(self.features_metadata),
            'safe_features': len(self.features_metadata) - len(unsafe_features),
            'unsafe_features': unsafe_features,
            'validation_passed': len(unsafe_features) == 0
        }
        
        if not validation_report['validation_passed']:
            logger.warning(f"⚠️  Found {len(unsafe_features)} potentially unsafe features")
        else:
            logger.info("✅ All features validated as safe from leakage")
        
        return validation_report
    
    def save_metadata(self, output_path: str = "artifacts/feature_metadata.json") -> None:
        """Save feature metadata to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_dict = {
            'features': {
                name: {
                    'name': metadata.name,
                    'description': metadata.description,
                    'feature_type': metadata.feature_type,
                    'lookback_rule': metadata.lookback_rule,
                    'data_source': metadata.data_source,
                    'aggregation_method': metadata.aggregation_method,
                    'safe_from_leakage': metadata.safe_from_leakage,
                    'created_at': metadata.created_at,
                    'dependencies': metadata.dependencies
                }
                for name, metadata in self.features_metadata.items()
            },
            'feature_groups': self.feature_groups,
            'validation': self.validate_feature_safety()
        }
        
        with open(output_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Feature metadata saved to {output_path}")
    
    def load_metadata(self, input_path: str = "artifacts/feature_metadata.json") -> None:
        """Load feature metadata from file"""
        input_file = Path(input_path)
        if not input_file.exists():
            logger.warning(f"Feature metadata file not found: {input_path}")
            return
        
        with open(input_file, 'r') as f:
            metadata_dict = json.load(f)
        
        # Load features
        for name, feature_data in metadata_dict['features'].items():
            metadata = FeatureMetadata(
                name=feature_data['name'],
                description=feature_data['description'],
                feature_type=feature_data['feature_type'],
                lookback_rule=feature_data['lookback_rule'],
                data_source=feature_data['data_source'],
                aggregation_method=feature_data['aggregation_method'],
                safe_from_leakage=feature_data['safe_from_leakage'],
                created_at=feature_data['created_at'],
                dependencies=feature_data['dependencies']
            )
            self.features_metadata[name] = metadata
        
        # Load feature groups
        self.feature_groups = metadata_dict['feature_groups']
        
        logger.info(f"Feature metadata loaded from {input_path}")
    
    def generate_feature_report(self) -> str:
        """Generate a comprehensive feature report"""
        report = []
        report.append("=" * 80)
        report.append("WIUT FINTECH HACKATHON - FEATURE STORE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY:")
        report.append(f"Total Features: {len(self.features_metadata)}")
        report.append(f"Feature Groups: {len(self.feature_groups)}")
        
        # Features by type
        feature_types = {}
        for metadata in self.features_metadata.values():
            feature_types[metadata.feature_type] = feature_types.get(metadata.feature_type, 0) + 1
        
        report.append("\nFEATURES BY TYPE:")
        for feature_type, count in sorted(feature_types.items()):
            report.append(f"  {feature_type}: {count}")
        
        # Feature groups
        report.append("\nFEATURE GROUPS:")
        for group_name, features in self.feature_groups.items():
            report.append(f"  {group_name}: {len(features)} features")
        
        # Safety validation
        validation = self.validate_feature_safety()
        report.append("\nSAFETY VALIDATION:")
        report.append(f"  Safe Features: {validation['safe_features']}")
        report.append(f"  Unsafe Features: {len(validation['unsafe_features'])}")
        report.append(f"  Validation Passed: {validation['validation_passed']}")
        
        # Detailed feature listing
        report.append("\nDETAILED FEATURE LISTING:")
        for name, metadata in sorted(self.features_metadata.items()):
            report.append(f"\n{name}:")
            report.append(f"  Type: {metadata.feature_type}")
            report.append(f"  Description: {metadata.description}")
            report.append(f"  Lookback Rule: {metadata.lookback_rule}")
            report.append(f"  Data Source: {metadata.data_source}")
            report.append(f"  Aggregation: {metadata.aggregation_method}")
            report.append(f"  Safe: {metadata.safe_from_leakage}")
        
        return "\n".join(report)
    
    def create_feature_dataframe(self, features_dict: Dict[str, pd.Series], 
                                 signal_ids: pd.Series) -> pd.DataFrame:
        """Create a feature DataFrame from a dictionary of feature series"""
        feature_df = pd.DataFrame(features_dict)
        feature_df['signal_id'] = signal_ids.values
        
        # Reorder columns to put signal_id first
        cols = ['signal_id'] + [col for col in feature_df.columns if col != 'signal_id']
        feature_df = feature_df[cols]
        
        return feature_df
    
    def merge_features(self, base_df: pd.DataFrame, 
                      new_features: pd.DataFrame, 
                      on: str = 'signal_id') -> pd.DataFrame:
        """Merge new features into base dataframe"""
        merged = base_df.merge(new_features, on=on, how='left')
        logger.info(f"Merged {len(new_features.columns) - 1} features into dataframe")
        return merged


def create_systematic_feature_name(base_name: str, prefix: str = "", suffix: str = "") -> str:
    """Create systematic feature names with consistent naming convention"""
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base_name)
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                default: float = 0.0) -> pd.Series:
    """Safe division that handles zero denominators"""
    if np.isscalar(numerator) and np.isscalar(denominator):
        if denominator == 0 or not np.isfinite(numerator) or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
    result = numerator / denominator
    if not isinstance(result, (pd.Series, pd.DataFrame)):
        return result if np.isfinite(result) else default
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def safe_ratio(numerator: pd.Series, denominator: pd.Series, 
               default: float = 0.0) -> pd.Series:
    """Calculate safe ratio (numerator / (numerator + denominator))"""
    total = numerator + denominator
    return safe_divide(numerator, total, default)


def main():
    """Main function to demonstrate feature store functionality"""
    feature_store = FeatureStore()
    
    # Example feature registration
    example_metadata = FeatureMetadata(
        name="transaction_count",
        description="Total number of transactions per signal",
        feature_type="count",
        lookback_rule="All transactions <= signal date",
        data_source="transactions",
        aggregation_method="count"
    )
    
    feature_store.register_feature(example_metadata)
    
    # Generate report
    report = feature_store.generate_feature_report()
    print(report)
    
    return {
        'status': 'Feature store initialized',
        'registered_features': len(feature_store.features_metadata)
    }


if __name__ == "__main__":
    main()
