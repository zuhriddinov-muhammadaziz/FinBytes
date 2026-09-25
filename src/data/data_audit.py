"""
Data Audit Module for WIUT FinTech Hackathon
Performs comprehensive schema inspection without loading full datasets into memory
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAuditor:
    """Comprehensive data auditing without loading full datasets"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.paths = self.config.get_paths()
        self.results = {}
    
    def audit_all_files(self) -> Dict[str, Any]:
        """Perform comprehensive audit of all data files"""
        logger.info("Starting comprehensive data audit...")
        
        # Audit CSV files
        self.results['train_signals'] = self._audit_csv(
            self.paths['train_signals'], 'train_signals'
        )
        self.results['test_signals'] = self._audit_csv(
            self.paths['test_signals'], 'test_signals'
        )
        self.results['sample_submission'] = self._audit_csv(
            self.paths['sample_submission'], 'sample_submission'
        )
        
        # Audit Parquet files (metadata only)
        self.results['train_transactions'] = self._audit_parquet(
            self.paths['train_transactions'], 'train_transactions'
        )
        self.results['test_transactions'] = self._audit_parquet(
            self.paths['test_transactions'], 'test_transactions'
        )
        
        # Cross-file validation
        self._validate_relationships()
        
        logger.info("Data audit completed successfully")
        return self.results
    
    def _audit_csv(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Audit CSV file schema and basic statistics"""
        logger.info(f"Auditing CSV file: {file_name}")
        
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        result = {
            "file_name": file_name,
            "file_path": str(path),
            "file_size_mb": path.stat().st_size / (1024 * 1024),
            "file_type": "csv"
        }
        
        # Read just the header and first few rows for schema
        try:
            # Read header
            with open(path, 'r') as f:
                header = f.readline().strip().split(',')
                result["columns"] = header
                result["column_count"] = len(header)
            
            # Read first few rows for type inference and null checking
            df_sample = pd.read_csv(path, nrows=1000)
            
            result["row_count_sample"] = len(df_sample)
            result["dtypes"] = df_sample.dtypes.astype(str).to_dict()
            result["null_counts"] = df_sample.isnull().sum().to_dict()
            result["null_percentages"] = (df_sample.isnull().sum() / len(df_sample) * 100).to_dict()
            
            # Get unique counts for categorical columns
            result["unique_counts"] = {}
            for col in df_sample.columns:
                if df_sample[col].dtype == 'object':
                    result["unique_counts"][col] = df_sample[col].nunique()
            
            # Date range detection
            result["date_ranges"] = self._detect_date_ranges(df_sample)
            
            # Get total row count (efficient)
            with open(path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
            result["total_row_count"] = total_lines
            
            logger.info(f"✅ {file_name}: {total_lines} rows, {len(header)} columns, {result['file_size_mb']:.2f} MB")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Error auditing {file_name}: {e}")
        
        return result
    
    def _audit_parquet(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Audit Parquet file using metadata (no full data load)"""
        logger.info(f"Auditing Parquet file: {file_name}")
        
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        result = {
            "file_name": file_name,
            "file_path": str(path),
            "file_size_mb": path.stat().st_size / (1024 * 1024),
            "file_type": "parquet"
        }
        
        try:
            # Read metadata only (very efficient)
            parquet_file = pq.ParquetFile(path)
            
            # Schema information
            schema = parquet_file.schema_arrow
            result["columns"] = schema.names
            result["column_count"] = len(schema.names)
            result["dtypes"] = {name: str(dtype) for name, dtype in zip(schema.names, schema.types)}
            
            # Row count from metadata
            result["total_row_count"] = parquet_file.metadata.num_rows
            
            # Read small sample for additional analysis
            df_sample = parquet_file.read_row_group(0).to_pandas()
            
            result["sample_row_count"] = len(df_sample)
            result["null_counts_sample"] = df_sample.isnull().sum().to_dict()
            result["null_percentages_sample"] = (df_sample.isnull().sum() / len(df_sample) * 100).to_dict()
            
            # Unique counts for string columns
            result["unique_counts_sample"] = {}
            for col in df_sample.columns:
                if df_sample[col].dtype == 'object':
                    result["unique_counts_sample"][col] = df_sample[col].nunique()
            
            # Date range detection
            result["date_ranges_sample"] = self._detect_date_ranges(df_sample)
            
            # Check for potential date columns
            result["potential_date_columns"] = self._identify_potential_date_columns(df_sample)
            
            logger.info(f"✅ {file_name}: {result['total_row_count']} rows, {result['column_count']} columns, {result['file_size_mb']:.2f} MB")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Error auditing {file_name}: {e}")
        
        return result
    
    def _detect_date_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect date ranges for date-like columns"""
        date_ranges = {}
        
        for col in df.columns:
            # Try to convert to datetime
            try:
                dates = pd.to_datetime(df[col], errors='coerce')
                if dates.notna().sum() > 0:  # If some valid dates
                    valid_dates = dates.dropna()
                    if len(valid_dates) > 0:
                        date_ranges[col] = {
                            "min": str(valid_dates.min()),
                            "max": str(valid_dates.max()),
                            "valid_count": int(valid_dates.count()),
                            "invalid_count": int(dates.isna().sum())
                        }
            except:
                continue
        
        return date_ranges
    
    def _identify_potential_date_columns(self, df: pd.DataFrame) -> list:
        """Identify columns that might contain date information"""
        potential_date_cols = []
        
        for col in df.columns:
            # Check column name for date indicators
            date_keywords = ['date', 'time', 'day', 'month', 'year', 'timestamp', 'sanasi']
            if any(keyword in col.lower() for keyword in date_keywords):
                potential_date_cols.append(col)
                continue
            
            # Try to convert sample to datetime
            try:
                sample_conversion = pd.to_datetime(df[col].head(100), errors='coerce')
                if sample_conversion.notna().sum() > 50:  # At least 50% valid
                    potential_date_cols.append(col)
            except:
                continue
        
        return potential_date_cols
    
    def _validate_relationships(self):
        """Validate relationships between files"""
        logger.info("Validating cross-file relationships...")
        
        # Check signal_id relationship
        self._validate_signal_id_relationship()
        
        # Check schema consistency
        self._validate_schema_consistency()
    
    def _validate_signal_id_relationship(self):
        """Validate signal_id as the relational key"""
        if 'train_signals' not in self.results or 'test_signals' not in self.results:
            logger.warning("Cannot validate signal_id relationship - missing data")
            return
        
        train_cols = self.results['train_signals'].get('columns', [])
        test_cols = self.results['test_signals'].get('columns', [])
        
        # Check if signal_id exists in both
        signal_id_in_train = 'signal_id' in train_cols
        signal_id_in_test = 'signal_id' in test_cols
        
        self.results['relationship_validation'] = {
            'signal_id_in_train_signals': signal_id_in_train,
            'signal_id_in_test_signals': signal_id_in_test,
            'signal_id_consistent': signal_id_in_train and signal_id_in_test
        }
        
        if signal_id_in_train and signal_id_in_test:
            logger.info("✅ signal_id confirmed as relational key between train and test signals")
        else:
            logger.warning("⚠️  signal_id not consistent as relational key")
    
    def _validate_schema_consistency(self):
        """Validate schema consistency between train and test files"""
        if 'train_signals' not in self.results or 'test_signals' not in self.results:
            return
        
        train_cols = set(self.results['train_signals'].get('columns', []))
        test_cols = set(self.results['test_signals'].get('columns', []))
        
        # Train should have target column, test should not
        has_target_in_train = 'eskalatsiya' in train_cols
        has_target_in_test = 'eskalatsiya' in test_cols
        
        self.results['schema_validation'] = {
            'train_columns': list(train_cols),
            'test_columns': list(test_cols),
            'has_target_in_train': has_target_in_train,
            'has_target_in_test': has_target_in_test,
            'schema_consistent': has_target_in_train and not has_target_in_test
        }
        
        if has_target_in_train and not has_target_in_test:
            logger.info("✅ Schema consistent: train has target, test does not")
        else:
            logger.warning("⚠️  Schema inconsistency detected")
    
    def generate_report(self) -> str:
        """Generate comprehensive audit report"""
        report = []
        report.append("=" * 80)
        report.append("WIUT FINTECH HACKATHON - DATA AUDIT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        for file_name, result in self.results.items():
            if 'error' not in result:
                size_mb = result.get('file_size_mb', 0)
                rows = result.get('total_row_count', result.get('row_count_sample', 'N/A'))
                cols = result.get('column_count', 'N/A')
                report.append(f"{file_name}: {rows} rows, {cols} cols, {size_mb:.2f} MB")
            else:
                report.append(f"{file_name}: ERROR - {result['error']}")
        
        report.append("")
        
        # Detailed results
        for file_name, result in self.results.items():
            report.append(f"{file_name.upper()}:")
            report.append("-" * 40)
            
            if 'error' in result:
                report.append(f"ERROR: {result['error']}")
            else:
                report.append(f"File: {result.get('file_path', 'N/A')}")
                report.append(f"Size: {result.get('file_size_mb', 0):.2f} MB")
                report.append(f"Rows: {result.get('total_row_count', result.get('row_count_sample', 'N/A'))}")
                report.append(f"Columns: {result.get('column_count', 'N/A')}")
                report.append(f"Column Names: {result.get('columns', [])}")
                
                if 'dtypes' in result:
                    report.append("Data Types:")
                    for col, dtype in result['dtypes'].items():
                        report.append(f"  {col}: {dtype}")
                
                if 'null_counts' in result:
                    report.append("Null Counts (sample):")
                    for col, count in result['null_counts'].items():
                        if count > 0:
                            pct = result['null_percentages'].get(col, 0)
                            report.append(f"  {col}: {count} ({pct:.1f}%)")
                
                if 'date_ranges' in result and result['date_ranges']:
                    report.append("Date Ranges:")
                    for col, ranges in result['date_ranges'].items():
                        report.append(f"  {col}: {ranges['min']} to {ranges['max']}")
                
                if 'potential_date_columns' in result:
                    report.append(f"Potential Date Columns: {result['potential_date_columns']}")
            
            report.append("")
        
        # Relationship validation
        if 'relationship_validation' in self.results:
            report.append("RELATIONSHIP VALIDATION:")
            report.append("-" * 40)
            for key, value in self.results['relationship_validation'].items():
                status = "✅" if value else "❌"
                report.append(f"{status} {key}: {value}")
            report.append("")
        
        # Schema validation
        if 'schema_validation' in self.results:
            report.append("SCHEMA VALIDATION:")
            report.append("-" * 40)
            for key, value in self.results['schema_validation'].items():
                status = "✅" if value else "❌"
                report.append(f"{status} {key}: {value}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run data audit"""
    auditor = DataAuditor()
    results = auditor.audit_all_files()
    
    # Generate and print report
    report = auditor.generate_report()
    print(report)
    
    # Save report to file
    report_path = Path("artifacts/data_audit_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Data audit report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    main()