#!/usr/bin/env python3
"""
Pipeline Runner - Three-Stage DataFrame Processing

This script demonstrates the complete three-stage DataFrame pipeline:
1. Data Preprocessing (CSV → DataFrame)
2. Case Segmentation (DataFrame → DataFrame with Case Number)
3. Case Classification (DataFrame → DataFrame with category)

Usage:
    python src/pipeline_runner.py --mode r3 --save-output output.csv
    python src/pipeline_runner.py --mode kelvin --no-classify
    python src/pipeline_runner.py --input custom_data.csv --output results.csv
"""

import argparse
import time
import pandas as pd
import sys
import os
from datetime import datetime
from typing import Optional

# Since we're already in src directory, add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import our three pipeline stages
from data_preprocessor import DataPreprocessor
from channel_segmenter import ChannelSegmenter
from case_classifier import CaseClassifier


class PipelineRunner:
    """Orchestrates the three-stage DataFrame pipeline"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.case_parser = ChannelSegmenter()
        self.classifier = CaseClassifier()
        
        # Pipeline statistics
        self.start_time = None
        self.stage_times = {}
        self.stage_results = {}
    
    def run_complete_pipeline(self, 
                            input_file: str, 
                            output_file: Optional[str] = None,
                            mode: Optional[str] = None,
                            skip_data: bool = False,
                            skip_channel: bool = False,
                            skip_case: bool = False) -> pd.DataFrame:
        """
        Run the complete three-stage pipeline.
        
        Args:
            input_file: Input CSV file path
            output_file: Optional output CSV file path. If None, updates input file with minimal columns
            mode: Processing mode ('r3', 'kelvin', or None for full dataset)
            skip_data: If True, skip data preprocessing stage
            skip_channel: If True, skip channel segmentation stage  
            skip_case: If True, skip case classification stage
            
        Returns:
            Final DataFrame with all pipeline results
        """
        self.start_time = time.time()
        print("=" * 60)
        print("🚀 STARTING THREE-STAGE DATAFRAME PIPELINE")
        print("=" * 60)
        print(f"📁 Input: {input_file}")
        print(f"📊 Mode: {mode if mode else 'full dataset'}")
        print(f"🔄 Data Preprocessing: {'skipped' if skip_data else 'enabled'}")
        print(f"🔗 Channel Segmentation: {'skipped' if skip_channel else 'enabled'}")
        print(f"🏷️ Case Classification: {'skipped' if skip_case else 'enabled'}")
        if output_file:
            print(f"💾 Output: {output_file}")
        else:
            print(f"💾 Output: Will update {input_file} with Case Number and Category columns only")
        print()
        
        # Stage 1: Data Preprocessing
        if not skip_data:
            stage1_start = time.time()
            print("🔄 STAGE 1: DATA PREPROCESSING")
            print("-" * 40)
            
            df_preprocessed = self.preprocessor.process_to_dataframe(input_file, mode)
            
            stage1_time = time.time() - stage1_start
            self.stage_times['preprocessing'] = stage1_time
            self.stage_results['preprocessing'] = {
                'input_rows': self.preprocessor.stats.total_rows,
                'output_rows': len(df_preprocessed),
                'channels': df_preprocessed['Channel URL'].nunique(),
                'columns': len(df_preprocessed.columns)
            }
            
            print(f"✅ Stage 1 complete: {len(df_preprocessed)} rows, {len(df_preprocessed.columns)} columns")
            print(f"⏱️  Time: {stage1_time:.2f}s")
            print()
        else:
            print("⏭️  STAGE 1: DATA PREPROCESSING (SKIPPED)")
            print("-" * 40)
            print("Loading input file directly as DataFrame...")
            df_preprocessed = pd.read_csv(input_file)
            self.stage_results['preprocessing'] = {
                'input_rows': len(df_preprocessed),
                'output_rows': len(df_preprocessed),
                'channels': df_preprocessed['Channel URL'].nunique() if 'Channel URL' in df_preprocessed.columns else 0,
                'columns': len(df_preprocessed.columns)
            }
            print(f"✅ Input loaded: {len(df_preprocessed)} rows, {len(df_preprocessed.columns)} columns")
            print()
        
        # Stage 2: Case Segmentation
        if not skip_channel:
            stage2_start = time.time()
            print("🔄 STAGE 2: CASE SEGMENTATION")
            print("-" * 40)
            
            df_with_cases = self.case_parser.process_dataframe(df_preprocessed)
            
            stage2_time = time.time() - stage2_start
            self.stage_times['segmentation'] = stage2_time
            unique_cases = df_with_cases['Case Number'].nunique()
            case_columns = [col for col in df_with_cases.columns if col.startswith('case_')]
            self.stage_results['segmentation'] = {
                'unique_cases': unique_cases,
                'case_columns_added': len(case_columns),
                'avg_confidence': df_with_cases['case_confidence'].mean(),
                'total_columns': len(df_with_cases.columns)
            }
            
            print(f"✅ Stage 2 complete: {unique_cases} cases identified")
            print(f"⏱️  Time: {stage2_time:.2f}s")
            
            # Export segmentation markdown report
            print("📋 Exporting segmentation summary...")
            self.case_parser.export_segmentation_summary_md()
            print()
        else:
            print("⏭️  STAGE 2: CASE SEGMENTATION (SKIPPED)")
            print("-" * 40)
            df_with_cases = df_preprocessed.copy()
            # Add empty Case Number column if it doesn't exist
            if 'Case Number' not in df_with_cases.columns:
                df_with_cases['Case Number'] = ''
            self.stage_results['segmentation'] = {
                'unique_cases': 0,
                'case_columns_added': 0,
                'avg_confidence': 0.0,
                'total_columns': len(df_with_cases.columns)
            }
            print("✅ Stage 2 skipped")
            print()
        
        # Stage 3: Case Classification (optional)
        if not skip_case:
            stage3_start = time.time()
            print("🔄 STAGE 3: CASE CLASSIFICATION")
            print("-" * 40)
            
            df_final = self.classifier.classify_dataframe(df_with_cases)
            
            stage3_time = time.time() - stage3_start
            self.stage_times['classification'] = stage3_time
            classified_cases = df_final[df_final['Category'] != '']['Case Number'].nunique()
            unique_categories = df_final['Category'].nunique()
            self.stage_results['classification'] = {
                'classified_cases': classified_cases,
                'unique_categories': unique_categories,
                'avg_classification_confidence': df_final[df_final['classification_confidence'] > 0]['classification_confidence'].mean(),
                'total_columns': len(df_final.columns)
            }
            
            print(f"✅ Stage 3 complete: {classified_cases} cases classified into {unique_categories} categories")
            print(f"⏱️  Time: {stage3_time:.2f}s")
            
            # Export classification markdown report
            if classified_cases > 0:
                print("📋 Exporting classification summary...")
                from case_classifier import CaseClassification
                
                # Create classification objects for export
                classifications = []
                for _, row in df_final[df_final['Category'] != ''].iterrows():
                    if row['Case Number'] and row['Category']:
                        classification = CaseClassification(
                            case_id=row['Case Number'],
                            category=row['Category'],
                            confidence=row['classification_confidence'],
                            reasoning=row['classification_reasoning'],
                            classified_at=pd.to_datetime(row['classified_at']).to_pydatetime()
                        )
                        classifications.append(classification)
                
                # Remove duplicates by case_id
                unique_classifications = {}
                for c in classifications:
                    unique_classifications[c.case_id] = c
                unique_classifications_list = list(unique_classifications.values())
                
                if unique_classifications_list:
                    self.classifier.export_classification_summary_md(unique_classifications_list)
            print()
        else:
            df_final = df_with_cases.copy()
            # Add empty Category column if it doesn't exist
            if 'Category' not in df_final.columns:
                df_final['Category'] = ''
            print("⏭️  STAGE 3: CASE CLASSIFICATION (SKIPPED)")
            print("-" * 40)
            print("✅ Stage 3 skipped")
            print()
        
        # Save output
        if output_file:
            print(f"💾 Saving complete results to {output_file}...")
            df_final.to_csv(output_file, index=False)
            print(f"✅ Saved {len(df_final)} rows with {len(df_final.columns)} columns to {output_file}")
            print()
        else:
            print(f"💾 Updating source file {input_file}...")
            # Load original input file
            original_df = pd.read_csv(input_file)
            
            # Add only Case Number and Category columns to original
            output_df = original_df.copy()
            
            # Create mapping from message_id to Case Number and Category
            case_mapping = {}
            category_mapping = {}
            
            for _, row in df_final.iterrows():
                msg_id = str(row['Message ID'])
                if 'Case Number' in row and row['Case Number']:
                    case_mapping[msg_id] = str(row['Case Number'])
                if 'Category' in row and row['Category']:
                    category_mapping[msg_id] = str(row['Category'])
            
            # Add Case Number column
            if 'Case Number' not in output_df.columns:
                output_df['Case Number'] = ''
            
            # Add Category column  
            if 'Category' not in output_df.columns:
                output_df['Category'] = ''
            
            # Apply mappings
            for i, row in output_df.iterrows():
                msg_id = str(row['Message ID'])
                if msg_id in case_mapping:
                    output_df.at[i, 'Case Number'] = case_mapping[msg_id]
                if msg_id in category_mapping:
                    output_df.at[i, 'Category'] = category_mapping[msg_id]
            
            # Save back to input file
            output_df.to_csv(input_file, index=False)
            
            cases_added = len([v for v in case_mapping.values() if v])
            categories_added = len([v for v in category_mapping.values() if v])
            print(f"✅ Updated {input_file} with {cases_added} case assignments and {categories_added} category assignments")
            print()
        
        # Print final summary
        self._print_pipeline_summary(df_final, skip_data, skip_channel, skip_case)
        
        return df_final
    
    def _print_pipeline_summary(self, final_df: pd.DataFrame, skip_data: bool, skip_channel: bool, skip_case: bool):
        """Print comprehensive pipeline summary"""
        total_time = time.time() - self.start_time
        
        print("=" * 60)
        print("📈 PIPELINE SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Final DataFrame shape: {final_df.shape}")
        print()
        
        # Stage breakdown
        if self.stage_times:
            print("⏱️  Stage Timing:")
            for stage, duration in self.stage_times.items():
                percentage = (duration / total_time) * 100
                print(f"  {stage.title()}: {duration:.2f}s ({percentage:.1f}%)")
            print()
        
        # Data flow summary
        print("📊 Data Flow:")
        if 'preprocessing' in self.stage_results:
            prep_stats = self.stage_results['preprocessing']
            print(f"  Input: {prep_stats['input_rows']:,} rows")
            if not skip_data:
                print(f"  After preprocessing: {prep_stats['output_rows']:,} rows ({prep_stats['channels']} channels)")
            else:
                print(f"  Preprocessing: SKIPPED")
        
        if 'segmentation' in self.stage_results:
            seg_stats = self.stage_results['segmentation']
            if not skip_channel:
                print(f"  After segmentation: {seg_stats['unique_cases']} cases (avg conf: {seg_stats['avg_confidence']:.3f})")
            else:
                print(f"  Segmentation: SKIPPED")
        
        if not skip_case and 'classification' in self.stage_results:
            class_stats = self.stage_results['classification']
            print(f"  After classification: {class_stats['classified_cases']} cases classified")
            print(f"  Categories identified: {class_stats['unique_categories']}")
            if not pd.isna(class_stats['avg_classification_confidence']):
                print(f"  Avg classification confidence: {class_stats['avg_classification_confidence']:.3f}")
        elif skip_case:
            print(f"  Classification: SKIPPED")
        print()
        
        # Quality indicators
        print("✅ Quality Indicators:")
        
        if not skip_channel and 'segmentation' in self.stage_results:
            seg_stats = self.stage_results['segmentation']
            avg_seg_conf = seg_stats['avg_confidence']
            if avg_seg_conf >= 0.8:
                print(f"  🟢 High segmentation quality (confidence: {avg_seg_conf:.3f})")
            elif avg_seg_conf >= 0.6:
                print(f"  🟡 Medium segmentation quality (confidence: {avg_seg_conf:.3f})")
            else:
                print(f"  🔴 Low segmentation quality (confidence: {avg_seg_conf:.3f})")
        
        if not skip_case and 'classification' in self.stage_results and 'segmentation' in self.stage_results:
            class_stats = self.stage_results['classification']
            seg_stats = self.stage_results['segmentation']
            if seg_stats['unique_cases'] > 0:
                class_rate = class_stats['classified_cases'] / seg_stats['unique_cases']
                if class_rate >= 0.9:
                    print(f"  🟢 High classification coverage ({class_rate:.1%})")
                elif class_rate >= 0.7:
                    print(f"  🟡 Medium classification coverage ({class_rate:.1%})")
                else:
                    print(f"  🔴 Low classification coverage ({class_rate:.1%})")
        
        print()
        print("=" * 60)


def main():
    """Command-line interface for the pipeline runner"""
    parser = argparse.ArgumentParser(
        description='Run the three-stage DataFrame pipeline for case segmentation and classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_runner.py --mode r3
  python pipeline_runner.py --mode kelvin --output results.csv
  python pipeline_runner.py --input custom_data.csv --skip-case
  python pipeline_runner.py --mode r3 --skip-data --skip-channel
  python pipeline_runner.py --input preprocessed.csv --skip-data
        """
    )
    
    parser.add_argument('--input', default='assets/preprocessed_support_msg.csv',
                       help='Input CSV file path (default: assets/support_msg.csv)')
    parser.add_argument('--output', 
                       help='Output CSV file path (optional)')
    parser.add_argument('--mode', choices=['r3', 'kelvin'],
                       help='Processing mode: r3 (3 representative channels) or kelvin (specific channels)')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip the data preprocessing stage')
    parser.add_argument('--skip-channel', action='store_true',
                       help='Skip the channel segmentation stage')
    parser.add_argument('--skip-case', action='store_true',
                       help='Skip the case classification stage')
    
    args = parser.parse_args()
    
    # Validate skip arguments - at least one stage must run
    if args.skip_data and args.skip_channel and args.skip_case:
        print("❌ Error: Cannot skip all stages. At least one stage must be enabled.")
        return
    
    # Run the pipeline
    runner = PipelineRunner()
    
    try:
        runner.run_complete_pipeline(
            input_file=args.input,
            output_file=args.output,
            mode=args.mode,
            skip_data=args.skip_data,
            skip_channel=args.skip_channel,
            skip_case=args.skip_case
        )
        
        print(f"\n🎉 Pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found - {e}")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()