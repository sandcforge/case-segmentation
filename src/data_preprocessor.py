#!/usr/bin/env python3
"""
Data Preprocessor for Chat Log Case Segmentation

This class preprocesses the raw CSV file to:
1. Extract only necessary fields for case parsing
2. Clean invalid/illegal characters
3. Group by channel and sort by timestamp
4. Optionally create demo datasets with representative channels
5. Save optimized data for faster loading
"""

import csv
import re
import argparse
import random
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass

# Pre-compiled regex patterns for performance (reused from case_parser_channel.py)
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
WHITESPACE_PATTERN = re.compile(r'\s+')


@dataclass
class ProcessingStats:
    """Statistics for preprocessing operation"""
    total_rows: int = 0
    processed_rows: int = 0
    filtered_rows: int = 0
    error_rows: int = 0
    channels_found: int = 0
    channels_selected: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ChannelInfo:
    """Information about a channel for demo selection"""
    channel_url: str
    message_count: int
    size_category: str  # 'many', 'medium', 'few'


class DataPreprocessor:
    """Preprocesses raw CSV data for optimal case parsing performance"""
    
    def __init__(self):
        self.stats = ProcessingStats()
        self.channel_message_counts: Dict[str, int] = defaultdict(int)
        
    def clean_message_content(self, content: str) -> str:
        """Clean message content to remove invalid control characters"""
        if not content:
            return ""
        
        try:
            # Remove control characters except newlines and tabs
            cleaned = CONTROL_CHAR_PATTERN.sub('', content)
            
            # Replace problematic quotes and characters
            cleaned = cleaned.replace('"', '"').replace('"', '"')
            cleaned = cleaned.replace(''', "'").replace(''', "'")
            
            # Remove null bytes and other problematic characters
            cleaned = cleaned.replace('\x00', '').replace('\ufffd', '')
            
            # Replace multiple whitespace with single space
            cleaned = WHITESPACE_PATTERN.sub(' ', cleaned)
            
            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Ensure we have valid content
            if not cleaned:
                return "[empty message]"
            
            return cleaned
            
        except Exception as e:
            print(f"Error cleaning message content: {e}")
            return "[invalid message content]"
    
    def analyze_channels(self, input_file: str) -> Dict[str, int]:
        """Analyze channels to count messages for demo selection"""
        print("📊 Analyzing channels for demo selection...")
        
        channel_counts = defaultdict(int)
        
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if i % 10000 == 0:
                    print(f"  Analyzed {i} rows...")
                
                try:
                    channel_url = row.get('channel_url', '')
                    if channel_url:
                        channel_counts[channel_url] += 1
                except Exception:
                    continue
        
        print(f"  Found {len(channel_counts)} channels")
        return dict(channel_counts)
    
    def select_r3_channels(self, channel_counts: Dict[str, int]) -> List[str]:
        """Select 3 representative channels for r3 mode"""
        # Categorize channels by message count
        many_channels = []    # >50 messages
        medium_channels = []  # 10-50 messages
        few_channels = []     # <10 messages
        
        for channel_url, count in channel_counts.items():
            if count > 50:
                many_channels.append((channel_url, count))
            elif count >= 10:
                medium_channels.append((channel_url, count))
            else:
                few_channels.append((channel_url, count))
        
        # Sort by message count for better selection
        many_channels.sort(key=lambda x: x[1], reverse=True)
        medium_channels.sort(key=lambda x: x[1], reverse=True)
        few_channels.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        
        # Select one from each category
        if many_channels:
            # Pick randomly from top 5 largest channels
            candidates = many_channels[:min(5, len(many_channels))]
            selected_channel = random.choice(candidates)
            selected.append(selected_channel[0])
            print(f"  Selected 'many messages' channel: {selected_channel[1]} messages")
        
        if medium_channels:
            # Pick randomly from medium channels
            selected_channel = random.choice(medium_channels)
            selected.append(selected_channel[0])
            print(f"  Selected 'medium messages' channel: {selected_channel[1]} messages")
        
        if few_channels:
            # Pick randomly from few channels
            selected_channel = random.choice(few_channels)
            selected.append(selected_channel[0])
            print(f"  Selected 'few messages' channel: {selected_channel[1]} messages")
        
        print(f"📋 Selected {len(selected)} channels for r3 mode")
        return selected
    
    def select_kelvin_channels(self, channel_counts: Dict[str, int]) -> List[str]:
        """Select specific channels for Kelvin mode"""
        kelvin_channels = [
            'sendbird_group_channel_215482988_c05f4430399a29e30820acdfef8a267d81a3400b',
            'sendbird_group_channel_215482988_da7183281699b7999e2677616b1e2a0e12c6c224', 
            'sendbird_group_channel_215482988_dac86e93e1a8af85e898adf1317edfc157fd42db'
        ]
        
        selected = []
        for channel_url in kelvin_channels:
            if channel_url in channel_counts:
                selected.append(channel_url)
                print(f"  Selected Kelvin channel: {channel_counts[channel_url]} messages")
            else:
                print(f"  ⚠️ Kelvin channel not found: {channel_url[:50]}...")
        
        print(f"📋 Selected {len(selected)} channels for Kelvin mode")
        return selected
    
    def process_csv(self, input_file: str, output_file: str, mode: str = None) -> ProcessingStats:
        """Process CSV file and create optimized output"""
        self.stats = ProcessingStats()
        self.stats.start_time = datetime.now()
        self._mode = mode  # Store for statistics display
        
        print(f"🚀 Starting data preprocessing...")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Mode: {mode if mode else 'full dataset'}")
        
        # Required columns mapping (keep original field names for compatibility)
        required_columns = {
            'review': 'review',  # First column with blank values
            'created_time': 'created_time',
            'sender_id': 'sender_id',
            'message': 'message',
            'message_id': 'message_id',
            'type': 'type',
            'channel_url': 'channel_url',
            'file_url': 'file_url',
            'sender_type': 'sender_type'  # New column to be computed
        }
        
        # Mode-based channel selection
        selected_channels = None
        if mode:
            channel_counts = self.analyze_channels(input_file)
            if mode == 'kelvin':
                selected_channels = set(self.select_kelvin_channels(channel_counts))
            elif mode == 'r3':
                selected_channels = set(self.select_r3_channels(channel_counts))
            if selected_channels:
                self.stats.channels_selected = len(selected_channels)
        
        # Process the CSV file
        processed_data = []
        
        print("📁 Processing CSV data...")
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for i, row in enumerate(reader):
                self.stats.total_rows += 1
                
                if i % 10000 == 0:
                    print(f"  Processed {i} rows...")
                
                try:
                    # Mode-based channel filtering
                    if mode and selected_channels:
                        channel_url = row.get('channel_url', '')
                        if channel_url not in selected_channels:
                            self.stats.filtered_rows += 1
                            continue
                    
                    # Extract and validate required fields
                    extracted_row = {}
                    sender_id = row.get('sender_id', '')  # Get sender_id early for sender_type logic
                    
                    for csv_col, field_name in required_columns.items():
                        if field_name == 'review':
                            # Review column starts blank
                            value = ""
                        elif field_name == 'sender_type':
                            # Compute sender_type based on sender_id
                            value = "customer_service" if sender_id.startswith("psops") else "user"
                        else:
                            value = row.get(csv_col, '')
                            
                            # Special handling for different field types
                            if field_name == 'message':
                                # Clean message content
                                value = self.clean_message_content(value)
                            elif field_name == 'created_time':
                                # Validate timestamp format
                                if value:
                                    try:
                                        # Parse to validate format
                                        datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    except Exception:
                                        print(f"Warning: Invalid timestamp in row {i}: {value}")
                                        value = ''
                        
                        extracted_row[field_name] = value
                    
                    # Skip rows with missing essential data
                    if not extracted_row.get('message_id') or not extracted_row.get('channel_url'):
                        self.stats.filtered_rows += 1
                        continue
                    
                    # Track channel
                    channel_url = extracted_row['channel_url']
                    self.channel_message_counts[channel_url] += 1
                    
                    processed_data.append(extracted_row)
                    self.stats.processed_rows += 1
                    
                except Exception as e:
                    self.stats.error_rows += 1
                    if self.stats.error_rows <= 5:  # Only show first few errors
                        print(f"Error processing row {i}: {e}")
                    continue
        
        self.stats.channels_found = len(self.channel_message_counts)
        
        # Group by channel and sort by timestamp
        print("📊 Grouping by channel and sorting...")
        channels_data = defaultdict(list)
        
        for row in processed_data:
            channels_data[row['channel_url']].append(row)
        
        # Sort messages within each channel by timestamp
        sorted_data = []
        for channel_url, messages in channels_data.items():
            # Sort by timestamp
            messages.sort(key=lambda x: x['created_time'] or '0')
            sorted_data.extend(messages)
        
        # Write processed data to output file
        print("💾 Writing processed data...")
        fieldnames = list(required_columns.values())
        
        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_data)
        
        self.stats.end_time = datetime.now()
        return self.stats
    
    def print_statistics(self):
        """Print comprehensive processing statistics"""
        if not self.stats.start_time or not self.stats.end_time:
            print("No statistics available")
            return
        
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("📈 PREPROCESSING STATISTICS")
        print("="*60)
        print(f"Total rows processed:     {self.stats.total_rows:,}")
        print(f"Successfully processed:   {self.stats.processed_rows:,}")
        print(f"Filtered out:             {self.stats.filtered_rows:,}")
        print(f"Error rows:               {self.stats.error_rows:,}")
        print(f"Channels found:           {self.stats.channels_found:,}")
        
        if self.stats.channels_selected > 0:
            mode_name = getattr(self, '_mode', 'unknown')
            print(f"Channels selected ({mode_name}): {self.stats.channels_selected}")
        
        print(f"Processing time:          {duration:.2f} seconds")
        print(f"Rows per second:          {self.stats.total_rows / duration:.0f}")
        
        # Show channel size distribution
        if self.channel_message_counts:
            print("\n📊 Channel Size Distribution:")
            sizes = list(self.channel_message_counts.values())
            sizes.sort(reverse=True)
            
            many = len([s for s in sizes if s > 50])
            medium = len([s for s in sizes if 10 <= s <= 50])
            few = len([s for s in sizes if s < 10])
            
            print(f"  Many messages (>50):    {many} channels")
            print(f"  Medium messages (10-50): {medium} channels")
            print(f"  Few messages (<10):     {few} channels")
            
            if sizes:
                print(f"  Largest channel:        {sizes[0]} messages")
                print(f"  Average messages:       {sum(sizes) / len(sizes):.1f}")
        
        print("="*60)


def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess chat log CSV data for case segmentation')
    parser.add_argument('--input', default='assets/support_msg.csv', 
                       help='Input CSV file path (default: assets/support_msg.csv)')
    parser.add_argument('--output', default='assets/preprocessed_support_msg.csv',
                       help='Output CSV file path (default: assets/preprocessed_support_msg.csv)')
    parser.add_argument('--mode', choices=['r3', 'kelvin'],
                       help='Processing mode: r3 (3 representative channels) or kelvin (specific channels)')
    
    args = parser.parse_args()
    
    
    # Create preprocessor and process data
    preprocessor = DataPreprocessor()
    
    try:
        stats = preprocessor.process_csv(
            input_file=args.input,
            output_file=args.output,
            mode=args.mode
        )
        
        # Print statistics
        preprocessor.print_statistics()
        
        print(f"\n✅ Preprocessing complete!")
        print(f"📁 Output saved to: {args.output}")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{args.input}' not found")
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")


if __name__ == "__main__":
    main()