# Chat Log Case Segmentation & Classification System

A sophisticated system for automatically segmenting customer support chat logs into individual cases and classifying them using Large Language Models (LLMs) with a modular DataFrame-based pipeline architecture.

## 🎯 Project Overview

This system processes large CSV files containing customer support conversations and intelligently segments them into discrete support cases, then classifies each case into a hierarchical taxonomy. The modular three-stage pipeline uses pandas DataFrames for seamless data flow and supports multiple LLM providers.

### Key Features

- 🔄 **Three-Stage DataFrame Pipeline**: Data Preprocessing → Case Segmentation → Case Classification
- 🤖 **Multi-Provider LLM Support**: OpenAI GPT, Anthropic Claude, and Google Gemini
- 📊 **Modular Architecture**: Independent stages with DataFrame input/output
- 🏷️ **Hierarchical Classification**: 9 primary categories, 62+ secondary categories
- 📈 **Comprehensive Analytics**: Token usage, confidence scoring, performance metrics
- ⚙️ **Flexible Configuration**: Easy model/provider switching
- 📤 **Rich Exports**: JSON, CSV, and Markdown reports with case assignments and classifications

## 🏗️ Pipeline Architecture

### Three-Stage Processing Pipeline

The system uses a streamlined three-stage approach optimized for performance and modularity:

```
📊 Raw CSV Data
    ↓
🔄 Stage 1: Data Preprocessing (DataPreprocessor)
    ↓ pandas DataFrame
🔄 Stage 2: Case Segmentation (ChannelSegmenter)  
    ↓ DataFrame + case_number
🔄 Stage 3: Case Classification (CaseClassifier)
    ↓ DataFrame + category
📤 Final Results
```

#### Stage 1: Data Preprocessing (`src/data_preprocessor.py`)
- Extracts necessary fields, cleans content, filters FILE messages
- Groups by channel, sorts by timestamp
- Returns optimized pandas DataFrame

#### Stage 2: Case Segmentation (`src/channel_segmenter.py`)
- Loads DataFrame, performs conversation analysis with LLM
- Implements token truncation using binary search
- Adds case_number and metadata columns to DataFrame

#### Stage 3: Case Classification (`src/case_classifier.py`)
- Classifies cases using hierarchical taxonomy (Primary_Secondary format)
- LLM-based classification with confidence scoring
- Adds category and classification metadata to DataFrame

### LLM Provider Abstraction

```python
# Unified interface supporting multiple providers
segmenter = ChannelSegmenter(
    llm_provider="anthropic",  # or "openai", "gemini"
    llm_model_type="default"   # default, high_quality, balanced, budget
)

classifier = CaseClassifier(
    llm_provider="anthropic",
    llm_model_type="default"
)
```

## 📁 Project Structure

```
case_segmentation/
├── README.md                    # This file
├── CLAUDE.md                   # Claude Code development guidance
├── requirements.txt            # Python dependencies
├── config.json                 # Model and algorithm configuration
├── .env                        # API keys (create from .env.example)
│
├── src/                        # Core implementation
│   ├── data_preprocessor.py    # Stage 1: Data preprocessing
│   ├── channel_segmenter.py    # Stage 2: Case segmentation (ChannelSegmenter)
│   ├── case_classifier.py      # Stage 3: Case classification
│   ├── llm_provider.py         # Multi-provider LLM abstraction
│   └── config_manager.py       # Configuration management
│
│   └── pipeline_runner.py      # Complete three-stage pipeline orchestrator
├── assets/                     # Data files
│   ├── support_msg.csv         # Raw dataset
│   └── preprocessed_support_msg.csv # Processed data
│
├── output/                     # Generated reports and exports
├── debug_output/               # LLM debugging information
└── tmp/                        # Archived and temporary files
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd case_segmentation

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set up API keys
cp .env.example .env
# Edit .env with your actual API keys:
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# GEMINI_API_KEY=your_gemini_key
```

### 3. Run Complete Pipeline

```bash
# Run complete three-stage pipeline
python src/pipeline_runner.py --mode r3

# Run without classification
python src/pipeline_runner.py --mode r3 --no-classify

# Save results and sample data
python src/pipeline_runner.py --mode kelvin --output results.csv --save-sample 10
```

### 4. Run Individual Stages

```bash
# Stage 1: Data preprocessing (returns DataFrame)
python src/data_preprocessor.py --dataframe --mode r3

# Stage 2: Case segmentation only
python src/channel_segmenter.py

# Stage 3: Demonstrate DataFrame-based classification
python -c "from src.case_classifier import demo_dataframe_classification; demo_dataframe_classification()"
```

## 📊 Pipeline Features

| Stage | Input | Output | Key Features |
|-------|-------|--------|-------------|
| **Preprocessing** | Raw CSV | Clean DataFrame | Field extraction, content cleaning, channel grouping |
| **Segmentation** | DataFrame | DataFrame + cases | LLM boundary detection, case_number assignment |
| **Classification** | DataFrame + cases | DataFrame + categories | Hierarchical taxonomy, confidence scoring |

## ⚙️ Configuration

### Model Selection

```json
{
  "llm": {
    "primary_provider": "anthropic",
    "providers": {
      "anthropic": {
        "models": {
          "default": "claude-sonnet-4-20250514",
          "high_quality": "claude-sonnet-4-20250514",
          "balanced": "claude-3-5-sonnet-20241022",
          "budget": "claude-3-haiku-20240307"
        }
      },
      "openai": {
        "models": {
          "default": "gpt-4o-mini",
          "high_quality": "gpt-4o",
          "balanced": "gpt-4-turbo",
          "budget": "gpt-3.5-turbo"
        }
      }
    }
  }
}
```

### Algorithm Parameters

```json
{
  "parsing": {
    "channel_algorithm": {
      "max_context_tokens": 100000,
      "reserve_tokens": 5000
    }
  }
}
```

## 🏷️ Classification Taxonomy

The system uses a hierarchical taxonomy with 9 primary categories and 62+ secondary categories:

### Primary Categories
- **Order**: Status, cancellations, updates, refunds
- **Shipment**: Tracking, delays, delivery issues
- **Payment**: Transactions, disputes, withdrawals
- **Tax and Fee**: Sales tax, invoices, forms
- **User**: Account management, credentials
- **Seller**: Applications, foundation plans
- **App Functionality**: Login, settings, bugs
- **Referral and Promotion**: Bonuses, credits
- **Other**: General issues, feedback

### Output Format
Categories are combined using underscore format: `Primary_Secondary`
- Example: `Payment_Withdrawal_Delay`, `Order_Refund_Request`

## 📈 Output Formats

### DataFrame Schema (Final Output)

```
Input columns (from preprocessing):
- review, created_time, sender_id, real_sender_id, message, message_id, type, channel_url, file_url, sender_type

Case columns (from segmentation):
- case_number, case_start_time, case_end_time, case_duration_minutes, case_confidence, case_summary, case_truncated, case_forced_ending, case_forced_starting

Classification columns (from classification):
- category, classification_confidence, classification_reasoning, classified_at
```

### Export Formats
- **CSV**: Complete DataFrame with all pipeline data
- **JSON**: Structured case data with metadata and statistics
- **Markdown**: Human-readable summaries and analytics reports

## 🛠️ Advanced Usage

### Complete Pipeline with Custom Configuration

```python
from src.pipeline_runner import PipelineRunner

# Create pipeline runner
runner = PipelineRunner()

# Run complete pipeline
final_df = runner.run_complete_pipeline(
    input_file='assets/support_msg.csv',
    output_file='results.csv',
    mode='r3',
    skip_classification=False
)

print(f"Final DataFrame shape: {final_df.shape}")
print(f"Unique cases: {final_df['case_number'].nunique()}")
print(f"Categories found: {final_df['category'].nunique()}")
```

### Individual Stage Usage

```python
from src.data_preprocessor import DataPreprocessor
from src.channel_segmenter import ChannelSegmenter
from src.case_classifier import CaseClassifier

# Stage 1: Preprocessing
preprocessor = DataPreprocessor()
df = preprocessor.process_to_dataframe('assets/support_msg.csv', mode='r3')

# Stage 2: Case Segmentation
segmenter = ChannelSegmenter(llm_provider="anthropic")
case_df = segmenter.process_dataframe(df)

# Stage 3: Classification
classifier = CaseClassifier(llm_provider="anthropic")
final_df = classifier.classify_dataframe(case_df)
```

### Custom Implementation

```python
from src.channel_segmenter import ChannelSegmenter

# Custom configuration
segmenter = ChannelSegmenter(
    llm_provider="openai",
    llm_model_type="high_quality"
)

# Process with DataFrame input/output
output_df = segmenter.process_dataframe(input_df)

# Export with custom paths
segmenter.export_json('custom_output.json')
segmenter.export_cases_csv('custom_cases.csv')  
segmenter.export_segmentation_summary_md('custom_summary.md')
```

## 📋 Data Format

### Input CSV Structure
```csv
message_id,type,message,sender_id,created_time,channel_url,file_url,sender_type,review
msg_001,MESG,"Hello, I need help",user123,2024-01-01T10:00:00Z,channel_001,,user,1
msg_002,MESG,"How can I assist you?",psops000_agent,2024-01-01T10:01:00Z,channel_001,,customer_service,1
```

### Supported Message Types
- **MESG**: Text messages (processed)
- **FILE**: File attachments (filtered out)
- **SYSTEM**: System messages (processed)

## 🔬 Testing & Validation

### Run Complete Pipeline Tests
```bash
# Test with demo data (3 representative channels)
python src/pipeline_runner.py --mode r3

# Test with specific channels
python src/pipeline_runner.py --mode kelvin

# Test segmentation only
python src/pipeline_runner.py --mode r3 --no-classify

# Test with sample output
python src/pipeline_runner.py --mode r3 --save-sample 10
```

### Performance Metrics
- **Segmentation Quality**: Boundary detection confidence scores
- **Classification Accuracy**: Category assignment confidence
- **Processing Efficiency**: Tokens per case, processing time per stage
- **Cost Analysis**: Provider-specific token usage and costs

## 🚨 Troubleshooting

### Common Issues

**Rate Limits**
```bash
# Switch to budget models in config.json
# Or add delays between API calls
```

**Memory Issues**
```bash
# Reduce token limits for large conversations
# Edit config.json: "max_context_tokens": 50000
```

**Classification Errors**
```bash
# Check debug_output/ for classification interaction logs
# Verify taxonomy categories in case_classifier.py
```

### Debug Mode
```python
# Debug information is automatically saved to debug_output/
# Check these files when LLM parsing or classification fails
ls debug_output/
```

## 📈 Performance Benchmarks

### Sample Results (764 messages, 1 channel, r3 mode)
- **Preprocessing**: 0.12s (764 rows → DataFrame)
- **Segmentation**: ~41s (7 cases, avg confidence: 0.72)
- **Classification**: ~35s (7/7 cases classified, avg confidence: 0.83)
- **Total Pipeline**: ~76s for complete processing

### Quality Indicators
- **High Segmentation Quality**: Confidence ≥ 0.8
- **High Classification Coverage**: 90%+ cases successfully classified
- **Cost Efficiency**: Optimized token usage with binary search truncation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow existing DataFrame pipeline patterns
- Maintain modularity between stages
- Add tests for new features
- Update documentation and configuration
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI, Anthropic, and Google for LLM APIs
- pandas community for DataFrame functionality
- StreamingLLM research (MIT 2024)
- Topic segmentation research community

## 📞 Support

For questions and support:
1. Check the [CLAUDE.md](CLAUDE.md) for development guidance
2. Review debug_output/ for LLM interaction logs
3. Open an issue for bugs or feature requests

---

*Built with ❤️ for better customer support analytics using modular DataFrame architecture*