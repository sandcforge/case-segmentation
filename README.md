# Chat Log Case Segmentation Parser

A sophisticated system for automatically segmenting customer support chat logs into individual cases using Large Language Models (LLMs) with configurable algorithms and providers.

## 🎯 Project Overview

This system processes large CSV files containing customer support conversations and intelligently segments them into discrete support cases using AI-powered boundary detection. It supports multiple algorithms, LLM providers, and provides comprehensive analytics.

### Key Features

- 🤖 **Optimized Algorithm**: Single high-performance conversation analysis approach
- 🔄 **Multi-Provider LLM**: OpenAI GPT and Anthropic Claude support
- 📊 **Token Management**: Intelligent binary search truncation for large conversations
- 📈 **Comprehensive Analytics**: Token usage, confidence scoring, performance metrics
- ⚙️ **Flexible Configuration**: Easy model/provider switching
- 📤 **Rich Exports**: JSON, CSV, and Markdown reports with case assignments

## 🏗️ Architecture

### Channel Full Conversation Algorithm

The system uses a single, optimized algorithm designed for accuracy and performance:

**Channel Full Conversation** (`src/case_parser_channel.py`)
- Loads entire channel conversations at once for complete context
- Single LLM analysis with comprehensive boundary detection
- Binary search truncation for efficient token limit management
- Built-in review and validation for improved accuracy
- Optimized for customer support conversation patterns

### LLM Provider Abstraction

```python
# Unified interface supporting multiple providers
parser = ChannelCaseParser(
    llm_provider="anthropic",  # or "openai"
    llm_model_type="default"   # default, high_quality, balanced, budget
)
```

## 📁 Project Structure

```
case_segmentation/
├── README.md                    # This file
├── CLAUDE.md                   # Claude Code development guidance
├── PRD.md                      # Product Requirements Document
├── requirements.txt            # Python dependencies
├── config.json                 # Model and algorithm configuration
├── .env                        # API keys (create from .env.example)
│
├── src/                        # Core implementation
│   ├── case_parser_channel.py  # Main case segmentation algorithm
│   ├── data_preprocessor.py    # Data preprocessing and cleaning
│   ├── llm_provider.py         # Multi-provider LLM abstraction
│   └── config_manager.py       # Configuration management
│
├── assets/                     # Data files
│   ├── support_msg.csv         # Raw dataset
│   └── preprocessed_support_msg.csv # Processed data
│
├── output/                     # Generated reports and exports
├── debug_output/               # LLM debugging information
├── tmp/                        # Archived and temporary files
└── test_results/               # Test outputs and comparisons
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
```

### 3. Run Your First Segmentation

```bash
# Preprocess your data first
python src/data_preprocessor.py --demo  # Use demo mode for testing

# Run case segmentation
python src/case_parser_channel.py

# Check the generated reports
ls output/
```

## 📊 Algorithm Features

The Channel Full Conversation algorithm provides optimal balance of features:

| Feature | Channel Full Conversation |
|---------|---------------------------|
| **Approach** | Complete conversation analysis |
| **Memory** | Efficient with binary search truncation |
| **Context** | Full conversation context maintained |
| **Speed** | Single-pass optimized processing |
| **Accuracy** | Excellent with built-in review |
| **Best For** | Customer support conversations |

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

## 📈 Output Formats

### JSON Export
```json
{
  "summary": {
    "total_cases": 12,
    "total_channels": 3,
    "average_confidence": 0.894,
    "total_tokens": 15420,
    "algorithm": "Channel Full Conversation"
  },
  "cases": [
    {
      "case_id": "CASE_0001",
      "summary": "Customer issue with order delivery",
      "confidence": 0.95,
      "duration_minutes": 15.3,
      "forced_ending": false,
      "truncated": false,
      "messages": [...],
      "participants": ["user123", "support_agent"]
    }
  ]
}
```

### Export Formats
- **JSON**: Complete case data with metadata and statistics
- **CSV**: Message-level data with case assignments and review flags
- **Markdown**: Human-readable case summaries and analytics reports

## 🔬 Testing & Validation

### Run Tests
```bash
# Test with demo data
python src/data_preprocessor.py --demo
python src/case_parser_channel.py

# Process full dataset
python src/data_preprocessor.py
python src/case_parser_channel.py

# Check results
ls output/
cat output/cases_channel_segmentation_summary.md
```

### Performance Metrics
- **Accuracy**: Boundary detection precision/recall
- **Efficiency**: Tokens per case, processing time
- **Cost**: Provider-specific cost analysis
- **Quality**: Confidence scores and manual validation

## 🛠️ Advanced Usage

### Custom Implementation
```python
from src.case_parser_channel import ChannelCaseParser

# Custom configuration
parser = ChannelCaseParser(
    llm_provider="openai",
    llm_model_type="high_quality"
)

# Process specific channels
channels = parser.load_csv('assets/preprocessed_support_msg.csv')
cases = parser.process_all_channels(channels)

# Export with custom paths
parser.export_json('custom_output.json')
parser.export_cases_csv('custom_cases.csv')  
parser.export_segmentation_summary_md('custom_summary.md')
```

### Data Preprocessing
```python
from src.data_preprocessor import DataPreprocessor

# Create preprocessor and clean data
preprocessor = DataPreprocessor()
stats = preprocessor.process_csv(
    input_file='assets/support_msg.csv',
    output_file='assets/preprocessed_support_msg.csv',
    demo_mode=False  # Set to True for demo
)
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

## 🧪 Research Foundation

### Conversation Analysis Optimization
- Full conversation context analysis for optimal boundary detection
- Binary search truncation for efficient token management
- Multi-strategy JSON parsing with fallback mechanisms

### LLM Prompt Engineering
- Comprehensive boundary detection criteria with primary and secondary indicators
- Built-in review and validation steps for improved accuracy
- Confidence scoring and quality assessment integration

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

**Token Limits**
```bash
# System automatically truncates with binary search
# Check debug_output/ for truncation details
```

### Debug Mode
```python
# Debug information is automatically saved to debug_output/
# Check these files when LLM parsing fails
ls debug_output/
```

## 📈 Performance Benchmarks

### Sample Results (809 messages, 3 channels)
- **Channel Algorithm**: High accuracy with complete context analysis
- **Token Usage**: Optimized with binary search truncation
- **Average Confidence**: 0.8+ for most cases with built-in review
- **Processing**: Single-pass analysis with comprehensive boundary detection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow existing code patterns
- Add tests for new features
- Update documentation
- Maintain configuration compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- StreamingLLM research (MIT 2024)
- Topic segmentation research community
- OpenAI and Anthropic for LLM APIs
- Sentence Transformers library

## 📞 Support

For questions and support:
1. Check the [test_methodology.md](test_methodology.md) for detailed testing
2. Review [PRD.md](PRD.md) for technical specifications
3. Open an issue for bugs or feature requests

---

*Built with ❤️ for better customer support analytics*