# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Data Preprocessing (Required First Step)
```bash
# Preprocess raw CSV data for optimal performance
python src/data_preprocessor.py --input assets/support_msg.csv --output assets/preprocessed_support_msg.csv

# Demo mode: select 3 representative channels (many/medium/few messages)
python src/data_preprocessor.py --demo

# Full dataset preprocessing
python src/data_preprocessor.py
```

### Running the Case Parser
```bash
# Run case segmentation on preprocessed data
python src/case_parser_channel.py

# Check outputs
ls output/                       # View generated reports
cat output/cases_channel_segmentation_summary.md  # View latest results
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY
```

## High-Level Architecture

### Two-Stage Processing Pipeline
The system uses a streamlined two-stage approach optimized for performance:

1. **Data Preprocessing** (`src/data_preprocessor.py`) - Extracts necessary fields, cleans content, filters FILE messages, groups by channel, sorts by timestamp, and creates optimized CSV output.

2. **Case Segmentation** (`src/case_parser_channel.py`) - Loads preprocessed data, performs full conversation analysis with LLM, implements token truncation, and generates comprehensive reports.

### Configuration Management System
- **Secrets vs Settings**: API keys in `.env`, configuration in `config.json`
- **Provider Abstraction**: Unified interface for OpenAI and Anthropic with model tiers (default, high_quality, balanced, budget)
- **Algorithm-Specific Parameters**: Each algorithm has distinct configuration sections in `config.json`
- **Runtime Override**: Code can override config file settings for provider/model selection

### LLM Provider Abstraction
The `src/llm_provider.py` implements a factory pattern with `LLMManager` coordinating multiple providers:
- Standardized `LLMResponse` with tokens, timing, provider metadata
- Automatic fallback between providers on failure
- Provider-specific token counting (tiktoken for OpenAI, approximation for Claude)

### Data Flow Architecture
1. **CSV Loading**: Channel-grouped message loading with FILE message filtering
2. **Queue Management**: Algorithm-specific message processing (incremental vs full)
3. **LLM Analysis**: Unified prompt engineering with multi-strategy JSON response parsing
4. **Case Extraction**: Message slicing with confidence scoring and force segmentation tracking
5. **Multi-Format Export**: JSON, Markdown, and summary reports with comprehensive statistics

### Critical Implementation Details
- **Full Conversation Processing**: Loads entire channel conversations for complete context analysis
- **Token Truncation**: Binary search algorithm to fit conversations within 100K token limit
- **Confidence Tracking**: All cases include LLM confidence scores (0.0-1.0) for quality assessment
- **Comprehensive Analysis**: Single-pass analysis with built-in review and boundary validation

## Key Development Patterns

### Case Processing Pipeline
The system follows this streamlined pattern:
1. Load CSV → group by channel_url → sort by timestamp
2. Full conversation analysis with token truncation as needed
3. LLM boundary detection with multi-strategy JSON parsing fallbacks
4. Case extraction with confidence tracking and quality assessment
5. Statistics aggregation with property-based caching
6. Export in multiple formats (JSON, CSV, Markdown)

### Configuration-First Development
1. Configure API keys in `.env`
2. Adjust models/providers in `config.json`
3. Configure token limits and processing parameters
4. Run preprocessing and case parsing with optimized settings

### Error Handling Strategy
- Multi-strategy JSON parsing (direct → extraction → cleaning)
- Graceful provider fallback in `LLMManager`
- Token truncation with binary search optimization
- Comprehensive debug logging and statistics caching

## Data Processing Notes

### Input Format
CSV with required columns: `message_id`, `type`, `message`, `sender_id`, `created_time`, `channel_url`
- **FILE messages automatically filtered** during processing (don't contribute to conversation analysis)
- Messages grouped by `channel_url` and sorted chronologically by `created_time`
- Supports malformed rows with graceful error handling

### Single Algorithm Implementation
The system uses the **Channel Full Conversation** algorithm for optimal accuracy:
- Loads entire channel conversations at once for complete context
- Single LLM analysis with comprehensive boundary detection
- Binary search truncation when conversations exceed token limits
- Best balance of accuracy and performance for customer support conversations

### Token Truncation Handling
When conversation length exceeds limits:
- **Token Truncation**: Cases marked with `truncated: true`, summary includes warning text
- **Binary Search Optimization**: Efficiently finds maximum messages that fit within token limits
- **Quality Tracking**: Truncation events tracked in statistics for quality assessment

### Multi-Strategy JSON Parsing
LLM responses use fallback parsing:
1. Direct JSON parsing
2. Extract JSON from code blocks (```json)
3. Extract using regex patterns
4. Clean control characters and retry
5. Fallback to heuristic boundary detection

## Configuration Structure

### Model Selection
```json
"llm": {
  "primary_provider": "anthropic",
  "providers": {
    "anthropic": {
      "models": {
        "default": "claude-sonnet-4-20250514"
      }
    }
  }
}
```

### Algorithm Parameters
```json
"parsing": {
  "channel_algorithm": { 
    "max_context_tokens": 100000, 
    "reserve_tokens": 5000 
  }
}
```

## Performance Optimization

### Caching Strategy
All parsers implement property-based caching for expensive operations:
- `@property total_stats` - Aggregated statistics across channels
- `@property sorted_cases` - Chronologically sorted case list
- Cache invalidation on new case additions

### Memory Optimization
- **Full conversation loading** with efficient channel grouping and sorting
- **Pre-compiled regex patterns** for content cleaning (CONTROL_CHAR_PATTERN, WHITESPACE_PATTERN)  
- **Property-based caching** for expensive statistics calculations
- **FILE message filtering** reduces processing overhead by ~30% on typical datasets

## Working with the System

### Troubleshooting Common Issues
- **"No API key found"**: Check `.env` file has OPENAI_API_KEY or ANTHROPIC_API_KEY
- **"Rate limit exceeded"**: Reduce batch_size in config.json or switch to budget models
- **"Context length exceeded"**: Reduce max_context_tokens in config.json for automatic truncation
- **"No cases found"**: Check sample data quality, verify message filtering, review conversation content

### Debug Information System
The system automatically creates comprehensive debug dumps when LLM parsing fails:
- **Location**: Debug files are saved in `debug_output/` directory
- **Naming**: `{error_type}_{call_type}_{timestamp}.txt` format
- **Contents**: Full LLM responses, prompts, error details, response metadata
- **Types**: `api_call_failed`, `complete_parsing_failed`, `validation_failed`, etc.
- **Usage**: Review debug files when troubleshooting LLM response parsing issues or API failures

### Output Interpretation
- **Confidence scores**: >0.8 high quality, 0.5-0.8 moderate, <0.5 review needed
- **Truncation indicators**: Look for `truncated: true` in case data for incomplete conversations
- **Token usage**: Monitor input/output tokens per channel for cost optimization
- **Processing time**: Full conversation analysis optimized for accuracy and completeness