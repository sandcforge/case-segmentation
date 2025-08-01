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

### Running the Case Parser with Classification
```bash
# Run case segmentation AND classification on preprocessed data
python src/channel_segmenter.py

# Check outputs (now include classification data)
ls output/                       # View generated reports
cat output/channel_segmentation_summary_*.md  # View latest results (with timestamp)

# Check debug logs (includes classification debug info)
ls debug_output/                 # View LLM interaction logs
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

### Three-Stage Processing Pipeline
The system uses a comprehensive three-stage approach optimized for both performance and intelligence:

1. **Data Preprocessing** (`src/data_preprocessor.py`) - Extracts necessary fields, cleans content, filters FILE messages, groups by channel, sorts by timestamp, and creates optimized CSV output.

2. **Case Segmentation** (`src/channel_segmenter.py`) - Loads preprocessed data, performs full conversation analysis with LLM, implements token truncation, and identifies case boundaries.

3. **Case Classification** (`src/case_classifier.py`) - Automatically classifies each identified case using LLM analysis into a hierarchical taxonomy with 9 primary categories and 62+ secondary categories.

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
2. **Full Conversation Analysis**: Complete context analysis with token truncation management
3. **LLM Case Segmentation**: Unified prompt engineering with multi-strategy JSON response parsing
4. **Case Extraction**: Message slicing with confidence scoring and boundary validation
5. **LLM Case Classification**: Hierarchical taxonomy classification with confidence scoring and reasoning
6. **Multi-Format Export**: JSON, CSV, and Markdown reports with comprehensive statistics and classification data

### Critical Implementation Details
- **Full Conversation Processing**: Loads entire channel conversations for complete context analysis
- **Token Truncation**: Binary search algorithm to fit conversations within 100K token limit
- **Confidence Tracking**: All cases include LLM confidence scores (0.0-1.0) for quality assessment
- **Hierarchical Classification**: Automatic case classification with 9 primary and 62+ secondary categories
- **Dual LLM Analysis**: Separate LLM calls for segmentation and classification with independent validation

## Case Classification System

### Hierarchical Taxonomy Structure
The system automatically classifies cases into a comprehensive 9-category taxonomy:

**Primary Categories (9):**
- **Order** (11 subcategories): Status, Cancel, Update, Failure, Missing Coupon, Refund Status/Request/Full/Partial, Giveaway, Other
- **Shipment** (18 subcategories): Status, Delay, Lost, Wrong Address, Reshipping, Local Pickup, Carrier Claim, Address/Carrier/Label Updates, Item Damaged/Missing/Wrong, Tracking Status/Invalid/Not Updating/False Completion, Other
- **Payment** (13 subcategories): Status, Verification, Failure, Dispute, Chargeback, Method Update, Withdrawal Status/Delay/Method, Coupon Redemption/Failure, Pay by Credit, Other
- **Tax and Fee** (6 subcategories): Sales Fee, Sales Tax, Invoice, Tax Information, Form 1099, Other
- **User** (6 subcategories): Update Username/Email/Password, Account Delete/Recover, Other
- **Seller** (8 subcategories): Application Request/Rejection/Additional Materials/Trust Review, Foundation Plan Enroll/Update/Cancel, Live Quota, Other
- **App Functionality** (13 subcategories): Login, Logout, Settings, Live, Live Auction, Purge, Marketplace, Long-Form Auction, Bug, OBS Connection, Permission, Content Moderation, System Update, Other
- **Referral and Promotion** (4 subcategories): Referral Bonus Information/Not Received, Credit, Gift, Other
- **Other** (7 subcategories): Issue Resolved/Reopened, Feedback, Complaint, Copyright, Courtesy, Other

### Classification Quality Metrics
- **Confidence Scoring**: Each classification includes LLM confidence (0.0-1.0)
- **Reasoning Provided**: LLM explains classification decisions for transparency
- **Taxonomy Validation**: All classifications validated against predefined categories
- **Error Handling**: Invalid categories rejected with debug logging

### Classification Configuration
- **Shared LLM Configuration**: Uses same provider/model settings as case segmentation
- **Independent Processing**: Classification runs as separate LLM calls after segmentation
- **Batch Processing**: Multiple cases classified efficiently in sequence
- **Debug Logging**: All classification interactions logged to `debug_output/` directory

## Key Development Patterns

### Case Processing Pipeline
The system follows this comprehensive pattern:
1. Load CSV → group by channel_url → sort by timestamp
2. Full conversation analysis with token truncation as needed
3. LLM boundary detection with multi-strategy JSON parsing fallbacks
4. Case extraction with confidence tracking and quality assessment
5. **LLM case classification** with hierarchical taxonomy validation
6. Statistics aggregation with property-based caching (includes classification metrics)
7. Export in multiple formats (JSON, CSV, Markdown) with classification data

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
- **Efficient classification processing** with batch operations and taxonomy validation

### Token Usage and Cost Considerations
- **Dual LLM Calls**: Each case requires both segmentation and classification analysis
- **Classification Overhead**: Additional ~2000-5000 tokens per case for classification
- **Batch Efficiency**: Classification processes multiple cases in sequence to minimize overhead
- **Token Optimization**: Reuses conversation content efficiently between segmentation and classification

## Working with the System

### Troubleshooting Common Issues
- **"No API key found"**: Check `.env` file has OPENAI_API_KEY or ANTHROPIC_API_KEY
- **"Rate limit exceeded"**: Reduce batch_size in config.json or switch to budget models
- **"Context length exceeded"**: Reduce max_context_tokens in config.json for automatic truncation
- **"No cases found"**: Check sample data quality, verify message filtering, review conversation content
- **"Classification failed"**: Check debug logs in `debug_output/` for LLM classification errors
- **"Invalid category"**: LLM tried to use non-existent category; check classification debug logs
- **"No LLM available for classification"**: Classification LLM initialization failed; check API keys and config

### Debug Information System
The system automatically creates comprehensive debug dumps for all LLM interactions:
- **Location**: Debug files are saved in `debug_output/` directory
- **Naming**: `{operation_type}_{status}_{case_id}_{timestamp}.txt` format
- **Contents**: Full LLM responses, prompts, error details, response metadata
- **Segmentation Types**: `llm_call_initial_analysis_{success|error}`, `api_call_failed`, `complete_parsing_failed`, etc.
- **Classification Types**: `classification_{success|error}_{case_id}`, includes taxonomy validation details
- **Usage**: Review debug files when troubleshooting LLM response parsing issues, API failures, or classification errors

### Output Interpretation

#### Case Segmentation Quality
- **Confidence scores**: >0.8 high quality, 0.5-0.8 moderate, <0.5 review needed
- **Truncation indicators**: Look for `truncated: true` in case data for incomplete conversations
- **Token usage**: Monitor input/output tokens per channel for cost optimization
- **Processing time**: Full conversation analysis optimized for accuracy and completeness

#### Case Classification Results
- **Primary/Secondary categories**: Hierarchical classification with 9 primary and 62+ secondary options
- **Classification confidence**: Independent confidence scoring for each classification (0.0-1.0)
- **Classification reasoning**: LLM provides explanation for each classification decision
- **Missing classifications**: Cases without classification indicate LLM classification failure

#### Enhanced Export Formats
- **JSON Export**: Individual case classifications with reasoning, plus comprehensive classification statistics
- **CSV Export**: New columns `primary_category`, `secondary_category`, `classification_confidence` for every message
- **Classification Statistics**: Category distributions, confidence metrics, and classification success rates


## Run Commands

- **Run all scripts under the root folder**: Execute all Python scripts in the repository for comprehensive processing

## IMPORTANT: Sound Notification

After finishing responding to my request or running a command, run this command to notify me by sound:

```bash
afplay /System/Library/Sounds/Funk.aiff
```

## Sound Notification Strategy
- Notify by sound after completing a task or running a command
- Use system sound to provide audio feedback