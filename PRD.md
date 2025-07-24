# Product Requirements Document: Chat Log Case Segmentation Parser

## Project Overview

Build a parser to segment chat logs into individual support cases using LLM-based topic change detection with human-like common sense.

## Data Source

- **File**: `assets/support_msg.csv` (163.5MB)
- **Format**: CSV with columns: `message_id`, `type`, `message`, `sender_id`, `created_time`, `channel_url`, etc.
- **Content**: Customer support conversations from multiple channels

## Core Requirements

### 1. Input Processing
- Parse large CSV file (163.5MB) containing support messages
- **Process one channel at a time** (`channel_url` grouping)
- Process chronologically ordered messages (`created_time`)
- Identify participants using `sender_id` (not `sender_type`)
- Track token consumption statistics per channel

### 2. Case Segmentation Logic
- **Primary Method**: LLM-based topic change detection using human common sense
- **Boundary Detection**: Identify when conversations shift to new issues/queries
- **Context Awareness**: Distinguish between:
  - New problems vs follow-ups
  - Different order numbers/products  
  - Conversation closure then new inquiry
  - Context shifts vs clarifications

### 3. Context Window Management Challenge
**Problem**: Long conversations exceed LLM context windows

**Proposed Solutions Discussed**:
1. **Sliding Window with Overlap** - Process in chunks with boundary overlap
2. **Hierarchical Two-Phase** - Rule-based pre-segmentation + LLM refinement  
3. **Progressive Summarization** - Summarize chunks and track active cases
4. **User's Queue Algorithm** ⭐ - Maintain sliding queue, incrementally detect boundaries
5. **Enhanced StreamingLLM** - Add attention sinks + coherence scoring

### 4. Implementation Approach

#### Version A: Basic Sliding Queue (User's Original)
```
1. Initialize empty queue
2. Push messages in batches (5-10 messages)
3. LLM analysis: "Complete case found? Clear boundary?"
4. If complete case → extract, remove from queue
5. If incomplete → push more messages
6. If queue too large → force boundary
```

#### Version B: Enhanced StreamingLLM Queue
```
1. Attention sink pattern (keep first 4 critical messages)
2. Utterance-pair coherence pre-filtering
3. Topic-aware contrastive learning
4. Hybrid fast heuristics + LLM analysis
```

## Output Requirements

### 1. JSON Format
- Array of case objects
- Each case contains:
  - `case_id`
  - `messages[]`
  - `start_time`/`end_time` 
  - `participants[]`
  - `summary`
  - `channel_url`

### 2. Markdown Format
- Human-readable case summaries
- Key conversation details
- **Per-channel token consumption statistics**
- Performance metrics summary

## Technical Specifications

### LLM Provider Abstraction
- **Multi-provider support**: OpenAI and Anthropic Claude
- **Unified interface**: Abstract LLM calls for easy provider switching
- **Automatic fallback**: Switch providers if primary fails
- **Token counting**: Provider-specific token calculation

### Configuration Management
- **API Keys**: Stored in `.env` file (secrets)
- **Model Selection**: Stored in `config.json` (configuration)
- **Environment separation**: Clear distinction between secrets and settings
- **Easy switching**: Change providers/models via config file or code parameters

### Performance Metrics
- LLM API calls count per provider
- Processing time per channel
- Memory usage tracking
- **Token consumption per channel** (input + output tokens)
- Provider-specific performance comparison
- Accuracy comparison between algorithms

### Model Options
**OpenAI Models:**
- `gpt-4o-mini` (default) - Best cost/performance
- `gpt-4o` (high_quality) - Maximum accuracy
- `gpt-4-turbo` (balanced) - Good performance
- `gpt-3.5-turbo` (budget) - Lowest cost

**Anthropic Models:**
- `claude-3-5-sonnet-20241022` (default) - High quality reasoning
- `claude-3-haiku-20240307` (budget) - Fast and economical

### Error Handling
- Large file processing (163.5MB)
- Malformed CSV rows
- API rate limits
- Context window overflow

## Success Criteria

1. **Accuracy**: Meaningful case boundaries that match human judgment
2. **Performance**: Process 163.5MB file efficiently 
3. **Scalability**: Handle unlimited conversation length
4. **Comparison**: Clear performance metrics between both algorithms
5. **Usability**: Clean JSON + readable Markdown outputs

## Research Insights Applied

### StreamingLLM (MIT 2024)
- Attention sinks for infinite context
- 22x speedup over sliding windows
- Key insight: Keep first 4 tokens + sliding window

### Topic Segmentation Research (2024)
- BERT-based utterance-pair coherence scoring
- 15.5% error reduction over traditional methods
- Multi-task neural networks for boundary detection

### Hybrid Approaches
- Combine lexical cohesion + linguistic features
- Fast pre-filtering before expensive LLM calls
- Real-time streaming conversation processing

## Implementation Status

- [x] Requirements analysis
- [x] Algorithm design  
- [x] **LLM provider abstraction layer**
- [x] **Configuration management system**
- [x] **Basic sliding queue implementation** (Version A)
- [x] **Enhanced StreamingLLM implementation** (Version B)
- [x] **Channel-by-channel processing**
- [x] **Token consumption tracking**
- [x] **JSON/Markdown output generation**
- [ ] Performance comparison framework
- [ ] Algorithm testing and validation
- [ ] Documentation completion

## File Structure
```
case_segmentation/
├── PRD.md (this file)
├── assets/
│   └── support_msg.csv (163.5MB chat data)
├── Core Implementation/
│   ├── case_parser_basic.py (Version A - User's sliding queue)
│   ├── case_parser_enhanced.py (Version B - StreamingLLM + attention sinks)
│   ├── llm_provider.py (Multi-provider abstraction layer)
│   └── config_manager.py (Configuration management)
├── Configuration/
│   ├── .env (API keys - secrets)
│   ├── .env.example (Template)
│   ├── config.json (Model selection & settings)
│   └── requirements.txt (Dependencies)
├── Output/
│   ├── cases_basic.json
│   ├── cases_basic.md
│   ├── cases_enhanced.json
│   ├── cases_enhanced.md
│   └── algorithm_comparison.md
└── Documentation/
    └── PRD.md
```

## Usage Instructions

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Adjust models in config.json if needed
```

### 2. Run Basic Algorithm (User's Sliding Queue)
```python
python case_parser_basic.py
```

### 3. Run Enhanced Algorithm (StreamingLLM + Attention Sinks)
```python
python case_parser_enhanced.py
```

### 4. Switch Providers/Models
**Via config.json:**
```json
{
  "llm": {
    "primary_provider": "anthropic",  // Use Claude
    "providers": {
      "openai": {
        "models": {
          "default": "gpt-4o"  // Use higher quality model
        }
      }
    }
  }
}
```

**Via code:**
```python
# Use Claude with high quality model
parser = BasicCaseParser(
    llm_provider="anthropic", 
    llm_model_type="default"
)

# Use OpenAI GPT-4o
parser = BasicCaseParser(
    llm_provider="openai", 
    llm_model_type="high_quality"
)
```

## Next Steps

1. ✅ **Core implementation complete**
2. **Test with sample data** to validate algorithms
3. **Performance comparison** between basic vs enhanced
4. **Cost analysis** across different models/providers
5. **Algorithm optimization** based on results
6. **Production deployment** with best-performing configuration