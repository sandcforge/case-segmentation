#!/usr/bin/env python3
"""
Case Parser - Channel Full Conversation Version

This implements a whole-conversation approach:
- Load entire channel conversation at once
- Single LLM analysis for complete context
- Extract all cases from single analysis
- Truncate if conversation exceeds token limits
- Mark truncated cases appropriately
"""

import csv
import json
import re
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from llm_provider import create_llm_manager
from config_manager import get_config, get_llm_config, get_parsing_config, get_output_config

# Pre-compiled regex patterns for performance
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
WHITESPACE_PATTERN = re.compile(r'\s+')
CODE_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL | re.IGNORECASE)
JSON_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)




@dataclass
class Message:
    message_id: str
    message_type: str
    content: str
    sender_id: str
    timestamp: datetime
    channel_url: str
    sender_type: str = ""  # customer_service or user
    review: str = ""  # review value from original CSV
    file_url: str = ""  # file URL if message type is FILE
    real_sender_id: str = ""  # real sender ID from original CSV
    round_columns: dict = field(default_factory=dict)  # Dictionary to store all round* columns


@dataclass
class Case:
    case_id: str
    messages: List[Message]
    start_time: datetime
    end_time: datetime
    participants: List[str]
    summary: str
    channel_url: str
    confidence: float = 0.0  # Segmentation confidence from LLM
    duration_minutes: float = 0.0  # Case duration in minutes
    forced_ending: bool = False  # True if case was force-extracted due to queue overflow
    forced_starting: bool = False  # True if case starts after a force extraction


@dataclass
class CaseReviewResult:
    """Review result for case segmentation quality assessment"""
    case_index: int  # Index of the case being reviewed
    needs_split: bool  # Whether the case should be split further
    confidence: float  # Reviewer confidence in the decision
    reasoning: str  # Explanation for the decision
    suggested_split_points: List[int] = field(default_factory=list)  # Message indices for splitting
    review_iteration: int = 0  # Which review iteration this is from


@dataclass
class ChannelStats:
    channel_url: str
    total_messages: int
    cases_found: int
    input_tokens: int
    output_tokens: int
    llm_calls: int
    processing_time: float
    review_iterations: int = 0  # Number of review iterations performed
    cases_reviewed: int = 0  # Number of cases that underwent review
    cases_split: int = 0  # Number of cases that were split after review


class ChannelSegmenter:
    def __init__(self, llm_provider: Optional[str] = None, llm_model_type: Optional[str] = None):
        # Load configuration
        self.config = get_config()
        
        # Get parsing configuration
        parsing_config = get_parsing_config("channel")
        self.max_context_tokens = parsing_config.max_context_tokens
        self.reserve_tokens = parsing_config.reserve_tokens
        self.review_enabled = getattr(parsing_config, 'review_enabled', False)
        self.max_review_iterations = getattr(parsing_config, 'max_review_iterations', 3)
        
        # Get model_type from config if not specified
        if llm_model_type is None:
            llm_model_type = self.config.config["llm"].get("model_type", "default")
        
        # Get LLM configuration
        llm_config = get_llm_config(llm_provider, llm_model_type)
        
        # Store provider for prompt customization
        self.llm_provider = llm_config.provider
        
        # Initialize LLM manager
        try:
            if llm_config.provider == "openai":
                self.llm_manager = create_llm_manager(
                    provider="openai", 
                    openai_model=llm_config.model,
                    output_max_token_size=llm_config.output_max_token_size
                )
            elif llm_config.provider == "anthropic":
                self.llm_manager = create_llm_manager(
                    provider="anthropic", 
                    anthropic_model=llm_config.model,
                    output_max_token_size=llm_config.output_max_token_size
                )
            elif llm_config.provider == "gemini":
                self.llm_manager = create_llm_manager(
                    provider="gemini", 
                    gemini_model=llm_config.model,
                    output_max_token_size=llm_config.output_max_token_size
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
            
            print(f"Initialized with provider: {self.llm_manager.get_provider_info()}")
            
        except Exception as e:
            print(f"Warning: LLM initialization failed: {e}")
            self.llm_manager = None
        
        # Processing state
        self.completed_cases: List[Case] = []
        self.case_counter = 1
        
        # Statistics per channel
        self.channel_stats: Dict[str, ChannelStats] = {}
        self.current_channel_stats: Optional[ChannelStats] = None
        
        # Cache for optimizations
        self._cached_total_stats = None
        self._sorted_cases_cache = None
        self._cache_invalidated = True
        
        
    def load_dataframe(self, df: pd.DataFrame) -> Dict[str, List[Message]]:
        """Load and parse DataFrame, group by channel"""
        channels = defaultdict(list)
        total_count = len(df)
        
        print(f"📊 Loading DataFrame with {total_count} rows...")
        
        for i, row in df.iterrows():
            if i % 100 == 0 and i > 0:
                print(f"  📊 Processed {i} rows, found {len(channels)} channels...")
            
            try:
                # Parse timestamp
                if pd.isna(row['Created Time']):
                    continue
                
                if isinstance(row['Created Time'], str):
                    timestamp = datetime.fromisoformat(row['Created Time'].replace('Z', '+00:00'))
                else:
                    timestamp = pd.to_datetime(row['Created Time']).to_pydatetime()
                
                # Extract all columns that start with "round"
                round_columns = {col: str(row.get(col, '')) for col in df.columns if col.startswith('round')}
                
                message = Message(
                    message_id=str(row['Message ID']),
                    message_type=str(row['Type']),
                    content=str(row['Message']),
                    sender_id=str(row['Sender ID']),
                    timestamp=timestamp,
                    channel_url=str(row['Channel URL']),
                    sender_type=str(row['Sender Type']),
                    review=str(row.get('review', '')),  # This might not exist in raw CSV
                    file_url=str(row.get('File URL', '')),
                    real_sender_id=str(row.get('Real Sender ID', '')),
                    round_columns=round_columns
                )
                channels[message.channel_url].append(message)
                
            except Exception as e:
                print(f"  ❌ Error parsing row {i}: {e}")
                continue
        
        processed_count = sum(len(msgs) for msgs in channels.values())
        print(f"✅ Loaded {processed_count} messages across {len(channels)} channels")
        
        # Show channel distribution
        channel_sizes = [len(msgs) for msgs in channels.values()]
        if channel_sizes:
            print(f"  📈 Channel sizes: min={min(channel_sizes)}, max={max(channel_sizes)}, avg={sum(channel_sizes)/len(channel_sizes):.1f}")
        return dict(channels)

    def load_csv(self, filepath: str) -> Dict[str, List[Message]]:
        """Load and parse preprocessed CSV file, group by channel"""
        channels = defaultdict(list)
        total_count = 0
        
        print("📁 Loading preprocessed CSV file...")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                total_count += 1
                if i % 100 == 0 and i > 0:  # More frequent updates for smaller files
                    print(f"  📊 Processed {i} rows, found {len(channels)} channels...")
                
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(row['Created Time'].replace('Z', '+00:00'))
                    
                    # Use preprocessed content directly (already cleaned)
                    content = row['Message']
                    
                    # Extract all columns that start with "round"
                    round_columns = {col: row.get(col, '') for col in row.keys() if col.startswith('round')}
                    
                    message = Message(
                        message_id=row['Message ID'],
                        message_type=row['Type'],
                        content=content,
                        sender_id=row['Sender ID'],
                        timestamp=timestamp,
                        channel_url=row['Channel URL'],
                        sender_type=row['Sender Type'],
                        review=row.get('review', ''),  # This might not exist in raw CSV
                        file_url=row.get('File URL', ''),
                        real_sender_id=row.get('Real Sender ID', ''),
                        round_columns=round_columns
                    )
                    channels[message.channel_url].append(message)
                    
                except Exception as e:
                    print(f"  ❌ Error parsing row {i}: {e}")
                    continue
        
        # Messages are already sorted by channel and timestamp in preprocessed data
        processed_count = sum(len(msgs) for msgs in channels.values())
        print(f"✅ Loaded {processed_count} preprocessed messages across {len(channels)} channels")
        
        # Show channel distribution
        channel_sizes = [len(msgs) for msgs in channels.values()]
        if channel_sizes:
            print(f"  📈 Channel sizes: min={min(channel_sizes)}, max={max(channel_sizes)}, avg={sum(channel_sizes)/len(channel_sizes):.1f}")
        return dict(channels)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.llm_manager:
            return self.llm_manager.count_tokens(text)
        else:
            # Fallback: rough approximation
            return len(text) // 4
    
    def extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response that may contain extra text"""
        # Clean the response first
        cleaned = response_text.strip()
        
        # First, try to extract content from <output></output> tags
        output_pattern = re.compile(r'<output>\s*(.*?)\s*</output>', re.DOTALL | re.IGNORECASE)
        output_match = output_pattern.search(cleaned)
        
        if output_match:
            # Extract content from output tags and process it
            output_content = output_match.group(1).strip()
            print("  🎯 Found content in <output></output> tags")
            
            # Try to parse the output content as JSON
            try:
                # Check if it's already clean JSON
                json.loads(output_content)
                return output_content
            except json.JSONDecodeError:
                # If not, continue with extraction methods below on the output content
                cleaned = output_content
        
        # Try to find JSON block between { and } (including nested braces)
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found complete JSON object
                    return cleaned[start_idx:i+1]
        
        # Try to find content between ```json blocks using pre-compiled pattern
        code_matches = CODE_BLOCK_PATTERN.findall(cleaned)
        
        if code_matches:
            return code_matches[0]
        
        # Try simpler JSON pattern using pre-compiled pattern
        matches = JSON_PATTERN.findall(cleaned)
        
        if matches:
            # Return the longest JSON-like match
            return max(matches, key=len)
        
        # If no JSON found, return original
        return cleaned
    
    def clean_message_content(self, content: str) -> str:
        """Clean message content to remove invalid control characters"""
        if not content:
            return ""
        
        try:
            # Remove control characters except newlines and tabs using pre-compiled pattern
            cleaned = CONTROL_CHAR_PATTERN.sub('', content)
            
            # Replace problematic quotes and characters
            cleaned = cleaned.replace('"', '"').replace('"', '"')
            cleaned = cleaned.replace(''', "'").replace(''', "'")
            
            # Remove null bytes and other problematic characters
            cleaned = cleaned.replace('\x00', '').replace('\ufffd', '')
            
            # Replace multiple whitespace with single space using pre-compiled pattern
            cleaned = WHITESPACE_PATTERN.sub(' ', cleaned)
            
            # Strip leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Ensure we have valid content
            if not cleaned:
                return "[empty message]"
            
            # Truncate very long messages
            if len(cleaned) > 500:
                cleaned = cleaned[:497] + "..."
            
            return cleaned
            
        except Exception as e:
            print(f"Error cleaning message content: {e}")
            return "[invalid message content]"
    
    def format_conversation_for_llm(self, messages: List[Message]) -> str:
        """Format entire conversation for LLM analysis"""
        formatted = "COMPLETE CONVERSATION:\n\n"
        
        for i, msg in enumerate(messages, 1):
            # Content is already cleaned from preprocessing
            formatted += f"<message id=\"{i}\">\n"
            formatted += f"<sender>{msg.sender_id}</sender>\n"
            formatted += f"<sender_type>{msg.sender_type}</sender_type>\n"
            formatted += f"<time>{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</time>\n"
            formatted += f"<content>{msg.content}</content>\n"
            formatted += f"</message>\n\n"
            
        return formatted
    
    
   
    def analyze_full_conversation(self, messages: List[Message]) -> Dict[str, Any]:
        """Use LLM to analyze conversation for case boundaries with built-in review"""
        if not self.llm_manager:
            print("  ⚠️ No LLM available - creating single case with note")
            return {
                "complete_cases": [{
                    "start_message": 1,
                    "end_message": len(messages),
                    "summary": "No LLM analysis available - processed as single case",
                    "confidence": 0.3
                }],
                "analysis": "No LLM provider available",
                "total_messages_analyzed": len(messages)
            }
        
        print(f"  🤖 Analyzing complete conversation with {len(messages)} messages...")
        
        # Create the analysis prompt
        prompt = f"""
You are an expert e-commerce customer service conversation analyst. Your task is to segment seller-support conversations into meaningful cases based on order context and business logic.

## Conversation Context
This is a conversation between ONE seller (sender_type="user") and customer service (sender_type="support") discussing various issues that may involve different buyers' orders or general business topics.

## Input Data Structure
Each message contains:
- **message_id**: Unique message identifier for ordering
- **sender**: Person's ID who sent the message
- **sender_type**: "support" (platform customer service) or "user" (seller)
- **time**: Message timestamp for temporal analysis
- **content**: Actual message content

## Core Segmentation Principle: Multi-Dimensional Business Logic

### **Primary Segmentation Drivers (Always Create Separate Cases)**:
- **Different Order Numbers**: Each order number represents a separate business transaction
- **Different Buyer**: Different buyer(username, user id) require separate case tracking
- **Different Business Process Types**: Claims vs refunds vs technical issues vs policy questions
- **Different Urgency Levels**: Emergency technical issues vs routine inquiries

### **Marketplace-Specific Business Logic**:

#### **Seller Support Categories**:
- **Order Management**: Address changes, cancellations, local pickup conversions
- **Financial Issues**: Payouts, fees, reimbursements, chargebacks
- **Shipping/Claims**: Lost packages, damage claims, delivery protection
- **Platform Issues**: App bugs, data sync problems, account access
- **Policy/Compliance**: Disputes, seller conduct, marketplace rules

#### **Resolution Process Continuity**:
- **Single Process**: Problem report → Investigation → Solution → Confirmation
- **Multi-Stage Claims**: Initial claim → Evidence gathering → Review → Payout → Follow-up
- **Escalation Chains**: Seller support → Supervisor → Engineering → Management

### **Enhanced Decision Framework**:

#### **Strong Segmentation Signals**:
- **Order Number Change**: "About order #123" → "Regarding order #456"
- **Process Type Change**: Shipping claim → App technical issue
- **buyer Change**: Different buyer or its username mentioned
- **Business Day Boundaries**: Cross-day issues unless active resolution
- **Urgency Level Change**: Emergency technical issue + routine inquiry

#### **Continuity Factors** (Keep Together):
- **Same Order Lifecycle**: Order → Shipping → Delivery → Issue → Resolution
- **Multi-Step Claims**: Claim submission → Evidence → Review → Payout
- **Technical Issue Resolution**: Bug report → Troubleshooting → Fix → Verification
- **Escalation Processes**: L1 support → L2 supervisor → Engineering

#### **Time-Based Segmentation Rules**:
- **Active Session (0-4 hours)**: Strong continuity for related issues
- **Business Day Span (4-24 hours)**: Moderate continuity, check order relationship
- **Multi-Day Issues (24+ hours)**: Evaluate business process continuity
- **Long-term Claims (weeks/months)**: Maintain case continuity for ongoing processes

### **Structured Data Analysis Approach**:

#### **Temporal Analysis Enhancement**:
- **Time Gap Calculation**: Use precise timestamps for interval analysis
- **Business Hours Context**: Consider timezone and business day patterns
- **Response Time Patterns**: Quick support responses vs delayed seller replies

#### **Content-Context Integration**:
- **Order Number Extraction**: Identify all order references in content
- **Issue Type Classification**: Map content to business process categories
- **Urgency Indicators**: Detect emergency language, escalation requests

<thinking>
Step 1: **Data Structure Validation** - Verify message ordering by time consistency
Step 2: **Seller and Agent Identification** - Map sender IDs to roles and track multiple sellers
Step 3: **Business Context Extraction** - Extract order numbers, buyer, issue types, and process stages
Step 4: **Temporal Pattern Analysis** - Calculate time gaps and identify session boundaries
Step 5: **Order-Process Mapping** - Group by order numbers and process categories
Step 6: **Cross-Reference Validation** - Ensure sender continuity matches content context
Step 7: **Business Logic Application** - Apply marketplace-specific segmentation rules
Step 8: **Operational Optimization** - Create segments supporting efficient case management
</thinking>

### **CONVERSATION TO ANALYZE:**
{self.format_conversation_for_llm(messages)}

### Output segmentation results in JSON format:

{{
    "complete_cases": [
        {{
            "start_message": 1,
            "end_message": 8,
            "summary": "Brief description of the issue, actions taken, and resolution status",
            "confidence": 0.9
        }}
    ],
    "total_messages_analyzed": [total_number_of_messages]
}}

        """
        
        # Execute LLM analysis and parse response
        response = self._execute_llm_call(prompt, "initial analysis", messages)
        result = None
        
        if response:
            parsed_result = self.parse_llm_response_json(response)
            if parsed_result and self.validate_segmentation_response(parsed_result):
                result = parsed_result
        
        # Create fallback if analysis failed
        if result is None:
            print("  ⚠️ Analysis failed, creating fallback single case")
            result = {
                "complete_cases": [{
                    "start_message": 1,
                    "end_message": len(messages),
                    "summary": "LLM analysis failed - processed as single case",
                    "confidence": 0.1
                }],
                "analysis": "LLM analysis failed",
                "total_messages_analyzed": len(messages)
            }
        
        # Log analysis results
        cases = result.get("complete_cases", [])
        if cases:
            avg_confidence = sum(case.get("confidence", 0.0) for case in cases) / len(cases)
            print(f"  🎯 Analysis complete: {len(cases)} cases, avg confidence: {avg_confidence:.3f}")
        else:
            print("  ⚠️ No cases identified in analysis")
        
        return result
    
    def _execute_llm_call(self, prompt: str, call_type: str, messages: List[Message] = None):
        """Execute LLM API call with logging and statistics tracking"""
        if messages is None:
            messages = []
            
        try:
            response = self.llm_manager.analyze_case_boundaries(prompt)
            
            # Log ALL LLM interactions for debugging
            self._log_llm_interaction(call_type, prompt, response, messages, success=True)
            
            # Update statistics
            self.update_channel_stats(response)
            
            print(f"    📊 {call_type}: {response.input_tokens} input + {response.output_tokens} output tokens")
            
            return response
        except Exception as e:
            self.handle_llm_api_error(e, call_type, prompt, messages)
            return None
    
    def review_case_segmentation(self, original_messages: List[Message], cases: List[Dict[str, Any]], 
                                iteration: int = 0) -> List[CaseReviewResult]:
        """Review case segmentation quality and suggest improvements"""
        if not self.llm_manager:
            print(f"    ⚠️ No LLM available for review - skipping")
            return []
        
        print(f"    🔍 Reviewing {len(cases)} cases (iteration {iteration})...")
        
        review_results = []
        
        for case_idx, case_info in enumerate(cases):
            start_idx = case_info["start_message"] - 1
            end_idx = case_info["end_message"]
            
            # Extract case messages
            if 0 <= start_idx < end_idx <= len(original_messages):
                case_messages = original_messages[start_idx:end_idx]
                
                # Create review prompt for this specific case
                review_prompt = f"""
You are an expert case segmentation reviewer. Your task is to analyze a single case and determine if it should be split into multiple separate cases.

## Case Analysis Guidelines

### Strong Indicators for Splitting:
- **Multiple distinct topics**: Different issues/problems being discussed
- **Clear topic transitions**: "Also", "another question", "by the way", "separately"
- **Resolution boundaries**: One issue resolved, then new issue starts
- **Temporal disconnections**: Long time gaps with context breaks
- **Different business processes**: Order issue → technical problem → billing inquiry

### Keep Together (Do NOT split):
- **Clarification requests**: Follow-up questions on same topic
- **Multi-step processes**: Troubleshooting, information gathering
- **Status updates**: Progress reports on ongoing issue
- **Context continuity**: Related conversation flow

## Case to Review (Messages {start_idx + 1} to {end_idx}):

{self.format_conversation_for_llm(case_messages)}

## Analysis Task

Carefully analyze this case and determine:
1. Does this case contain multiple distinct topics that should be separate cases?
2. Are there clear boundary points where a split would make sense?
3. Would splitting improve case management and tracking?

## Required Response Format

Return ONLY valid JSON in this exact format:

{{
    "needs_split": false,
    "confidence": 0.9,
    "reasoning": "Brief explanation of why this case should/shouldn't be split",
    "suggested_split_points": []
}}

If needs_split is true, provide message numbers (1-indexed) where splits should occur in suggested_split_points array.

Example: If case has messages 1-10 and should split after message 4 and 7:
{{
    "needs_split": true,
    "confidence": 0.8,
    "reasoning": "Contains three distinct topics: delivery issue (1-4), billing question (5-7), technical problem (8-10)",
    "suggested_split_points": [4, 7]
}}
"""
                
                try:
                    # Execute review analysis
                    llm_response = self._execute_llm_call(review_prompt, f"case_review_{case_idx}_iter_{iteration}", case_messages)
                    response = None
                    if llm_response:
                        response = self.parse_llm_response_json(llm_response)
                        # Validate the result if parsing succeeded
                        if response and not self.validate_review_response(response):
                            response = None
                    
                    if response and response.get("needs_split") is not None:
                        # Convert suggested split points to case-relative indices
                        split_points = response.get("suggested_split_points", [])
                        # Convert to message indices relative to the original conversation (LLM returns 1-based indices)
                        absolute_split_points = [start_idx + sp - 1 for sp in split_points if 0 < sp <= len(case_messages)]
                        
                        review_result = CaseReviewResult(
                            case_index=case_idx,
                            needs_split=response.get("needs_split", False),
                            confidence=response.get("confidence", 0.5),
                            reasoning=response.get("reasoning", "No reasoning provided"),
                            suggested_split_points=absolute_split_points,
                            review_iteration=iteration
                        )
                        review_results.append(review_result)
                        
                        if review_result.needs_split:
                            print(f"      🔄 Case {case_idx} flagged for splitting (conf: {review_result.confidence:.3f})")
                        else:
                            print(f"      ✅ Case {case_idx} approved (conf: {review_result.confidence:.3f})")
                    else:
                        print(f"      ⚠️ Case {case_idx} review failed - invalid response")
                        
                except Exception as e:
                    print(f"      ❌ Case {case_idx} review error: {e}")
                    continue
        
        return review_results
    
    def apply_review_splits(self, original_messages: List[Message], original_cases: List[Dict[str, Any]], 
                           review_results: List[CaseReviewResult]) -> List[Dict[str, Any]]:
        """Apply suggested splits from review results to create refined cases"""
        refined_cases = []
        
        for case_idx, case_info in enumerate(original_cases):
            # Find review result for this case
            review_result = next((r for r in review_results if r.case_index == case_idx), None)
            
            if review_result and review_result.needs_split and review_result.suggested_split_points:
                # Split this case
                print(f"      ✂️ Splitting case {case_idx} at {len(review_result.suggested_split_points)} points")
                
                case_start = case_info["start_message"] - 1  # Convert to 0-based
                case_end = case_info["end_message"]
                
                # Create split boundaries including case start/end
                split_points = [case_start] + review_result.suggested_split_points + [case_end]
                split_points = sorted(set(split_points))  # Remove duplicates and sort
                
                # Create new cases from split points
                for i in range(len(split_points) - 1):
                    split_start = split_points[i]
                    split_end = split_points[i + 1]
                    
                    if split_start < split_end:
                        # Extract messages for this split
                        split_messages = original_messages[split_start:split_end]
                        
                        if split_messages:
                            refined_case = {
                                "start_message": split_start + 1,  # Convert back to 1-based
                                "end_message": split_end,
                                "summary": f"Reviewed case split {i+1} from original case {case_idx}: {case_info.get('summary', 'No summary')[:60]}...",
                                "confidence": min(case_info.get("confidence", 0.5), review_result.confidence)  # Take minimum confidence
                            }
                            refined_cases.append(refined_case)
            else:
                # Keep original case unchanged
                refined_cases.append(case_info)
        
        return refined_cases
    
    def validate_segmentation_response(self, response_dict: Dict[str, Any]) -> bool:
        """Validate that segmentation LLM response has the required structure"""
        if not isinstance(response_dict, dict):
            return False
        
        # Check if complete_cases key exists and is a list
        if "complete_cases" not in response_dict:
            return False
        
        complete_cases = response_dict["complete_cases"]
        if not isinstance(complete_cases, list):
            return False
        
        # Validate each case has required fields
        for case in complete_cases:
            if not isinstance(case, dict):
                return False
            required_fields = ["start_message", "end_message", "summary", "confidence"]
            if not all(field in case for field in required_fields):
                return False
        
        return True
    
    def validate_review_response(self, response_dict: Dict[str, Any]) -> bool:
        """Validate that review LLM response has the required structure"""
        if not isinstance(response_dict, dict):
            return False
        
        # Check required fields for review response
        required_fields = ["needs_split", "confidence", "reasoning"]
        if not all(field in response_dict for field in required_fields):
            return False
        
        # Validate field types
        if not isinstance(response_dict["needs_split"], bool):
            return False
        
        if not isinstance(response_dict["confidence"], (int, float)):
            return False
        
        if not isinstance(response_dict["reasoning"], str):
            return False
        
        # suggested_split_points should be a list if present
        if "suggested_split_points" in response_dict:
            if not isinstance(response_dict["suggested_split_points"], list):
                return False
        
        return True
        
    def update_channel_stats(self, response):
        """Update current channel statistics with LLM response data"""
        if self.current_channel_stats and hasattr(response, 'input_tokens'):
            self.current_channel_stats.input_tokens += response.input_tokens
            self.current_channel_stats.output_tokens += response.output_tokens
            self.current_channel_stats.llm_calls += 1
            self.current_channel_stats.processing_time += response.processing_time
    
    def handle_llm_api_error(self, exception: Exception, call_type: str, prompt: str, messages: List[Message]) -> str:
        """Handle LLM API call errors with simplified logging and debug dumps"""
        print(f"    ❌ {call_type} failed: {exception}")
        
        # Simplified error details
        error_details = f"{str(exception)}\nType: {type(exception).__name__}"
        
        # Log the interaction and get debug filename
        debug_filename = self._log_llm_interaction(call_type, prompt, exception, messages, success=False, error_details=error_details)
        
        # Check for enhanced error response
        error_response = getattr(exception, 'error_response', None)
        
        if error_response:
            # Enhanced error with detailed response
            print(f"    📋 Enhanced error: finish_reason={error_response.finish_reason}")
            
            response_content = f"[ERROR_RESPONSE]\nFinish Reason: {error_response.finish_reason}\n"
            if error_response.error_details:
                response_content += f"Details: {json.dumps(error_response.error_details, indent=2, default=str)}\n"
            
            response_context = {
                "model": error_response.model,
                "provider": error_response.provider,
                "input_tokens": error_response.input_tokens,
                "output_tokens": error_response.output_tokens,
                "processing_time": f"{error_response.processing_time:.2f}s",
                "finish_reason": error_response.finish_reason,
                "error_response_available": True
            }
            
            enhanced_error_details = f"{error_details}\nEnhanced error with finish_reason: {error_response.finish_reason}"
            
            self._create_debug_dump("api_call_failed_with_response", call_type, messages, prompt, 
                                  response_content, enhanced_error_details, response_context)
        else:
            # Standard error without enhanced response
            self._create_debug_dump("api_call_failed", call_type, messages, prompt, 
                                  "", f"{error_details}\nNo enhanced error response available", 
                                  {"exception_type": type(exception).__name__})
        
        print(f"    🔍 Debug logged: {debug_filename}")
        return debug_filename
    
    def parse_llm_response_json(self, response):
        """
        Parse LLM response using 2-strategy approach.
        
        Args:
            response: LLM response object containing content
            
        Returns:
            Parsed JSON dict, or None if both strategies fail
        """
        response_content = response.content
        
        try:
            # Strategy 1: Direct JSON parsing
            result = json.loads(response_content)
            return result
        except json.JSONDecodeError:
            try:
                # Strategy 2: Extract JSON from mixed text response
                extracted_json = self.extract_json_from_response(response_content)
                result = json.loads(extracted_json)
                return result
            except (json.JSONDecodeError, Exception) as e:
                # Both strategies failed
                print(f"    ❌ JSON parsing failed: {type(e).__name__}: {str(e)}")
                return None
    
    
    def _log_llm_interaction(self, call_type: str, prompt: str, response, messages: List[Message] = None, 
                            success: bool = True, error_details: str = "") -> str:
        """Log every LLM interaction (both successful and failed) to debug_output/"""
        if messages is None:
            messages = []
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        status = "success" if success else "error"
        debug_filename = f"debug_output/llm_call_{call_type.replace(' ', '_')}_{status}_{timestamp}.txt"
        
        try:
            os.makedirs("debug_output", exist_ok=True)
            
            with open(debug_filename, 'w', encoding='utf-8') as debug_file:
                debug_file.write(f"=== LLM INTERACTION LOG ===\n")
                debug_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                debug_file.write(f"Call Type: {call_type}\n")
                debug_file.write(f"Status: {status.upper()}\n")
                debug_file.write(f"Message Count: {len(messages)}\n")
                debug_file.write(f"Channel URL: {messages[0].channel_url if messages else 'Unknown'}\n")
                
                if hasattr(response, 'model'):
                    debug_file.write(f"Model: {response.model}\n")
                    debug_file.write(f"Provider: {response.provider}\n")
                    debug_file.write(f"Input Tokens: {response.input_tokens}\n")
                    debug_file.write(f"Output Tokens: {response.output_tokens}\n")
                    debug_file.write(f"Processing Time: {response.processing_time:.2f}s\n")
                
                if error_details:
                    debug_file.write(f"\n=== ERROR DETAILS ===\n")
                    debug_file.write(f"{error_details}\n")
                
                debug_file.write(f"\n=== FULL PROMPT ===\n")
                debug_file.write(prompt)
                
                if hasattr(response, 'content'):
                    debug_file.write(f"\n\n=== FULL RESPONSE ===\n")
                    debug_file.write(f"Response length: {len(response.content)} characters\n\n")
                    debug_file.write(response.content)
                
                if messages:
                    debug_file.write(f"\n\n=== MESSAGE SAMPLE (First 3) ===\n")
                    for i, msg in enumerate(messages[:3]):
                        debug_file.write(f"Message {i+1}: [{msg.sender_id}] {msg.content[:100]}...\n")
            
            print(f"    🔍 LLM interaction logged: {debug_filename}")
            return debug_filename
            
        except Exception as debug_err:
            print(f"    ⚠️ Failed to log LLM interaction: {debug_err}")
            return ""

    def _create_debug_dump(self, error_type: str, call_type: str, messages: List[Message], 
                          prompt: str = "", response_content: str = "", error_details: str = "", 
                          additional_context: Dict[str, Any] = None) -> str:
        """Create comprehensive debug dump file for troubleshooting"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_filename = f"debug_output/{error_type}_{call_type.replace(' ', '_')}_{timestamp}.txt"
        
        try:
            os.makedirs("debug_output", exist_ok=True)
            
            with open(debug_filename, 'w', encoding='utf-8') as debug_file:
                debug_file.write(f"=== {error_type.upper()} DEBUG DUMP ===\n")
                debug_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                debug_file.write(f"Call Type: {call_type}\n")
                debug_file.write(f"Message Count: {len(messages)}\n")
                debug_file.write(f"Channel URL: {messages[0].channel_url if messages else 'Unknown'}\n")
                
                if additional_context:
                    debug_file.write("\n=== ADDITIONAL CONTEXT ===\n")
                    for key, value in additional_context.items():
                        debug_file.write(f"{key}: {value}\n")
                
                debug_file.write(f"\n=== ERROR DETAILS ===\n")
                debug_file.write(f"{error_details}\n")
                
                if response_content:
                    debug_file.write(f"\n=== LLM RESPONSE CONTENT ===\n")
                    debug_file.write(f"Response length: {len(response_content)} characters\n")
                    debug_file.write(f"Response preview: {response_content[:500]}...\n\n")
                    debug_file.write(f"=== FULL RESPONSE ===\n")
                    debug_file.write(response_content)
                
                # Add enhanced response details if available
                if additional_context and additional_context.get("finish_reason"):
                    debug_file.write(f"\n\n=== RESPONSE ANALYSIS ===\n")
                    debug_file.write(f"Finish Reason: {additional_context.get('finish_reason', 'Unknown')}\n")
                    
                    if additional_context.get("error_response_available"):
                        debug_file.write("Enhanced error response captured: YES\n")
                        debug_file.write("This error includes detailed response metadata from the LLM provider\n")
                    else:
                        debug_file.write("Enhanced error response captured: NO\n")
                
                if prompt:
                    debug_file.write(f"\n\n=== PROMPT SENT ===\n")
                    debug_file.write(prompt)
                
                debug_file.write(f"\n\n=== MESSAGE SAMPLE (First 3) ===\n")
                for i, msg in enumerate(messages[:3]):
                    debug_file.write(f"Message {i+1}: [{msg.sender_id}] {msg.content[:100]}...\n")
            
            print(f"    🔍 Debug dump created: {debug_filename}")
            return debug_filename
            
        except Exception as debug_err:
            print(f"    ⚠️ Failed to create debug dump: {debug_err}")
            return ""

    def extract_case(self, messages: List[Message], start_idx: int, end_idx: int, 
                    confidence: float = 0.5, summary: str = "") -> Case:
        """Extract a case from the message list"""
        # Convert to 0-based indexing and slice messages
        case_messages = messages[start_idx:end_idx]
        
        # Calculate duration in minutes
        duration = (case_messages[-1].timestamp - case_messages[0].timestamp).total_seconds() / 60.0
        
        case = Case(
            case_id=f"CASE_{self.case_counter:04d}",
            messages=case_messages,
            start_time=case_messages[0].timestamp,
            end_time=case_messages[-1].timestamp,
            participants=list(set(msg.sender_id for msg in case_messages)),
            summary=summary or f"Support case with {len(case_messages)} messages",
            channel_url=case_messages[0].channel_url,
            confidence=confidence,
            duration_minutes=duration,
            forced_ending=False,
            forced_starting=False
        )
        
        self.case_counter += 1
        self.current_channel_stats.cases_found += 1
        return case
    
    def process_channel(self, channel_url: str, messages: List[Message]) -> List[Case]:
        """Process a single channel using full conversation analysis"""
        print(f"\nProcessing channel: {channel_url}")
        print(f"Messages in channel: {len(messages)}")
        
        if not messages:
            print("  ⚠️ No messages in channel, skipping...")
            return []
        
        # Reset and initialize channel statistics (ensure clean state between channels)
        self.current_channel_stats = None
        self.current_channel_stats = ChannelStats(
            channel_url=channel_url,
            total_messages=len(messages),
            cases_found=0,
            input_tokens=0,
            output_tokens=0,
            llm_calls=0,
            processing_time=0.0
        )
        
        channel_cases = []
        
        try:
            # Step 1: Analyze the full conversation
            analysis = self.analyze_full_conversation(messages)
            
            initial_cases = analysis.get("complete_cases", [])
            cases_found = len(initial_cases)
            print(f"  🎯 LLM identified {cases_found} complete cases")
            
            # Step 2: Review and iteratively refine case segmentation
            refined_cases = initial_cases
            
            if self.review_enabled and cases_found > 0:
                print(f"  🔍 Starting iterative review process (max {self.max_review_iterations} iterations)...")
                
                for iteration in range(self.max_review_iterations):
                    # Review current cases
                    review_results = self.review_case_segmentation(messages, refined_cases, iteration)
                    
                    # Check if any cases need splitting
                    cases_needing_split = [r for r in review_results if r.needs_split]
                    
                    if not cases_needing_split:
                        print(f"    ✅ Review complete: No cases need further splitting (iteration {iteration})")
                        self.current_channel_stats.review_iterations = iteration
                        break
                    else:
                        print(f"    🔄 Iteration {iteration}: {len(cases_needing_split)} cases flagged for splitting")
                        
                        # Apply splits to create refined cases
                        refined_cases = self.apply_review_splits(messages, refined_cases, review_results)
                        
                        # Update statistics
                        self.current_channel_stats.cases_reviewed += len(review_results)
                        self.current_channel_stats.cases_split += len(cases_needing_split)
                        self.current_channel_stats.review_iterations = iteration + 1
                        
                        print(f"    📊 After splits: {len(refined_cases)} total cases")
                        
                        # If this is the last iteration, break
                        if iteration == self.max_review_iterations - 1:
                            print(f"    ⏹️ Maximum iterations reached ({self.max_review_iterations})")
                
                print(f"  🎯 Review complete: {len(initial_cases)} → {len(refined_cases)} cases")
            else:
                print(f"  ⚠️ Review disabled or no cases to review")
            
            # Step 3: Extract all refined cases
            for case_info in refined_cases:
                start_idx = case_info["start_message"] - 1  # Convert to 0-based
                end_idx = case_info["end_message"]
                
                if 0 <= start_idx < end_idx <= len(messages):
                    confidence = case_info.get("confidence", 0.5)
                    case = self.extract_case(
                        messages, start_idx, end_idx, 
                        confidence, case_info["summary"]
                    )
                    channel_cases.append(case)
                    print(f"    ✅ Extracted case {case.case_id} (conf: {confidence:.3f}): {case.summary[:60]}...")
            
            # Classification removed - now handled as separate step
            
        finally:
            # Always store channel statistics even if processing fails
            if self.current_channel_stats:
                self.channel_stats[channel_url] = self.current_channel_stats
        
        print(f"🎉 Channel complete: {len(channel_cases)} cases found")
        print(f"📊 Token usage: {self.current_channel_stats.input_tokens} input, {self.current_channel_stats.output_tokens} output")
        print(f"🤖 LLM calls: {self.current_channel_stats.llm_calls}")
        print(f"⏱️  Processing time: {self.current_channel_stats.processing_time:.2f}s")
        if self.current_channel_stats.review_iterations > 0:
            print(f"🔍 Review iterations: {self.current_channel_stats.review_iterations}")
            print(f"📋 Cases reviewed: {self.current_channel_stats.cases_reviewed}")
            print(f"✂️  Cases split: {self.current_channel_stats.cases_split}")
        
        return channel_cases
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame and return DataFrame with case numbers added"""
        print(f"🚀 Processing DataFrame with {len(df)} rows...")
        
        # Load messages from DataFrame
        channels = self.load_dataframe(df)
        
        # Process all channels to get cases
        cases = self.process_all_channels(channels)
        
        # Create output DataFrame with case assignments
        return self._create_output_dataframe(df, cases)
    
    
    def _create_output_dataframe(self, input_df: pd.DataFrame, cases: List[Case]) -> pd.DataFrame:
        """Create output DataFrame with case numbers and metadata"""
        print(f"📊 Creating output DataFrame...")
        
        # Create a copy of the input DataFrame
        output_df = input_df.copy()
        
        # Initialize case-related columns with proper dtype
        output_df['Case Number'] = ''
        output_df['case_start_time'] = pd.NaT
        output_df['case_end_time'] = pd.NaT
        # Convert to object dtype to avoid timezone incompatibility warnings
        output_df['case_start_time'] = output_df['case_start_time'].astype('object')
        output_df['case_end_time'] = output_df['case_end_time'].astype('object')
        output_df['case_duration_minutes'] = 0.0
        output_df['case_confidence'] = 0.0
        output_df['case_summary'] = ''
        output_df['case_forced_ending'] = False
        output_df['case_forced_starting'] = False
        
        # Create message_id to case mapping
        message_to_case = {}
        for case in cases:
            for message in case.messages:
                message_to_case[message.message_id] = case
        
        # Assign case information to each row
        for i, row in output_df.iterrows():
            message_id = str(row['Message ID'])
            if message_id in message_to_case:
                case = message_to_case[message_id]
                output_df.at[i, 'Case Number'] = case.case_id
                output_df.at[i, 'case_start_time'] = case.start_time
                output_df.at[i, 'case_end_time'] = case.end_time
                output_df.at[i, 'case_duration_minutes'] = case.duration_minutes
                output_df.at[i, 'case_confidence'] = case.confidence
                output_df.at[i, 'case_summary'] = case.summary
                output_df.at[i, 'case_forced_ending'] = case.forced_ending
                output_df.at[i, 'case_forced_starting'] = case.forced_starting
        
        print(f"✅ Created output DataFrame: {len(output_df)} rows, {len(output_df.columns)} columns")
        return output_df

    def process_all_channels(self, channels: Dict[str, List[Message]]) -> List[Case]:
        """Process all channels one by one"""
        all_cases = []
        all_classifications = []
        
        print(f"Processing {len(channels)} channels...")
        
        for i, (channel_url, messages) in enumerate(channels.items(), 1):
            print(f"\n--- Channel {i}/{len(channels)} ---")
            channel_cases = self.process_channel(channel_url, messages)
            all_cases.extend(channel_cases)
            
            # Accumulate classifications from this channel
            if hasattr(self, 'case_classifications') and self.case_classifications:
                all_classifications.extend(self.case_classifications)
        
        self.completed_cases = all_cases
        self.case_classifications = all_classifications  # Store all classifications
        self._invalidate_cache()  # Invalidate cache when new cases are added
        return all_cases
    
    def _invalidate_cache(self):
        """Invalidate all cached data"""
        self._cached_total_stats = None
        self._sorted_cases_cache = None
        self._cache_invalidated = True
    
    @property
    def total_stats(self) -> Dict[str, Any]:
        """Get cached total statistics across all channels"""
        if self._cached_total_stats is None or self._cache_invalidated:
            total_input_tokens = sum(stats.input_tokens for stats in self.channel_stats.values())
            total_output_tokens = sum(stats.output_tokens for stats in self.channel_stats.values())
            total_llm_calls = sum(stats.llm_calls for stats in self.channel_stats.values())
            total_processing_time = sum(stats.processing_time for stats in self.channel_stats.values())
            
            # Calculate confidence statistics
            total_confidence = sum(case.confidence for case in self.completed_cases)
            avg_confidence = total_confidence / len(self.completed_cases) if self.completed_cases else 0.0
            
            confidence_distribution = {
                "high_confidence": len([c for c in self.completed_cases if c.confidence >= 0.8]),
                "medium_confidence": len([c for c in self.completed_cases if 0.5 <= c.confidence < 0.8]),
                "low_confidence": len([c for c in self.completed_cases if c.confidence < 0.5])
            }
            
            # Review statistics
            total_review_iterations = sum(stats.review_iterations for stats in self.channel_stats.values())
            total_cases_reviewed = sum(stats.cases_reviewed for stats in self.channel_stats.values())
            total_cases_split = sum(stats.cases_split for stats in self.channel_stats.values())
            channels_with_review = len([c for c in self.channel_stats.values() if c.review_iterations > 0])
            
            self._cached_total_stats = {
                "total_cases": len(self.completed_cases),
                "total_channels": len(self.channel_stats),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_llm_calls": total_llm_calls,
                "total_processing_time": total_processing_time,
                "average_confidence": avg_confidence,
                "confidence_distribution": confidence_distribution,
                "total_review_iterations": total_review_iterations,
                "total_cases_reviewed": total_cases_reviewed,
                "total_cases_split": total_cases_split,
                "channels_with_review": channels_with_review
            }
            self._cache_invalidated = False
            
        return self._cached_total_stats
    
    @property
    def sorted_cases(self) -> List[Case]:
        """Get cached sorted cases by channel and time"""
        if self._sorted_cases_cache is None or self._cache_invalidated:
            self._sorted_cases_cache = sorted(
                self.completed_cases, 
                key=lambda x: (x.channel_url, x.start_time)
            )
        return self._sorted_cases_cache
    
    
    def export_segmentation_summary_md(self, filepath: Optional[str] = None):
        """Export comprehensive segmentation summary across all channels"""
        if filepath is None:
            output_config = get_output_config("channel")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = output_config.markdown_file.replace('.md', f'_segmentation_summary_{timestamp}.md')
            filepath = base_path
        
        print(f"📋 Exporting segmentation summary...")
        print(f"  📁 Output file: {filepath}")
        
        content = "# Case Segmentation Summary Report - Channel Algorithm\n\n"
        
        # Use cached overall statistics
        total_stats = self.total_stats
        
        content += f"**Algorithm:** Channel Full Conversation\n"
        content += f"**Total Cases Found:** {total_stats['total_cases']}\n"
        content += f"**Total Channels Processed:** {total_stats['total_channels']}\n"
        content += f"**Average Confidence:** {total_stats['average_confidence']:.3f}\n\n"
        
        # Use cached confidence distribution
        confidence_dist = total_stats['confidence_distribution']
        total_cases = total_stats['total_cases']
        
        content += "## Confidence Distribution\n\n"
        content += f"- **High Confidence (≥0.8):** {confidence_dist['high_confidence']} cases ({confidence_dist['high_confidence']/total_cases*100:.1f}%)\n"
        content += f"- **Medium Confidence (0.5-0.8):** {confidence_dist['medium_confidence']} cases ({confidence_dist['medium_confidence']/total_cases*100:.1f}%)\n"
        content += f"- **Low Confidence (<0.5):** {confidence_dist['low_confidence']} cases ({confidence_dist['low_confidence']/total_cases*100:.1f}%)\n\n"
        
        content += "## Review Analysis\n\n"
        total_review_iterations = total_stats['total_review_iterations']
        total_cases_reviewed = total_stats['total_cases_reviewed']
        total_cases_split = total_stats['total_cases_split']
        channels_with_review = total_stats['channels_with_review']
        
        content += f"- **Review Enabled:** {'Yes' if self.review_enabled else 'No'}\n"
        content += f"- **Channels with Review:** {channels_with_review} out of {total_stats['total_channels']}\n"
        content += f"- **Total Review Iterations:** {total_review_iterations}\n"
        content += f"- **Cases Reviewed:** {total_cases_reviewed}\n"
        content += f"- **Cases Split After Review:** {total_cases_split}\n"
        if total_cases_split > 0 and total_cases_reviewed > 0:
            content += f"- **Split Rate:** {total_cases_split/total_cases_reviewed*100:.1f}% of reviewed cases were split\n"
        content += "\n"
        
        # All segmentations table
        content += "## All Case Segmentations\n\n"
        content += "| Case ID | Channel | Start Time | End Time | Duration | Confidence | Summary |\n"
        content += "|---------|---------|------------|----------|----------|------------|----------|\n"
        
        # Use cached sorted cases
        for case in self.sorted_cases:
            short_channel = case.channel_url.split('_')[-1][:15] + "..."
            start_time = case.start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_time = case.end_time.strftime('%Y-%m-%d %H:%M:%S')
            duration = f"{case.duration_minutes:.1f}m"
            confidence = f"{case.confidence:.3f}"
            summary = case.summary[:40] + "..." if len(case.summary) > 40 else case.summary
            
            content += f"| {case.case_id} | {short_channel} | {start_time} | {end_time} | {duration} | {confidence} | {summary} |\n"
        
        content += "\n"
        
        # Channel breakdown
        content += "## Segmentation by Channel\n\n"
        
        # Group cases by channel
        channel_cases = {}
        for case in self.completed_cases:
            if case.channel_url not in channel_cases:
                channel_cases[case.channel_url] = []
            channel_cases[case.channel_url].append(case)
        
        for channel_url, cases in channel_cases.items():
            short_channel = channel_url.split('_')[-1][:20]
            avg_conf = sum(c.confidence for c in cases) / len(cases)
            total_duration = sum(c.duration_minutes for c in cases)
            
            content += f"### Channel: {short_channel}...\n\n"
            content += f"- **Cases:** {len(cases)}\n"
            content += f"- **Average Confidence:** {avg_conf:.3f}\n"
            content += f"- **Total Duration:** {total_duration:.1f} minutes\n"
            
            # Add review information for this channel
            channel_stats = self.channel_stats[channel_url]
            if channel_stats.review_iterations > 0:
                content += f"- **Review Iterations:** {channel_stats.review_iterations}\n"
                content += f"- **Cases Reviewed:** {channel_stats.cases_reviewed}\n"
                content += f"- **Cases Split:** {channel_stats.cases_split}\n"
            else:
                content += f"- **Review Status:** No review performed\n"
            
            content += f"- **Full URL:** `{channel_url}`\n\n"
            
            for case in sorted(cases, key=lambda x: x.start_time):
                time_span = f"{case.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {case.end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                content += f"  - **{case.case_id}** ({time_span}, {case.duration_minutes:.1f}m, conf: {case.confidence:.3f}): {case.summary}\n"
            content += "\n"
        
        # Recommendations using cached stats
        content += "## Quality Assessment\n\n"
        avg_confidence = total_stats['average_confidence']
        low_conf = confidence_dist['low_confidence']
        
        if avg_confidence >= 0.7:
            content += "✅ **Good Segmentation Quality** - High average confidence suggests reliable case boundaries.\n\n"
        elif avg_confidence >= 0.5:
            content += "⚠️ **Moderate Segmentation Quality** - Consider reviewing low confidence cases manually.\n\n"
        else:
            content += "❌ **Poor Segmentation Quality** - Many low confidence cases suggest algorithm tuning needed.\n\n"
        
        if low_conf > total_cases * 0.3:
            content += f"⚠️ **Warning:** {low_conf} cases ({low_conf/total_cases*100:.1f}%) have low confidence. Manual review recommended.\n\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Exported segmentation summary to {filepath}")


def demo_dataframe_processing():
    """Demo function showing DataFrame-based processing"""
    from data_preprocessor import DataPreprocessor
    
    print("=== DataFrame Processing Demo ===")
    
    # Step 1: Preprocess data to DataFrame
    preprocessor = DataPreprocessor()
    df = preprocessor.process_to_dataframe('assets/support_msg.csv', mode='r3')
    
    # Step 2: Process with case parser
    parser = ChannelSegmenter()
    output_df = parser.process_dataframe(df)
    
    print(f"\n✅ DataFrame processing complete!")
    print(f"📊 Output shape: {output_df.shape}")
    print(f"📊 Case columns: {[col for col in output_df.columns if col.startswith('case_')]}")
    print(f"📊 Unique cases: {output_df['Case Number'].nunique()}")
    print(f"📊 Sample output:\n{output_df[['message_id', 'Case Number', 'case_confidence', 'case_summary']].head()}")
    
    return output_df


def main():
    """Main execution function"""
    print("=== Configuration Status ===")
    config = get_config()
    status = config.validate_config()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    parser = ChannelSegmenter(
        llm_provider=None  # Use primary_provider and model_type from config.json
    )
    
    # Load data grouped by channel
    channels = parser.load_csv('assets/preprocessed_support_msg.csv')
    
    # Process all channels
    print(f"\n🚀 Starting case processing...")
    cases = parser.process_all_channels(channels)
    
    # Export results (Markdown only - CSV and JSON removed)
    print(f"\n📤 Exporting results...")
    parser.export_segmentation_summary_md()
    
    print(f"\n=== CHANNEL ALGORITHM COMPLETE ===")
    print(f"Total cases: {len(cases)}")
    print(f"Total channels: {len(channels)}")


if __name__ == "__main__":
    main()