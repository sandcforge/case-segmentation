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
import time
import re
import os
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
    truncated: bool = False  # True if conversation was truncated due to length limits


@dataclass
class ChannelStats:
    channel_url: str
    total_messages: int
    cases_found: int
    input_tokens: int
    output_tokens: int
    llm_calls: int
    processing_time: float
    was_truncated: bool = False


class ChannelCaseParser:
    def __init__(self, llm_provider: Optional[str] = None, llm_model_type: Optional[str] = None):
        # Load configuration
        self.config = get_config()
        
        # Get parsing configuration
        parsing_config = get_parsing_config("channel")
        self.max_context_tokens = parsing_config.max_context_tokens
        self.reserve_tokens = parsing_config.reserve_tokens
        
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
                    timestamp = datetime.fromisoformat(row['created_time'].replace('Z', '+00:00'))
                    
                    # Use preprocessed content directly (already cleaned)
                    content = row['message']
                    
                    # Extract all columns that start with "round"
                    round_columns = {col: row.get(col, '') for col in row.keys() if col.startswith('round')}
                    
                    message = Message(
                        message_id=row['message_id'],
                        message_type=row['type'],
                        content=content,
                        sender_id=row['sender_id'],
                        timestamp=timestamp,
                        channel_url=row['channel_url'],
                        sender_type=row['sender_type'],
                        review=row['review'],
                        file_url=row.get('file_url', ''),
                        real_sender_id=row.get('real_sender_id', ''),
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
    
    def format_conversation_for_llm(self, messages: List[Message], was_truncated: bool = False) -> str:
        """Format entire conversation for LLM analysis"""
        formatted = "COMPLETE CONVERSATION:\n\n"
        
        if was_truncated:
            formatted += "⚠️ NOTE: This conversation was truncated due to length limits.\n\n"
        
        for i, msg in enumerate(messages, 1):
            # Content is already cleaned from preprocessing
            formatted += f"<message id=\"{i}\">\n"
            formatted += f"<sender>{msg.sender_id}</sender>\n"
            formatted += f"<sender_type>{msg.sender_type}</sender_type>\n"
            formatted += f"<time>{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</time>\n"
            formatted += f"<content>{msg.content}</content>\n"
            formatted += f"</message>\n\n"
            
        return formatted
    
    def truncate_conversation_to_fit(self, messages: List[Message]) -> tuple[List[Message], bool]:
        """Truncate conversation to fit within token limits"""
        # Calculate approximate prompt tokens (reserve for instructions)
        available_tokens = self.max_context_tokens - self.reserve_tokens
        
        # Try with full conversation first
        full_conversation = self.format_conversation_for_llm(messages)
        total_tokens = self.count_tokens(full_conversation)
        
        if total_tokens <= available_tokens:
            print(f"  📊 Full conversation fits: {total_tokens} tokens")
            return messages, False
        
        print(f"  ⚠️ Conversation too long: {total_tokens} tokens > {available_tokens} limit")
        print(f"  ✂️ Truncating conversation to fit token limit...")
        
        # Binary search to find maximum messages that fit
        left, right = 1, len(messages)
        best_count = 1
        
        while left <= right:
            mid = (left + right) // 2
            truncated_messages = messages[:mid]
            truncated_conversation = self.format_conversation_for_llm(truncated_messages, was_truncated=True)
            tokens = self.count_tokens(truncated_conversation)
            
            if tokens <= available_tokens:
                best_count = mid
                left = mid + 1
            else:
                right = mid - 1
        
        truncated_messages = messages[:best_count]
        final_tokens = self.count_tokens(self.format_conversation_for_llm(truncated_messages, was_truncated=True))
        print(f"  ✅ Truncated to {best_count} messages ({final_tokens} tokens)")
        
        return truncated_messages, True
    
    def analyze_full_conversation_back(self, messages: List[Message], was_truncated: bool = False) -> Dict[str, Any]:
        """Use LLM to analyze entire conversation for case boundaries"""
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
        
        truncation_note = ""
        if was_truncated:
            truncation_note = "⚠️ NOTE: This conversation was truncated due to length limits. The last case may be incomplete.\n\n"
        
        prompt = f"""
        ✦ You are an expert AI assistant specializing in analyzing customer service interactions. Your task is to process and segment this continuous chat log into distinct customer service cases.

        The conversation contains interactions between a support team and customers. You must identify natural start and end points of each distinct issue or topic discussed.

        {truncation_note}{self.format_conversation_for_llm(messages, was_truncated)}

        For each case you identify, provide these four fields:

        1. **Start Message**: The message number where the case begins (when a new, distinct topic is introduced)
        2. **End Message**: The message number where the case concludes (when issue is resolved, topic concluded, or conversation moves on)
        3. **Summary**: A concise, one-to-two-sentence summary including the initial problem, key discussion points, and final resolution or status
        4. **Confidence**: Your confidence level (0.9 = High, 0.7 = Medium, 0.5 = Low) that this represents a complete and distinct case

        Instructions:
        • Process the entire conversation in a single pass, leveraging your full context window
        • A case begins when a new, distinct topic is introduced
        • A case ends when the issue is resolved, clearly concluded, or conversation moves on without resolution
        • A single case can span multiple messages and time periods
        • If a topic is raised but not resolved in the log, note this in summary and assign Medium confidence (0.7)
        • Be mindful that intermediate responses, status updates, and clarifications are part of the same case, not new cases

        IMPORTANT: Return ONLY valid JSON, no other text or explanation.

        Required JSON format:
        {{
            "complete_cases": [
                {{
                    "start_message": 1,
                    "end_message": 8,
                    "summary": "Customer reported delivery issue with order #12345. Support investigated tracking and confirmed package was delivered to correct address, providing photo evidence. Issue resolved.",
                    "confidence": 0.9
                }}
            ],
            "analysis": "Brief explanation of your segmentation decisions",
            "total_messages_analyzed": {len(messages)}
        }}

        If no complete cases found, return empty array for complete_cases.
        Return ONLY the JSON object, nothing else.
        """
        
        try:
            # Use LLM manager for analysis
            response = self.llm_manager.analyze_case_boundaries(prompt)
            
            # Update statistics
            self.current_channel_stats.input_tokens += response.input_tokens
            self.current_channel_stats.output_tokens += response.output_tokens
            self.current_channel_stats.llm_calls += 1
            self.current_channel_stats.processing_time += response.processing_time
            
            print(f"  📊 LLM response: {response.input_tokens} input + {response.output_tokens} output tokens")
            
            # Enhanced JSON parsing with multiple strategies
            try:
                # Strategy 1: Direct JSON parsing
                result = json.loads(response.content)
                print("  ✅ Direct JSON parsing successful")
                
                # Validate the structure
                if self.validate_llm_response(result):
                    return result
                else:
                    print("  ⚠️ JSON structure invalid, missing complete_cases key")
                    raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError) as err:
                print(f"  ⚠️ Direct JSON parsing failed or invalid structure: {err}")
                
                try:
                    # Strategy 2: Extract JSON from mixed text response
                    extracted_json = self.extract_json_from_response(response.content)
                    result = json.loads(extracted_json)
                    print("  ✅ Successfully extracted JSON from mixed response")
                    
                    # Validate the structure
                    if self.validate_llm_response(result):
                        return result
                    else:
                        print("  ⚠️ Extracted JSON structure invalid")
                        raise ValueError("Invalid JSON structure")
                except (json.JSONDecodeError, ValueError):
                    print(f"  ⚠️ JSON extraction failed, trying cleaning...")
                    
                    try:
                        # Strategy 3: Clean control characters and retry
                        cleaned_response = self.clean_message_content(response.content)
                        extracted_json = self.extract_json_from_response(cleaned_response)
                        result = json.loads(extracted_json)
                        print("  ✅ Successfully parsed with cleaned extraction")
                        
                        # Validate the structure
                        if self.validate_llm_response(result):
                            return result
                        else:
                            print("  ⚠️ Cleaned JSON structure invalid")
                            raise ValueError("Invalid JSON structure")
                    except:
                        print(f"  ❌ All JSON parsing strategies failed")
                        print(f"  📝 Raw LLM response: {response.content[:300]}...")
                        
                        # DEBUG: Dump full response to file for analysis
                        debug_filename = f"debug_llm_response_{int(time.time())}.txt"
                        try:
                            with open(debug_filename, 'w', encoding='utf-8') as debug_file:
                                debug_file.write("=== LLM RESPONSE DEBUG DUMP ===\n")
                                debug_file.write(f"Model: {response.model}\n")
                                debug_file.write(f"Provider: {response.provider}\n")
                                debug_file.write(f"Input tokens: {response.input_tokens}\n")
                                debug_file.write(f"Output tokens: {response.output_tokens}\n")
                                debug_file.write(f"Processing time: {response.processing_time:.2f}s\n")
                                debug_file.write(f"Message count: {len(messages)}\n")
                                debug_file.write(f"Truncated: {was_truncated}\n")
                                debug_file.write("\n=== FULL RESPONSE CONTENT ===\n")
                                debug_file.write(response.content)
                                debug_file.write("\n\n=== PROMPT SENT ===\n")
                                debug_file.write(prompt)
                            print(f"  🔍 Debug response saved to: {debug_filename}")
                        except Exception as debug_err:
                            print(f"  ⚠️ Failed to save debug file: {debug_err}")
                        
                        # Strategy 4: Retry with explicit JSON format instructions
                        retry_result = self.retry_llm_for_json(messages, response.content)
                        if retry_result:
                            return retry_result
                        
                        print("  ⚠️ Creating single case with parsing failure note")
                        return {
                            "complete_cases": [{
                                "start_message": 1,
                                "end_message": len(messages),
                                "summary": "LLM response parsing failed after retry - processed as single case",
                                "confidence": 0.2
                            }],
                            "analysis": "LLM JSON parsing failed after retry attempt",
                            "total_messages_analyzed": len(messages)
                        }
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return {
                "complete_cases": [{
                    "start_message": 1,
                    "end_message": len(messages),
                    "summary": f"LLM analysis failed ({str(e)[:50]}) - processed as single case",
                    "confidence": 0.1
                }],
                "analysis": f"LLM analysis exception: {e}",
                "total_messages_analyzed": len(messages)
            }
    
    def analyze_full_conversation(self, messages: List[Message], was_truncated: bool = False) -> Dict[str, Any]:
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
        
        # Perform single analysis with built-in review
        result = self._perform_initial_analysis(messages, was_truncated)
        
        if result is None:
            # Create fallback result when analysis fails
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
        
        # Calculate final confidence
        cases = result.get("complete_cases", [])
        if cases:
            avg_confidence = sum(case.get("confidence", 0.0) for case in cases) / len(cases)
            print(f"  🎯 Analysis complete: {len(cases)} cases, avg confidence: {avg_confidence:.3f}")
        else:
            print("  ⚠️ No cases identified in analysis")
        
        return result
    
    def _perform_initial_analysis(self, messages: List[Message], was_truncated: bool) -> Dict[str, Any]:
        """Perform the initial LLM analysis (same as original method but with iteration tracking)"""
        truncation_note = ""
        if was_truncated:
            truncation_note = "⚠️ NOTE: This conversation was truncated due to length limits. The last case may be incomplete.\n\n"
        
        prompt1 = f"""
        ✦ You are an expert conversation analyst specializing in customer service interaction segmentation. Your task is to analyze this customer support conversation and identify distinct cases based on comprehensive boundary detection criteria.

        ## Conversation Analysis Framework

        **CONVERSATION TO ANALYZE:**
        {truncation_note}{self.format_conversation_for_llm(messages, was_truncated)}

        ## Detailed Boundary Identification Criteria

        ### Primary Case Boundary Indicators:
        1. **Topic Transitions**: Clear shifts in the subject matter or problem being discussed
        2. **Participant Changes**: New customer joining conversation or handoff between support agents
        3. **Temporal Gaps**: Significant time breaks that indicate conversation resumption (>30 minutes as guideline)
        4. **Resolution Points**: Explicit confirmation that an issue has been resolved or closed
        5. **New Issue Introduction**: Customer raises a completely different concern or problem

        ### Secondary Indicators:
        1. **Context Shifts**: Changes from troubleshooting to billing, technical to account issues, etc.
        2. **Process Changes**: Shifts from initial inquiry to escalation, or from support to sales
        3. **Reference Number Changes**: Different ticket/order/case numbers being discussed
        4. **Communication Mode Shifts**: Phone to chat handoffs, different departments involved

        ## Systematic Analysis Approach

        ### Step 1: Chronological Reading
        - Read through messages sequentially, tracking conversational context
        - Note timestamps and identify any significant time gaps
        - Track participants and their roles throughout the conversation

        ### Step 2: Context Tracking
        - Identify the main issue or topic being discussed in each segment
        - Note when new topics are introduced vs. when existing topics are continued
        - Distinguish between clarifications/follow-ups vs. entirely new issues

        ### Step 3: Boundary Detection
        - Apply primary boundary indicators to identify potential case breaks
        - Validate boundaries using secondary indicators
        - Ensure each identified case has a clear beginning and logical conclusion

        ### Step 4: Case Validation
        - Verify each case represents a complete, coherent interaction
        - Ensure cases don't artificially split related conversations
        - Confirm cases don't inappropriately merge distinct issues

        ## Special Considerations

        ### Message Type Analysis:
        - **Greeting Messages**: Usually part of case initiation, not separate cases
        - **Status Updates**: Continuation of existing cases unless introducing new issues
        - **Follow-up Questions**: Part of the same case unless opening new topics
        - **Closing/Thank You**: Usually case conclusion markers

        ### Quality Thresholds:
        - **High Confidence (0.8-1.0)**: Clear boundaries with obvious topic shifts, resolution points, or participant changes
        - **Medium Confidence (0.5-0.8)**: Probable boundaries based on context shifts or timing, but some ambiguity exists
        - **Low Confidence (0.3-0.5)**: Uncertain boundaries where segmentation is subjective or unclear

        ## Case Extraction Guidelines

        For each identified case, provide:
        1. **Start Message**: Message number where the distinct case begins
        2. **End Message**: Message number where the case concludes or transitions
        3. **Summary**: Comprehensive 1-2 sentence summary including: initial problem, key actions taken, and final resolution status
        4. **Confidence**: Assessment of boundary certainty based on analysis criteria above

        ## Final Review and Adjustment

        Before providing your final output, revisit your segmentation analysis:
        - Review each identified case boundary for accuracy
        - Check if any cases should be merged or split
        - Verify that no distinct issues were missed
        - Adjust confidence scores based on boundary clarity
        - Ensure summaries accurately reflect the complete case content

        ## Output Requirements

        Return analysis in this EXACT JSON structure:

        {{
            "complete_cases": [
                {{
                    "start_message": 1,
                    "end_message": 8,
                    "summary": "Customer reported delivery issue with order #12345. Support investigated tracking and confirmed package was delivered to correct address, providing photo evidence. Issue resolved.",
                    "confidence": 0.9
                }}
            ],
            "analysis": "Comprehensive explanation incorporating business logic, customer journey insights, and boundary reasoning based on the 6-step systematic analysis",
            "total_messages_analyzed": {len(messages)}
        }}
        </output>
        """
    
        prompt = f"""

        Analyze customer service conversations and segment them into separate cases by topic.

        Segmentation Rules:
        1. Clear topic change → New segment
        2. Customer says "also", "another question", "by the way" → New segment  
        3. New question after problem resolution → New segment
        4. 24+ hour gap → New segment

        Do NOT segment for:
        - Follow-up clarifications on same issue
        - Polite responses like "thank you"
        - Information confirmations

        <thinking>
        Step 1: Read through the entire conversation chronologically to understand overall flow
        Step 2: Identify main topics and issues discussed throughout the conversation
        Step 3: Mark potential segment boundaries based on clear topic changes
        Step 4: Check for explicit transition signals ("also", "another question", "by the way")
        Step 5: Look for resolution points followed by new questions or issues
        Step 6: Evaluate time gaps and assess whether they indicate natural conversation breaks
        Step 7: Validate that each segment represents a coherent, actionable case
        Step 8: Apply business logic to ensure segments make sense for case management
        </thinking>


        **CONVERSATION TO ANALYZE:**
        {truncation_note}{self.format_conversation_for_llm(messages, was_truncated)}

        Output segmentation results in JSON format:
        <output>
        {{
            "complete_cases": [
                {{
                    "start_message": 1,
                    "end_message": 8,
                    "summary": "Brief description of the issue, actions taken, and resolution status",
                    "confidence": 0.9
                }}
            ],
            "analysis": "Comprehensive explanation incorporating business logic, customer journey insights, and boundary reasoning based on the systematic analysis above",
            "total_messages_analyzed": [total_number_of_messages]
        }}
        </output>
        """
        
        return self._execute_llm_call(prompt, "initial analysis", messages)
    
    
    def _execute_llm_call(self, prompt: str, call_type: str, messages: List[Message] = None) -> Dict[str, Any]:
        """Execute an LLM call with comprehensive error handling and debug dumping"""
        if messages is None:
            messages = []
            
        try:
            response = self.llm_manager.analyze_case_boundaries(prompt)
            
            # Log ALL LLM interactions for debugging
            self._log_llm_interaction(call_type, prompt, response, messages, success=True)
            
            # Update statistics
            self.current_channel_stats.input_tokens += response.input_tokens
            self.current_channel_stats.output_tokens += response.output_tokens
            self.current_channel_stats.llm_calls += 1
            self.current_channel_stats.processing_time += response.processing_time
            
            print(f"    📊 {call_type}: {response.input_tokens} input + {response.output_tokens} output tokens")
            
            # Create context for debug dumps
            response_context = {
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "processing_time": f"{response.processing_time:.2f}s"
            }
            
            # Enhanced JSON parsing with multiple strategies
            try:
                # Strategy 1: Direct JSON parsing
                result = json.loads(response.content)
                print(f"    ✅ {call_type}: Direct JSON parsing successful")
                
                # Validate the structure
                if self.validate_llm_response(result):
                    return result
                else:
                    print(f"    ⚠️ {call_type}: JSON structure invalid, missing complete_cases key")
                    self._create_debug_dump("validation_failed", call_type, messages, prompt, 
                                          response.content, "Direct JSON validation failed", response_context)
                    raise ValueError("Invalid JSON structure")
            except (json.JSONDecodeError, ValueError) as err:
                print(f"    ⚠️ {call_type}: Direct JSON parsing failed: {err}")
                
                try:
                    # Strategy 2: Extract JSON from mixed text response
                    extracted_json = self.extract_json_from_response(response.content)
                    result = json.loads(extracted_json)
                    print(f"    ✅ {call_type}: Successfully extracted JSON from mixed response")
                    
                    # Validate the structure
                    if self.validate_llm_response(result):
                        return result
                    else:
                        print(f"    ⚠️ {call_type}: Extracted JSON structure invalid")
                        self._create_debug_dump("extraction_validation_failed", call_type, messages, prompt, 
                                              response.content, f"JSON extraction validation failed: {err}", response_context)
                        raise ValueError("Invalid JSON structure")
                except (json.JSONDecodeError, ValueError) as extract_err:
                    print(f"    ⚠️ {call_type}: JSON extraction failed, trying cleaning...")
                    
                    try:
                        # Strategy 3: Clean control characters and retry
                        cleaned_response = self.clean_message_content(response.content)
                        extracted_json = self.extract_json_from_response(cleaned_response)
                        result = json.loads(extracted_json)
                        print(f"    ✅ {call_type}: Successfully parsed with cleaned extraction")
                        
                        # Validate the structure
                        if self.validate_llm_response(result):
                            return result
                        else:
                            print(f"    ⚠️ {call_type}: Cleaned JSON structure invalid")
                            self._create_debug_dump("clean_validation_failed", call_type, messages, prompt, 
                                                  response.content, f"Cleaned JSON validation failed: {extract_err}", response_context)
                            raise ValueError("Invalid JSON structure")
                    except Exception as clean_err:
                        print(f"    ❌ {call_type}: All JSON parsing strategies failed")
                        
                        # Comprehensive debug dump for complete parsing failure
                        error_details = f"""
Direct parsing error: {err}
Extraction error: {extract_err}  
Cleaning error: {clean_err}
All parsing strategies exhausted.
                        """.strip()
                        
                        self._create_debug_dump("complete_parsing_failed", call_type, messages, prompt, 
                                              response.content, error_details, response_context)
                        return None
        
        except Exception as e:
            print(f"    ❌ {call_type} LLM call failed: {e}")
            
            # Log failed LLM interactions
            error_details = f"LLM API call exception: {str(e)}\nException type: {type(e).__name__}"
            self._log_llm_interaction(call_type, prompt, e, messages, success=False, error_details=error_details)
            
            # Check if exception has enhanced error response data
            error_response = getattr(e, 'error_response', None)
            
            if error_response:
                # Enhanced error with response details available
                print(f"    📋 Enhanced error details available: finish_reason={error_response.finish_reason}")
                
                response_content = f"[ERROR_RESPONSE_CONTENT]\nFinish Reason: {error_response.finish_reason}\n"
                if error_response.error_details:
                    response_content += f"Error Details: {json.dumps(error_response.error_details, indent=2, default=str)}\n"
                
                response_context = {
                    "model": error_response.model,
                    "provider": error_response.provider,
                    "input_tokens": error_response.input_tokens,
                    "output_tokens": error_response.output_tokens,
                    "processing_time": f"{error_response.processing_time:.2f}s",
                    "finish_reason": error_response.finish_reason,
                    "error_response_available": True
                }
                
                error_details = f"LLM API call exception: {str(e)}\nException type: {type(e).__name__}\nEnhanced error response captured with finish_reason: {error_response.finish_reason}"
                
                self._create_debug_dump("api_call_failed_with_response", call_type, messages, prompt, 
                                      response_content, error_details, response_context)
            else:
                # Standard error without response details
                error_details = f"LLM API call exception: {str(e)}\nException type: {type(e).__name__}\nNo enhanced error response available"
                self._create_debug_dump("api_call_failed", call_type, messages, prompt, 
                                      "", error_details, {"exception_type": type(e).__name__})
            return None
    
    def validate_llm_response(self, response_dict: Dict[str, Any]) -> bool:
        """Validate that LLM response has the required structure"""
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

    def retry_llm_for_json(self, messages: List[Message], original_response: str = "") -> Dict[str, Any]:
        """Retry LLM with more explicit JSON formatting instructions"""
        print("  🔄 Retrying LLM with explicit JSON format request...")
        
        prompt = f"""
The previous response was not in the correct JSON format. Please analyze this conversation and return ONLY a valid JSON object with this EXACT structure:

{{
    "complete_cases": [
        {{
            "start_message": 1,
            "end_message": {len(messages)},
            "summary": "Brief summary of the support case",
            "confidence": 0.8
        }}
    ],
    "analysis": "Brief analysis of the conversation",
    "total_messages_analyzed": {len(messages)}
}}

If you cannot identify distinct cases, return a single case spanning all messages.

Conversation to analyze:
{self.format_conversation_for_llm(messages[:50])}

IMPORTANT: Return ONLY the JSON object, no other text.
"""
        
        try:
            response = self.llm_manager.analyze_case_boundaries(prompt)
            
            # Log retry LLM interactions
            self._log_llm_interaction("json_retry", prompt, response, messages, success=True)
            
            result = json.loads(response.content)
            
            if self.validate_llm_response(result):
                print("  ✅ Retry successful - valid JSON structure received")
                return result
            else:
                print("  ⚠️ Retry returned invalid structure")
                self._create_debug_dump("retry_validation_failed", "json_retry", messages, 
                                      prompt, response.content, "Retry response failed validation")
                return None
                
        except Exception as e:
            print(f"  ❌ Retry failed: {e}")
            
            # Log failed retry attempts
            error_details = f"Retry exception: {str(e)}\nException type: {type(e).__name__}"
            self._log_llm_interaction("json_retry", prompt, e, messages, success=False, error_details=error_details)
            
            self._create_debug_dump("retry_failed", "json_retry", messages, 
                                  prompt, "", f"Retry exception: {str(e)}")
            return None
    
    
    def extract_case(self, messages: List[Message], start_idx: int, end_idx: int, 
                    confidence: float = 0.5, summary: str = "", is_last_case: bool = False, 
                    was_truncated: bool = False) -> Case:
        """Extract a case from the message list"""
        # Convert to 0-based indexing and slice messages
        case_messages = messages[start_idx:end_idx]
        
        # Calculate duration in minutes
        duration = (case_messages[-1].timestamp - case_messages[0].timestamp).total_seconds() / 60.0
        
        # Determine if this case is truncated (only last case can be truncated)
        truncated = is_last_case and was_truncated
        
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
            forced_starting=False,
            truncated=truncated
        )
        
        if truncated:
            case.summary = f"{case.summary} (TRUNCATED - conversation may be incomplete)"
        
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
        
        # Initialize channel statistics
        self.current_channel_stats = ChannelStats(
            channel_url=channel_url,
            total_messages=len(messages),
            cases_found=0,
            input_tokens=0,
            output_tokens=0,
            llm_calls=0,
            processing_time=0.0,
            was_truncated=False
        )
        
        channel_cases = []
        
        # Step 1: Check if conversation fits in token limit
        truncated_messages, was_truncated = self.truncate_conversation_to_fit(messages)
        self.current_channel_stats.was_truncated = was_truncated
        
        # Step 2: Analyze the full (or truncated) conversation
        analysis = self.analyze_full_conversation(truncated_messages, was_truncated)
        
        cases_found = len(analysis.get("complete_cases", []))
        print(f"  🎯 LLM identified {cases_found} complete cases")
        
        # Step 3: Extract all identified cases
        for case_info in analysis.get("complete_cases", []):
            start_idx = case_info["start_message"] - 1  # Convert to 0-based
            end_idx = case_info["end_message"]
            
            if 0 <= start_idx < end_idx <= len(truncated_messages):
                # Check if this is the last case (for truncation marking)
                is_last_case = (case_info == analysis["complete_cases"][-1])
                
                confidence = case_info.get("confidence", 0.5)
                case = self.extract_case(
                    truncated_messages, start_idx, end_idx, 
                    confidence, case_info["summary"], is_last_case, was_truncated
                )
                channel_cases.append(case)
                print(f"    ✅ Extracted case {case.case_id} (conf: {confidence:.3f}): {case.summary[:60]}...")
        
        # Store channel statistics
        self.channel_stats[channel_url] = self.current_channel_stats
        
        print(f"🎉 Channel complete: {len(channel_cases)} cases found")
        print(f"📊 Token usage: {self.current_channel_stats.input_tokens} input, {self.current_channel_stats.output_tokens} output")
        print(f"🤖 LLM calls: {self.current_channel_stats.llm_calls}")
        print(f"⏱️  Processing time: {self.current_channel_stats.processing_time:.2f}s")
        if was_truncated:
            print(f"⚠️  Conversation was truncated to fit token limits")
        
        return channel_cases
    
    def process_all_channels(self, channels: Dict[str, List[Message]]) -> List[Case]:
        """Process all channels one by one"""
        all_cases = []
        
        print(f"Processing {len(channels)} channels...")
        
        for i, (channel_url, messages) in enumerate(channels.items(), 1):
            print(f"\n--- Channel {i}/{len(channels)} ---")
            channel_cases = self.process_channel(channel_url, messages)
            all_cases.extend(channel_cases)
        
        self.completed_cases = all_cases
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
            
            # Truncation statistics
            truncated_cases = len([c for c in self.completed_cases if c.truncated])
            truncated_channels = len([c for c in self.channel_stats.values() if c.was_truncated])
            
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
                "truncated_cases": truncated_cases,
                "truncated_channels": truncated_channels
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
    
    def export_json(self, filepath: Optional[str] = None):
        """Export cases to JSON with channel statistics"""
        if filepath is None:
            output_config = get_output_config("channel")
            filepath = output_config.json_file
        
        print(f"📄 Exporting JSON report...")
        print(f"  📁 Output file: {filepath}")
        # Cases data with enhanced information
        cases_data = []
        for case in self.completed_cases:
            case_data = {
                "case_id": case.case_id,
                "summary": case.summary,
                "start_time": case.start_time.isoformat(),
                "end_time": case.end_time.isoformat(),
                "duration_minutes": round(case.duration_minutes, 2),
                "confidence": round(case.confidence, 3),
                "participants": case.participants,
                "channel_url": case.channel_url,
                "message_count": len(case.messages),
                "forced_ending": case.forced_ending,
                "forced_starting": case.forced_starting,
                "truncated": case.truncated,
                "messages": [
                    {
                        "message_id": msg.message_id,
                        "sender_id": msg.sender_id,
                        "timestamp": msg.timestamp.isoformat(),
                        "content": msg.content,
                        "type": msg.message_type
                    }
                    for msg in case.messages
                ]
            }
            cases_data.append(case_data)
        
        # Channel statistics
        channel_stats_data = {}
        for channel_url, stats in self.channel_stats.items():
            channel_stats_data[channel_url] = {
                "total_messages": stats.total_messages,
                "cases_found": stats.cases_found,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "total_tokens": stats.input_tokens + stats.output_tokens,
                "llm_calls": stats.llm_calls,
                "processing_time": round(stats.processing_time, 2),
                "was_truncated": stats.was_truncated
            }
        
        # Use cached total statistics
        total_stats = self.total_stats
        
        # Combined output with enhanced metrics using cached stats
        output_data = {
            "summary": {
                "total_cases": total_stats["total_cases"],
                "total_channels": total_stats["total_channels"],
                "average_confidence": round(total_stats["average_confidence"], 3),
                "confidence_distribution": total_stats["confidence_distribution"],
                "total_input_tokens": total_stats["total_input_tokens"],
                "total_output_tokens": total_stats["total_output_tokens"],
                "total_tokens": total_stats["total_tokens"],
                "total_llm_calls": total_stats["total_llm_calls"],
                "total_processing_time": round(total_stats["total_processing_time"], 2),
                "truncated_cases": total_stats["truncated_cases"],
                "truncated_channels": total_stats["truncated_channels"],
                "algorithm": "Channel Full Conversation"
            },
            "channel_statistics": channel_stats_data,
            "cases": cases_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(cases_data)} cases and statistics to {filepath}")
    
    def export_cases_csv(self, filepath: Optional[str] = None):
        """Export cases to CSV with original message data plus case numbers"""
        if filepath is None:
            output_config = get_output_config("channel")
            # Create CSV filename based on markdown filename
            base_path = output_config.markdown_file.replace('.md', '_with_cases.csv')
            filepath = base_path
        
        print(f"📄 Exporting CSV with case numbers...")
        print(f"  📁 Output file: {filepath}")
        
        # Collect all round column names from all messages
        round_column_names = set()
        for case in self.completed_cases:
            for msg in case.messages:
                if msg.round_columns:
                    round_column_names.update(msg.round_columns.keys())
        
        # Sort round column names for consistent ordering
        round_column_names = sorted(round_column_names)
        
        # Write CSV with all messages and their case assignments
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            # Define CSV columns with case_number first, then round columns, then all original columns
            fieldnames = ['case_number'] + round_column_names + ['review', 'created_time', 'sender_id', 'real_sender_id', 'message', 'message_id', 'type', 'channel_url', 'file_url', 'sender_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write all messages with their case assignments
            total_rows = 0
            for case in self.completed_cases:
                for msg in case.messages:
                    # Base row data
                    row_data = {
                        'case_number': case.case_id,
                        'review': msg.review,
                        'created_time': msg.timestamp.isoformat(),
                        'sender_id': msg.sender_id,
                        'real_sender_id': msg.real_sender_id,
                        'message': msg.content,
                        'message_id': msg.message_id,
                        'type': msg.message_type,
                        'channel_url': msg.channel_url,
                        'file_url': msg.file_url,
                        'sender_type': msg.sender_type
                    }
                    
                    # Add round columns data
                    for round_col in round_column_names:
                        row_data[round_col] = msg.round_columns.get(round_col, '') if msg.round_columns else ''
                    
                    writer.writerow(row_data)
                    total_rows += 1
        
        print(f"✅ Exported {total_rows} messages with case assignments to {filepath}")
        
        # Print summary statistics
        total_stats = self.total_stats
        print(f"  📊 Total cases: {total_stats['total_cases']}")
        print(f"  📊 Total channels: {total_stats['total_channels']}")
        print(f"  📊 Average confidence: {total_stats['average_confidence']:.3f}")
    
    def export_segmentation_summary_md(self, filepath: Optional[str] = None):
        """Export comprehensive segmentation summary across all channels"""
        if filepath is None:
            output_config = get_output_config("channel")
            base_path = output_config.markdown_file.replace('.md', '_segmentation_summary.md')
            filepath = base_path
        
        print(f"📋 Exporting segmentation summary...")
        print(f"  📁 Output file: {filepath}")
        
        content = "# Case Segmentation Summary Report - Channel Algorithm\n\n"
        
        # Use cached overall statistics
        total_stats = self.total_stats
        
        content += f"**Algorithm:** Channel Full Conversation\n"
        content += f"**Total Cases Found:** {total_stats['total_cases']}\n"
        content += f"**Total Channels Processed:** {total_stats['total_channels']}\n"
        content += f"**Average Confidence:** {total_stats['average_confidence']:.3f}\n"
        content += f"**Truncated Channels:** {total_stats['truncated_channels']}\n"
        content += f"**Truncated Cases:** {total_stats['truncated_cases']}\n\n"
        
        # Use cached confidence distribution
        confidence_dist = total_stats['confidence_distribution']
        total_cases = total_stats['total_cases']
        
        content += "## Confidence Distribution\n\n"
        content += f"- **High Confidence (≥0.8):** {confidence_dist['high_confidence']} cases ({confidence_dist['high_confidence']/total_cases*100:.1f}%)\n"
        content += f"- **Medium Confidence (0.5-0.8):** {confidence_dist['medium_confidence']} cases ({confidence_dist['medium_confidence']/total_cases*100:.1f}%)\n"
        content += f"- **Low Confidence (<0.5):** {confidence_dist['low_confidence']} cases ({confidence_dist['low_confidence']/total_cases*100:.1f}%)\n\n"
        
        content += "## Truncation Analysis\n\n"
        truncated_cases = total_stats['truncated_cases']
        truncated_channels = total_stats['truncated_channels']
        content += f"- **Truncated Channels:** {truncated_channels} channels\n"
        content += f"- **Truncated Cases:** {truncated_cases} cases ({truncated_cases/total_cases*100:.1f}%)\n"
        if truncated_cases > 0:
            content += f"- **Note:** Truncated cases may be incomplete due to token limit constraints\n"
        content += "\n"
        
        # All segmentations table
        content += "## All Case Segmentations\n\n"
        content += "**Legend:** ⚠️ = Truncated\n\n"
        content += "| Case ID | Channel | Start Time | End Time | Duration | Confidence | Truncated | Summary |\n"
        content += "|---------|---------|------------|----------|----------|------------|-----------|----------|\n"
        
        # Use cached sorted cases
        for case in self.sorted_cases:
            short_channel = case.channel_url.split('_')[-1][:15] + "..."
            start_time = case.start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_time = case.end_time.strftime('%Y-%m-%d %H:%M:%S')
            duration = f"{case.duration_minutes:.1f}m"
            confidence = f"{case.confidence:.3f}"
            truncated_indicator = "⚠️" if case.truncated else "-"
            summary = case.summary[:40] + "..." if len(case.summary) > 40 else case.summary
            
            content += f"| {case.case_id} | {short_channel} | {start_time} | {end_time} | {duration} | {confidence} | {truncated_indicator} | {summary} |\n"
        
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
            was_truncated = self.channel_stats[channel_url].was_truncated
            
            content += f"### Channel: {short_channel}...\n\n"
            content += f"- **Cases:** {len(cases)}\n"
            content += f"- **Average Confidence:** {avg_conf:.3f}\n"
            content += f"- **Total Duration:** {total_duration:.1f} minutes\n"
            content += f"- **Was Truncated:** {'Yes' if was_truncated else 'No'}\n"
            content += f"- **Full URL:** `{channel_url}`\n\n"
            
            for case in sorted(cases, key=lambda x: x.start_time):
                time_span = f"{case.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {case.end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                truncated_note = " [TRUNCATED]" if case.truncated else ""
                content += f"  - **{case.case_id}** ({time_span}, {case.duration_minutes:.1f}m, conf: {case.confidence:.3f}){truncated_note}: {case.summary}\n"
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
        
        if truncated_cases > 0:
            content += f"⚠️ **Truncation Warning:** {truncated_cases} cases were truncated. Consider increasing token limits or splitting long conversations.\n\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Exported segmentation summary to {filepath}")


def main():
    """Main execution function"""
    print("=== Configuration Status ===")
    config = get_config()
    status = config.validate_config()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    parser = ChannelCaseParser(
        llm_provider=None  # Use primary_provider and model_type from config.json
    )
    
    # Load data grouped by channel
    channels = parser.load_csv('assets/preprocessed_support_msg.csv')
    
    # Process all channels
    print(f"\n🚀 Starting case processing...")
    cases = parser.process_all_channels(channels)
    
    # Export results (uses config file paths)
    print(f"\n📤 Exporting results...")
    parser.export_json()
    parser.export_cases_csv()
    parser.export_segmentation_summary_md()
    
    print(f"\n=== CHANNEL ALGORITHM COMPLETE ===")
    print(f"Total cases: {len(cases)}")
    print(f"Total channels: {len(channels)}")


if __name__ == "__main__":
    main()