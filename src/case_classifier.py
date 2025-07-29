#!/usr/bin/env python3
"""
Case Classifier with LLM-Based Hierarchical Taxonomy

This module provides automated classification of customer service cases using LLM
to categorize them into a specific hierarchical taxonomy with primary and secondary categories.
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm_provider import create_llm_manager
from config_manager import get_llm_config


@dataclass
class CaseClassification:
    """Classification result for a single case"""
    case_id: str
    category: str  # Combined primary_secondary format
    confidence: float
    reasoning: str
    classified_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "case_id": self.case_id,
            "category": self.category,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "classified_at": self.classified_at.isoformat()
        }


class CaseClassifier:
    """
    LLM-based case classifier using hierarchical taxonomy.
    
    Classifies customer service cases into predefined categories using
    LLM analysis of case content and context.
    """
    
    def __init__(self, llm_provider: Optional[str] = None, llm_model_type: Optional[str] = None):
        """Initialize classifier with LLM configuration"""
        self.taxonomy = self._load_classification_taxonomy()
        self.llm_manager = None
        
        # Initialize LLM manager using same pattern as case parser
        try:
            llm_config = get_llm_config(llm_provider, llm_model_type)
            
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
            
            print(f"CaseClassifier initialized with provider: {self.llm_manager.get_provider_info()}")
            
        except Exception as e:
            print(f"Warning: LLM initialization failed for classifier: {e}")
            self.llm_manager = None
    
    def classify_case(self, case: Any) -> Optional[CaseClassification]:
        """
        Classify a single case using LLM analysis.
        
        Args:
            case: Case object from channel_segmenter.py
            
        Returns:
            CaseClassification object or None if classification fails
        """
        if not self.llm_manager:
            print(f"  ⚠️ No LLM available for classification of {case.case_id}")
            return None
        
        try:
            # Create classification prompt
            prompt = self._create_classification_prompt(case)
            
            # Get LLM response
            response = self.llm_manager.analyze_case_boundaries(prompt)
            
            # Log classification interaction
            self._log_classification_interaction(case.case_id, prompt, response, success=True)
            
            # Parse classification result
            classification_result = self._parse_classification_response(response.content, case.case_id)
            
            if classification_result:
                print(f"  🏷️ Classified {case.case_id}: {classification_result.category} (conf: {classification_result.confidence:.3f})")
                return classification_result
            else:
                print(f"  ⚠️ Failed to parse classification for {case.case_id}")
                return None
                
        except Exception as e:
            print(f"  ❌ Classification failed for {case.case_id}: {e}")
            self._log_classification_interaction(case.case_id, "", e, success=False)
            return None
    
    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify cases in a DataFrame and add classification columns.
        
        Args:
            df: DataFrame with Case Number column (from case parser)
            
        Returns:
            DataFrame with classification columns added
        """
        if 'Case Number' not in df.columns:
            raise ValueError("DataFrame must contain 'Case Number' column from case parser")
        
        print(f"🏷️ Starting classification of DataFrame with {len(df)} rows...")
        
        # Get unique cases to classify
        unique_cases = df[df['Case Number'] != '']['Case Number'].unique()
        print(f"  📊 Found {len(unique_cases)} unique cases to classify")
        
        # Create mock case objects for classification
        case_objects = []
        for case_id in unique_cases:
            case_df = df[df['Case Number'] == case_id].copy()
            if len(case_df) == 0:
                continue
                
            # Create a mock case object from DataFrame rows
            mock_case = self._create_mock_case_from_dataframe(case_df, case_id)
            case_objects.append(mock_case)
        
        # Classify all cases
        print(f"🏷️ Starting classification of {len(case_objects)} cases...")
        
        classifications = []
        for i, case in enumerate(case_objects):
            if i % 10 == 0:
                print(f"  📊 Classified {i}/{len(case_objects)} cases...")
            
            classification = self.classify_case(case)
            if classification:
                classifications.append(classification)
        
        print(f"✅ Classification complete: {len(classifications)}/{len(case_objects)} cases classified successfully")
        
        # Create classification mapping
        classification_map = {c.case_id: c for c in classifications}
        
        # Add classification columns to DataFrame
        output_df = df.copy()
        output_df['Category'] = ''
        output_df['classification_confidence'] = 0.0
        output_df['classification_reasoning'] = ''
        output_df['classified_at'] = pd.NaT
        
        # Apply classifications to DataFrame
        for i, row in output_df.iterrows():
            case_id = row['Case Number']
            if case_id in classification_map:
                classification = classification_map[case_id]
                output_df.at[i, 'Category'] = classification.category
                output_df.at[i, 'classification_confidence'] = classification.confidence
                output_df.at[i, 'classification_reasoning'] = classification.reasoning
                output_df.at[i, 'classified_at'] = classification.classified_at
        
        print(f"✅ Classification complete: {len(classifications)} cases classified")
        return output_df
    
    def _create_mock_case_from_dataframe(self, case_df: pd.DataFrame, case_id: str):
        """Create a mock case object from DataFrame rows for classification"""
        # Sort by timestamp
        case_df = case_df.sort_values('Created Time')
        
        # Create mock messages
        mock_messages = []
        for _, row in case_df.iterrows():
            mock_message = type('MockMessage', (), {
                'content': str(row['Message']),
                'timestamp': pd.to_datetime(row['Created Time']),
                'sender_type': str(row['Sender Type']),
                'sender_id': str(row['Sender ID'])
            })()
            mock_messages.append(mock_message)
        
        # Create mock case object
        summary = str(case_df.iloc[0]['case_summary']) if 'case_summary' in case_df.columns else 'No summary available'
        duration = float(case_df.iloc[0]['case_duration_minutes']) if 'case_duration_minutes' in case_df.columns else 0.0
        
        mock_case = type('MockCase', (), {
            'case_id': case_id,
            'messages': mock_messages,
            'summary': summary,
            'duration_minutes': duration
        })()
        
        return mock_case

    
    def generate_classification_report(self, classifications: List[CaseClassification]) -> Dict[str, Any]:
        """Generate comprehensive classification analytics report"""
        if not classifications:
            return {"error": "No classifications provided"}
        
        total_cases = len(classifications)
        
        # Category distributions
        category_dist = {}
        primary_dist = {}
        
        for classification in classifications:
            # Combined category distribution
            category = classification.category
            category_dist[category] = category_dist.get(category, 0) + 1
            
            # Primary category distribution (extract from combined format)
            primary = category.split('_')[0] if '_' in category else category
            primary_dist[primary] = primary_dist.get(primary, 0) + 1
        
        # Confidence analysis
        confidences = [c.confidence for c in classifications]
        avg_confidence = sum(confidences) / len(confidences)
        high_confidence = len([c for c in confidences if c >= 0.8])
        medium_confidence = len([c for c in confidences if 0.5 <= c < 0.8])
        low_confidence = len([c for c in confidences if c < 0.5])
        
        return {
            "summary": {
                "total_classified_cases": total_cases,
                "classification_timestamp": datetime.now().isoformat(),
                "average_confidence": round(avg_confidence, 3),
                "confidence_distribution": {
                    "high_confidence": f"{high_confidence} ({high_confidence/total_cases*100:.1f}%)",
                    "medium_confidence": f"{medium_confidence} ({medium_confidence/total_cases*100:.1f}%)",
                    "low_confidence": f"{low_confidence} ({low_confidence/total_cases*100:.1f}%)"
                }
            },
            "category_distributions": {
                "categories": category_dist,
                "primary_categories": primary_dist
            },
            "insights": {
                "most_common_category": max(category_dist.items(), key=lambda x: x[1]) if category_dist else None,
                "most_common_primary": max(primary_dist.items(), key=lambda x: x[1]) if primary_dist else None,
                "unique_categories": len(category_dist),
                "unique_primary_categories": len(primary_dist)
            }
        }
    
    def export_classification_summary_md(self, classifications: List[CaseClassification], filepath: Optional[str] = None):
        """Export comprehensive classification summary to markdown file"""
        if not classifications:
            print("⚠️ No classifications provided for export")
            return
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"output/case_classification_summary_{timestamp}.md"
        
        print(f"📋 Exporting classification summary...")
        print(f"  📁 Output file: {filepath}")
        
        # Generate report data
        report = self.generate_classification_report(classifications)
        
        content = "# Case Classification Summary Report\n\n"
        
        # Header information
        content += f"**Classification Algorithm:** LLM-Based Hierarchical Taxonomy\n"
        content += f"**Total Cases Classified:** {report['summary']['total_classified_cases']}\n"
        content += f"**Classification Timestamp:** {report['summary']['classification_timestamp']}\n"
        content += f"**Average Confidence:** {report['summary']['average_confidence']:.3f}\n\n"
        
        # Confidence distribution
        content += "## Confidence Distribution\n\n"
        conf_dist = report['summary']['confidence_distribution']
        content += f"- **High Confidence (≥0.8):** {conf_dist['high_confidence']}\n"
        content += f"- **Medium Confidence (0.5-0.8):** {conf_dist['medium_confidence']}\n"
        content += f"- **Low Confidence (<0.5):** {conf_dist['low_confidence']}\n\n"
        
        # Primary category distribution
        content += "## Primary Category Distribution\n\n"
        primary_dist = report['category_distributions']['primary_categories']
        total_cases = report['summary']['total_classified_cases']
        
        content += "| Primary Category | Cases | Percentage |\n"
        content += "|------------------|-------|------------|\n"
        for category, count in sorted(primary_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_cases * 100
            content += f"| {category} | {count} | {percentage:.1f}% |\n"
        content += "\n"
        
        # Secondary category distribution (top 15)
        content += "## Secondary Category Distribution (Top 15)\n\n"
        category_dist = report['category_distributions']['categories']
        
        content += "| Combined Category | Cases | Percentage |\n"
        content += "|-------------------|-------|------------|\n"
        for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:15]:
            percentage = count / total_cases * 100
            # Format category name for display
            display_category = category.replace('_', ' - ').replace('  ', ' ')
            content += f"| {display_category} | {count} | {percentage:.1f}% |\n"
        
        if len(category_dist) > 15:
            remaining = len(category_dist) - 15
            content += f"| ... and {remaining} more categories | ... | ... |\n"
        content += "\n"
        
        # Key insights
        content += "## Key Insights\n\n"
        insights = report['insights']
        
        if insights['most_common_category']:
            most_common_cat, most_common_count = insights['most_common_category']
            display_cat = most_common_cat.replace('_', ' - ').replace('  ', ' ')
            content += f"- **Most Common Category:** {display_cat} ({most_common_count} cases)\n"
        
        if insights['most_common_primary']:
            most_common_pri, most_common_pri_count = insights['most_common_primary']
            content += f"- **Most Common Primary:** {most_common_pri} ({most_common_pri_count} cases)\n"
        
        content += f"- **Unique Categories:** {insights['unique_categories']}\n"
        content += f"- **Unique Primary Categories:** {insights['unique_primary_categories']}\n\n"
        
        # Detailed case classifications
        content += "## All Case Classifications\n\n"
        content += "**Legend:** 🟢 = High Confidence, 🟡 = Medium Confidence, 🔴 = Low Confidence\n\n"
        content += "| Case ID | Category | Confidence | Confidence Level | Reasoning | Classified At |\n"
        content += "|---------|----------|------------|------------------|-----------|---------------|\n"
        
        # Sort by confidence descending, then by case_id
        sorted_classifications = sorted(classifications, key=lambda x: (-x.confidence, x.case_id))
        
        for classification in sorted_classifications:
            # Confidence level indicator
            if classification.confidence >= 0.8:
                conf_indicator = "🟢"
            elif classification.confidence >= 0.5:
                conf_indicator = "🟡"
            else:
                conf_indicator = "🔴"
            
            # Format category name for display
            display_category = classification.category.replace('_', ' - ').replace('  ', ' ')
            
            # Truncate reasoning if too long
            reasoning = classification.reasoning[:50] + "..." if len(classification.reasoning) > 50 else classification.reasoning
            
            # Format timestamp
            classified_time = classification.classified_at.strftime('%Y-%m-%d %H:%M:%S')
            
            content += f"| {classification.case_id} | {display_category} | {classification.confidence:.3f} | {conf_indicator} | {reasoning} | {classified_time} |\n"
        
        content += "\n"
        
        # Quality assessment
        content += "## Quality Assessment\n\n"
        avg_confidence = report['summary']['average_confidence']
        high_conf_count = len([c for c in classifications if c.confidence >= 0.8])
        low_conf_count = len([c for c in classifications if c.confidence < 0.5])
        
        if avg_confidence >= 0.7:
            content += "✅ **Good Classification Quality** - High average confidence suggests reliable categorization.\n\n"
        elif avg_confidence >= 0.5:
            content += "⚠️ **Moderate Classification Quality** - Consider reviewing low confidence cases manually.\n\n"
        else:
            content += "❌ **Poor Classification Quality** - Many low confidence cases suggest model tuning needed.\n\n"
        
        if low_conf_count > total_cases * 0.3:
            content += f"⚠️ **Warning:** {low_conf_count} cases ({low_conf_count/total_cases*100:.1f}%) have low confidence. Manual review recommended.\n\n"
        
        if high_conf_count > total_cases * 0.7:
            content += f"✅ **Excellent:** {high_conf_count} cases ({high_conf_count/total_cases*100:.1f}%) have high confidence.\n\n"
        
        # Recommendations
        content += "## Recommendations\n\n"
        
        if low_conf_count > 0:
            content += f"1. **Review Low Confidence Cases:** {low_conf_count} cases need manual review\n"
        
        # Check for category imbalance
        max_category_count = max(primary_dist.values()) if primary_dist else 0
        if max_category_count > total_cases * 0.5:
            content += "2. **Category Imbalance:** Consider if category distribution reflects actual case types\n"
        
        if avg_confidence < 0.6:
            content += "3. **Model Improvement:** Consider refining classification prompts or taxonomy\n"
        
        content += "4. **Regular Review:** Periodically validate classifications against business requirements\n\n"
        
        # Export the markdown file
        try:
            import os
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Exported classification summary to {filepath}")
            print(f"  📊 Total cases: {total_cases}")
            print(f"  📊 Average confidence: {avg_confidence:.3f}")
            print(f"  📊 Unique categories: {insights['unique_categories']}")
            
        except Exception as e:
            print(f"❌ Failed to export classification summary: {e}")
    
    def _load_classification_taxonomy(self) -> Dict[str, List[str]]:
        """Load the hierarchical classification taxonomy"""
        return {
            "Order": [
                "Order - Status",
                "Order - Cancel",
                "Order - Update",
                "Order - Failure",
                "Order - Missing Coupon",
                "Refund - Status",
                "Refund - Request",
                "Refund - Full",
                "Refund - Partial",
                "Giveaway",
                "Other"
            ],
            "Shipment": [
                "Shipment - Status",
                "Shipment - Delay",
                "Shipment - Lost",
                "Shipment - Wrong Address",
                "Shipment - Reshipping",
                "Shipment - Local Pickup",
                "Shipment - Carrier Claim",
                "Update - Address",
                "Update - Carrier",
                "Update - Shipping Label",
                "Item - Damaged",
                "Item - Missing",
                "Item - Wrong",
                "Tracking - Status",
                "Tracking - Invalid",
                "Tracking - Status Not Updating",
                "Tracking - False Completion",
                "Other"
            ],
            "Payment": [
                "Payment - Status",
                "Payment - Verification",
                "Payment - Failure",
                "Payment - Dispute",
                "Payment - Chargeback",
                "Update - Method",
                "Withdrawal - Status",
                "Withdrawal - Delay",
                "Withdrawal - Method",
                "Coupon - Redemption",
                "Coupon - Redemption Failure",
                "Pay by Credit",
                "Other"
            ],
            "Tax and Fee": [
                "Sales Fee",
                "Sales Tax",
                "Invoice",
                "Tax Information",
                "Form 1099",
                "Other"
            ],
            "User": [
                "Update - Username",
                "Update - Email",
                "Update - Password",
                "Account - Delete",
                "Account - Recover",
                "Other"
            ],
            "Seller": [
                "Application - Request",
                "Application - Rejection",
                "Application - Additional Materials",
                "Application - Trust Review",
                "Foundation Plan - Enroll",
                "Foundation Plan - Update",
                "Foundation Plan - Cancel",
                "Live Quota",
                "Other"
            ],
            "App Functionality": [
                "Login",
                "Logout",
                "Settings",
                "Live",
                "Live Auction",
                "Purge",
                "Marketplace",
                "Long-Form Auction",
                "Bug",
                "OBS Connection",
                "Permission",
                "Content Moderation",
                "System Update",
                "Other"
            ],
            "Referral and Promotion": [
                "Referral Bonus - Information",
                "Referral Bonus - Not Received",
                "Credit",
                "Gift",
                "Other"
            ],
            "Other": [
                "Issue - Resolved",
                "Issue - Reopened",
                "Feedback",
                "Complaint",
                "Copyright",
                "Courtesy",
                "Other"
            ]
        }
    
    def _create_classification_prompt(self, case: Any) -> str:
        """Create structured prompt for case classification"""
        
        # Extract case information
        case_summary = getattr(case, 'summary', 'No summary available')
        case_messages = self._format_case_messages(case)
        
        # Format taxonomy for prompt
        taxonomy_text = self._format_taxonomy_for_prompt()
        
        prompt = f"""
# CUSTOMER SERVICE CASE CLASSIFICATION

## TASK
Classify the following customer service case into the appropriate category from the provided taxonomy.

## CASE INFORMATION
**Case ID:** {case.case_id}
**Case Summary:** {case_summary}
**Duration:** {getattr(case, 'duration_minutes', 0):.1f} minutes
**Message Count:** {len(case.messages)}

## CASE MESSAGES
{case_messages}

## CLASSIFICATION TAXONOMY
{taxonomy_text}

## CLASSIFICATION INSTRUCTIONS

1. **Analyze the case content** - Review the case summary and message content to understand the customer's primary issue or request.

2. **Identify the primary category** - Determine which of the 9 primary categories best fits the case:
   - Order, Shipment, Payment, Tax and Fee, User, Seller, App Functionality, Referral and Promotion, Other

3. **Select the secondary category** - Choose the most specific subcategory within the primary category that matches the case.

4. **Assess confidence** - Rate your confidence in this classification from 0.0 to 1.0:
   - 0.9-1.0: Very confident, clear case type
   - 0.7-0.8: Confident, good indicators
   - 0.5-0.6: Moderate confidence, some ambiguity
   - 0.0-0.4: Low confidence, unclear case type

5. **Provide reasoning** - Briefly explain why you chose this classification.

## OUTPUT FORMAT

Return your classification in this exact JSON format:

<output>
{{
    "primary_category": "Order",
    "secondary_category": "Order - Cancel",
    "confidence": 0.9,
    "reasoning": "Customer explicitly requested order cancellation and customer service processed the cancellation request."
}}
</output>

## CRITICAL REQUIREMENTS

- **Use exact category names** from the taxonomy provided above
- **Choose only one primary and one secondary category** 
- **Provide confidence score** between 0.0 and 1.0
- **Include brief reasoning** for your classification choice
- **Return only the JSON object** in the output tags, no additional text
"""
        
        return prompt
    
    def _format_case_messages(self, case: Any) -> str:
        """Format case messages for the classification prompt"""
        messages = []
        
        for i, msg in enumerate(case.messages[:10]):  # Limit to first 10 messages for prompt size
            sender_type = getattr(msg, 'sender_type', 'unknown')
            sender_id = getattr(msg, 'sender_id', 'unknown')
            content = msg.content.strip()
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            messages.append(f"**Message {i+1}** [{sender_type}] ({sender_id}) ({timestamp}): {content}")
        
        if len(case.messages) > 10:
            messages.append(f"... and {len(case.messages) - 10} more messages")
        
        return "\n\n".join(messages)
    
    def _format_taxonomy_for_prompt(self) -> str:
        """Format the classification taxonomy for the prompt"""
        taxonomy_lines = []
        
        for primary_category, secondary_categories in self.taxonomy.items():
            taxonomy_lines.append(f"**{primary_category}:**")
            for secondary in secondary_categories:
                taxonomy_lines.append(f"  - {secondary}")
            taxonomy_lines.append("")  # Empty line between categories
        
        return "\n".join(taxonomy_lines)
    
    def _parse_classification_response(self, response_content: str, case_id: str) -> Optional[CaseClassification]:
        """Parse LLM response to extract classification"""
        try:
            # Extract JSON from output tags
            extracted_json = self._extract_json_from_response(response_content)
            
            if not extracted_json:
                print(f"    ⚠️ No JSON found in classification response for {case_id}")
                return None
            
            result = json.loads(extracted_json)
            
            # Validate required fields
            required_fields = ["primary_category", "secondary_category", "confidence", "reasoning"]
            if not all(field in result for field in required_fields):
                print(f"    ⚠️ Missing required fields in classification for {case_id}")
                return None
            
            # Validate categories against taxonomy
            primary = result["primary_category"]
            secondary = result["secondary_category"]
            
            if primary not in self.taxonomy:
                print(f"    ⚠️ Invalid primary category '{primary}' for {case_id}")
                return None
            
            if secondary not in self.taxonomy[primary]:
                print(f"    ⚠️ Invalid secondary category '{secondary}' for primary '{primary}' in {case_id}")
                return None
            
            # Validate confidence
            confidence = float(result["confidence"])
            if not 0.0 <= confidence <= 1.0:
                print(f"    ⚠️ Invalid confidence {confidence} for {case_id}, clamping to valid range")
                confidence = max(0.0, min(1.0, confidence))
            
            # Create combined category in Primary_Secondary format
            combined_category = f"{primary}_{secondary.replace(' - ', '_').replace(' ', '_')}"
            
            return CaseClassification(
                case_id=case_id,
                category=combined_category,
                confidence=confidence,
                reasoning=result["reasoning"],
                classified_at=datetime.now()
            )
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON parsing failed for {case_id}: {e}")
            return None
        except Exception as e:
            print(f"    ⚠️ Classification parsing error for {case_id}: {e}")
            return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response with output tags"""
        import re
        
        # Look for content in <output></output> tags
        output_pattern = re.compile(r'<output>\s*(\{.*?\})\s*</output>', re.DOTALL | re.IGNORECASE)
        output_match = output_pattern.search(response_text)
        
        if output_match:
            return output_match.group(1).strip()
        
        # Fallback: look for JSON objects
        json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
        json_matches = json_pattern.findall(response_text)
        
        for match in json_matches:
            try:
                # Try to parse to validate it's valid JSON
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _log_classification_interaction(self, case_id: str, prompt: str, response, success: bool = True) -> None:
        """Log classification interactions for debugging"""
        try:
            os.makedirs("debug_output", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            status = "success" if success else "error"
            debug_filename = f"debug_output/classification_{status}_{case_id}_{timestamp}.txt"
            
            with open(debug_filename, 'w', encoding='utf-8') as debug_file:
                debug_file.write(f"=== CASE CLASSIFICATION LOG ===\n")
                debug_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                debug_file.write(f"Case ID: {case_id}\n")
                debug_file.write(f"Status: {status.upper()}\n")
                
                if hasattr(response, 'model'):
                    debug_file.write(f"Model: {response.model}\n")
                    debug_file.write(f"Provider: {response.provider}\n")
                    debug_file.write(f"Input Tokens: {response.input_tokens}\n")
                    debug_file.write(f"Output Tokens: {response.output_tokens}\n")
                
                if not success:
                    debug_file.write(f"\n=== ERROR ===\n")
                    debug_file.write(f"{str(response)}\n")
                
                debug_file.write(f"\n=== CLASSIFICATION PROMPT ===\n")
                debug_file.write(prompt)
                
                if hasattr(response, 'content'):
                    debug_file.write(f"\n\n=== LLM RESPONSE ===\n")
                    debug_file.write(response.content)
            
        except Exception as e:
            print(f"    ⚠️ Failed to log classification interaction for {case_id}: {e}")


def demo_dataframe_classification():
    """Demo function showing DataFrame-based classification"""
    from data_preprocessor import DataPreprocessor
    from channel_segmenter import ChannelSegmenter
    
    print("=== DataFrame Classification Demo ===")
    
    # Step 1: Preprocess data to DataFrame
    preprocessor = DataPreprocessor()
    df = preprocessor.process_to_dataframe('assets/support_msg.csv', mode='r3')
    
    # Step 2: Process with case parser
    parser = ChannelSegmenter()
    case_df = parser.process_dataframe(df)
    
    # Step 3: Classify cases
    classifier = CaseClassifier()
    classified_df = classifier.classify_dataframe(case_df)
    
    print(f"\n✅ Classification complete!")
    print(f"📊 Output shape: {classified_df.shape}")
    print(f"📊 Classification columns: {[col for col in classified_df.columns if 'classif' in col.lower() or col == 'Category']}")
    print(f"📊 Unique categories: {classified_df['Category'].unique()}")
    print(f"📊 Sample output:\n{classified_df[['Case Number', 'Category', 'classification_confidence']].drop_duplicates().head()}")
    
    return classified_df

def main():
    """Demo/test function for the classifier"""
    print("CaseClassifier initialized successfully")
    
    classifier = CaseClassifier()
    
    if classifier.llm_manager:
        print("✅ LLM manager initialized")
        print(f"Available taxonomy categories: {list(classifier.taxonomy.keys())}")
    else:
        print("❌ LLM manager initialization failed")


if __name__ == "__main__":
    main()