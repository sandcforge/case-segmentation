#!/usr/bin/env python3
"""
LLM Provider Abstraction Layer

Supports multiple LLM providers (OpenAI, Anthropic Claude, etc.)
with unified interface for case boundary analysis.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional imports - only load what's available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available - install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic not available - install with: pip install anthropic")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini not available - install with: pip install google-generativeai")


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    processing_time: float
    # Enhanced error/debug information
    raw_response: Optional[Any] = None
    error_details: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def analyze_case_boundaries(self, prompt: str) -> LLMResponse:
        """Analyze conversation for case boundaries"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", output_max_token_size: int = 4096):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.output_max_token_size = output_max_token_size
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key and OPENAI_AVAILABLE else None
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            # Fallback to gpt-4 tokenizer
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.client is not None
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def analyze_case_boundaries(self, prompt: str) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
        
        start_time = time.time()
        input_tokens = self.count_tokens(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=self.output_max_token_size
            )
            
            content = response.choices[0].message.content
            output_tokens = self.count_tokens(content)
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                provider="openai",
                processing_time=processing_time
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", output_max_token_size: int = 4096):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.output_max_token_size = output_max_token_size
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key and ANTHROPIC_AVAILABLE else None
        
        # Claude uses different tokenization, approximate with OpenAI tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = None
    
    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def analyze_case_boundaries(self, prompt: str) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available")
        
        start_time = time.time()
        input_tokens = self.count_tokens(prompt)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.output_max_token_size,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            output_tokens = self.count_tokens(content)
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                provider="anthropic",
                processing_time=processing_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", output_max_token_size: int = 8192):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = model
        self.output_max_token_size = output_max_token_size
        
        if self.api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            self.client = None
        
        # Gemini uses different tokenization, approximate with OpenAI tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = None
    
    def is_available(self) -> bool:
        return GEMINI_AVAILABLE and self.client is not None
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def analyze_case_boundaries(self, prompt: str) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Gemini provider not available")
        
        start_time = time.time()
        input_tokens = self.count_tokens(prompt)
        response = None
        
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=self.output_max_token_size
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Try to extract content, but capture response details even if it fails
            try:
                content = response.text
                output_tokens = self.count_tokens(content)
                processing_time = time.time() - start_time
                
                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                    provider="gemini",
                    processing_time=processing_time,
                    raw_response=response,
                    finish_reason=getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
                )
                
            except Exception as content_error:
                # Content extraction failed, but we have the response object
                processing_time = time.time() - start_time
                
                # Extract what we can from the response
                error_details = {
                    "content_extraction_error": str(content_error),
                    "candidates_count": len(response.candidates) if response.candidates else 0,
                    "prompt_feedback": getattr(response, 'prompt_feedback', None),
                }
                
                # Try to get finish_reason and safety ratings
                if response.candidates:
                    candidate = response.candidates[0]
                    error_details.update({
                        "finish_reason": getattr(candidate, 'finish_reason', None),
                        "safety_ratings": getattr(candidate, 'safety_ratings', None),
                        "citation_metadata": getattr(candidate, 'citation_metadata', None)
                    })
                
                # Create a partial response with error details
                partial_response = LLMResponse(
                    content="[CONTENT_EXTRACTION_FAILED]",
                    input_tokens=input_tokens,
                    output_tokens=0,
                    model=self.model,
                    provider="gemini",
                    processing_time=processing_time,
                    raw_response=response,
                    error_details=error_details,
                    finish_reason=str(error_details.get("finish_reason", "UNKNOWN"))
                )
                
                # Raise exception with enhanced details, but preserve response data
                raise RuntimeError(f"Gemini content extraction failed: {content_error}. Response details available.") from content_error
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Create error details even for complete API failures
            error_details = {
                "api_call_error": str(e),
                "error_type": type(e).__name__,
                "response_available": response is not None
            }
            
            if response:
                # We got a response but something went wrong
                error_details.update({
                    "candidates_count": len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 0,
                    "prompt_feedback": str(getattr(response, 'prompt_feedback', None))
                })
                
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    error_details.update({
                        "finish_reason": str(getattr(candidate, 'finish_reason', None)),
                        "safety_ratings": str(getattr(candidate, 'safety_ratings', None))
                    })
            
            # Store error response for debugging
            error_response = LLMResponse(
                content="[API_CALL_FAILED]",
                input_tokens=input_tokens,
                output_tokens=0,
                model=self.model,
                provider="gemini",
                processing_time=processing_time,
                raw_response=response,
                error_details=error_details,
                finish_reason=str(error_details.get("finish_reason", "API_FAILURE"))
            )
            
            # Attach error_response to exception for debug access
            enhanced_error = RuntimeError(f"Gemini API call failed: {e}")
            enhanced_error.error_response = error_response
            raise enhanced_error


class LocalLLMProvider(LLMProvider):
    """Local LLM provider (placeholder for Ollama, etc.)"""
    
    def __init__(self, model: str = "llama2", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint
        # Could integrate with Ollama, LlamaCpp, etc.
    
    def is_available(self) -> bool:
        # Check if local endpoint is available
        return False  # Placeholder
    
    def count_tokens(self, text: str) -> int:
        # Rough approximation
        return len(text) // 4
    
    def analyze_case_boundaries(self, prompt: str) -> LLMResponse:
        raise NotImplementedError("Local LLM provider not implemented yet")


class LLMManager:
    """Manager class to handle LLM providers"""
    
    def __init__(self, primary_provider: str = "openai"):
        self.primary_provider = primary_provider
        self.providers = {}
        
        # Initialize available providers
        self._init_providers()
        
        # Set primary provider
        if primary_provider not in self.providers:
            raise ValueError(f"Primary provider '{primary_provider}' not available")
        
        self.current_provider = self.providers[primary_provider]
    
    def _init_providers(self):
        """Initialize all available providers"""
        
        # OpenAI
        try:
            openai_model = os.getenv('OPENAI_MODEL', "gpt-4o-mini")
            openai_token_size = int(os.getenv('OPENAI_OUTPUT_MAX_TOKEN_SIZE', "4096"))
            openai_provider = OpenAIProvider(model=openai_model, output_max_token_size=openai_token_size)
            if openai_provider.is_available():
                self.providers["openai"] = openai_provider
                print("✓ OpenAI provider initialized")
        except Exception as e:
            print(f"✗ OpenAI provider failed: {e}")
        
        # Anthropic
        try:
            anthropic_model = os.getenv('ANTHROPIC_MODEL', "claude-3-5-sonnet-20241022")
            anthropic_token_size = int(os.getenv('ANTHROPIC_OUTPUT_MAX_TOKEN_SIZE', "4096"))
            anthropic_provider = AnthropicProvider(model=anthropic_model, output_max_token_size=anthropic_token_size)
            if anthropic_provider.is_available():
                self.providers["anthropic"] = anthropic_provider
                print("✓ Anthropic provider initialized")
        except Exception as e:
            print(f"✗ Anthropic provider failed: {e}")
        
        # Gemini
        try:
            gemini_model = os.getenv('GEMINI_MODEL', "gemini-2.5-flash")
            gemini_token_size = int(os.getenv('GEMINI_OUTPUT_MAX_TOKEN_SIZE', "8192"))
            gemini_provider = GeminiProvider(model=gemini_model, output_max_token_size=gemini_token_size)
            if gemini_provider.is_available():
                self.providers["gemini"] = gemini_provider
                print("✓ Gemini provider initialized")
        except Exception as e:
            print(f"✗ Gemini provider failed: {e}")
        
        # Local LLM (placeholder)
        # self.providers["local"] = LocalLLMProvider()
        
        if not self.providers:
            raise RuntimeError("No LLM providers available")
    
    def switch_provider(self, provider_name: str):
        """Switch to a different provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        self.current_provider = self.providers[provider_name]
        self.primary_provider = provider_name
        print(f"Switched to {provider_name} provider")
    
    def list_available_providers(self) -> Dict[str, str]:
        """List all available providers with their models"""
        result = {}
        for name, provider in self.providers.items():
            if hasattr(provider, 'model'):
                result[name] = provider.model
            else:
                result[name] = "Unknown model"
        return result
    
    def analyze_case_boundaries(self, prompt: str) -> LLMResponse:
        """Analyze case boundaries using current provider"""
        return self.current_provider.analyze_case_boundaries(prompt)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using current provider"""
        return self.current_provider.count_tokens(text)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider"""
        return {
            "provider": self.primary_provider,
            "model": getattr(self.current_provider, 'model', 'unknown'),
            "available_providers": list(self.providers.keys())
        }


def create_llm_manager(provider: str = "openai", 
                      openai_model: str = "gpt-4o-mini",
                      anthropic_model: str = "claude-3-5-sonnet-20241022",
                      gemini_model: str = "gemini-2.5-flash",
                      output_max_token_size: int = 4096) -> LLMManager:
    """Factory function to create LLM manager with custom models and token limits"""
    
    # Set custom models and token limits before creating manager
    if provider == "openai" and OPENAI_AVAILABLE:
        os.environ.setdefault('OPENAI_MODEL', openai_model)
        os.environ.setdefault('OPENAI_OUTPUT_MAX_TOKEN_SIZE', str(output_max_token_size))
    elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
        os.environ.setdefault('ANTHROPIC_MODEL', anthropic_model)
        os.environ.setdefault('ANTHROPIC_OUTPUT_MAX_TOKEN_SIZE', str(output_max_token_size))
    elif provider == "gemini" and GEMINI_AVAILABLE:
        os.environ.setdefault('GEMINI_MODEL', gemini_model)
        os.environ.setdefault('GEMINI_OUTPUT_MAX_TOKEN_SIZE', str(output_max_token_size))
    
    return LLMManager(primary_provider=provider)


# Usage example
if __name__ == "__main__":
    # Test the LLM manager
    try:
        manager = create_llm_manager(provider="openai")
        print("Available providers:", manager.list_available_providers())
        
        # Test prompt
        test_prompt = """
        Analyze this conversation for case boundaries:
        
        Message 1: User: "Hi, I have a problem with my order"
        Message 2: Support: "I'd be happy to help! What's your order number?"
        Message 3: User: "It's #12345"
        Message 4: Support: "Thanks! I see the issue and will fix it now"
        Message 5: User: "Thank you so much!"
        
        Return JSON with case boundaries.
        """
        
        response = manager.analyze_case_boundaries(test_prompt)
        print(f"Response from {response.provider}: {response.content[:100]}...")
        print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
        
    except Exception as e:
        print(f"Error: {e}")