#!/usr/bin/env python3
"""
Configuration Manager

Handles loading configuration from JSON file and environment variables.
Provides easy access to LLM settings, parsing parameters, and output options.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    output_max_token_size: int


@dataclass
class ParsingConfig:
    batch_size: int
    max_queue_size: int
    
    # Enhanced algorithm specific
    attention_sink_size: Optional[int] = None
    coherence_threshold: Optional[float] = None
    similarity_model: Optional[str] = None
    
    # Channel algorithm specific
    max_context_tokens: Optional[int] = None
    reserve_tokens: Optional[int] = None


@dataclass
class OutputConfig:
    json_file: str
    markdown_file: str
    comparison_file: str
    include_full_conversations: bool
    max_message_length: int


class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found. Please create a valid config.json file.")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing configuration file '{self.config_file}': {e}. Please fix the JSON syntax.", e.doc, e.pos)
    
    
    def get_llm_config(self, provider: Optional[str] = None, 
                      model_type: str = "default") -> LLMConfig:
        """Get LLM configuration for specified provider"""
        
        # Use specified provider or required from config
        provider = provider or self.config["llm"]["primary_provider"]
        
        # Get provider config - must exist
        if provider not in self.config["llm"]["providers"]:
            raise KeyError(f"Provider '{provider}' not found in config.json. Available providers: {list(self.config['llm']['providers'].keys())}")
        
        provider_config = self.config["llm"]["providers"][provider]
        
        # Get model name - must exist
        if "models" not in provider_config:
            raise KeyError(f"No models configuration found for provider '{provider}' in config.json")
        
        models = provider_config["models"]
        if model_type not in models:
            if "default" not in models:
                raise KeyError(f"Model type '{model_type}' not found and no 'default' model specified for provider '{provider}' in config.json")
            model = models["default"]
        else:
            model = models[model_type]
        
        # All required fields must exist in config
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=provider_config["temperature"],
            output_max_token_size=provider_config["output_max_token_size"]
        )
    
    def get_parsing_config(self, algorithm: str = "basic") -> ParsingConfig:
        """Get parsing configuration for specified algorithm"""
        
        algo_key = f"{algorithm}_algorithm"
        if "parsing" not in self.config:
            raise KeyError("No 'parsing' section found in config.json")
        
        if algo_key not in self.config["parsing"]:
            raise KeyError(f"Algorithm '{algorithm}' not found in config.json parsing section. Available algorithms: {list(self.config['parsing'].keys())}")
        
        algo_config = self.config["parsing"][algo_key]
        
        # Channel algorithm doesn't use batch_size/max_queue_size
        if algorithm == "channel":
            config = ParsingConfig(
                batch_size=0,  # Not used for channel algorithm
                max_queue_size=0,  # Not used for channel algorithm
                max_context_tokens=algo_config["max_context_tokens"],
                reserve_tokens=algo_config["reserve_tokens"]
            )
        else:
            # All other algorithms require basic settings
            config = ParsingConfig(
                batch_size=algo_config["batch_size"],
                max_queue_size=algo_config["max_queue_size"]
            )
            
            # Enhanced algorithm specific settings
            if algorithm == "enhanced":
                config.attention_sink_size = algo_config["attention_sink_size"]
                config.coherence_threshold = algo_config["coherence_threshold"]
                config.similarity_model = algo_config["similarity_model"]
        
        return config
    
    def get_output_config(self, algorithm: str = "basic") -> OutputConfig:
        """Get output configuration"""
        
        if "output" not in self.config:
            raise KeyError("No 'output' section found in config.json")
        
        output_config = self.config["output"]
        
        return OutputConfig(
            json_file=output_config["json_file"].format(algorithm=algorithm),
            markdown_file=output_config["markdown_file"].format(algorithm=algorithm),
            comparison_file=output_config["comparison_file"],
            include_full_conversations=output_config["include_full_conversations"],
            max_message_length=output_config["max_message_length"]
        )
    
    def list_available_models(self, provider: str = "openai") -> Dict[str, str]:
        """List available models for a provider"""
        if provider not in self.config["llm"]["providers"]:
            raise KeyError(f"Provider '{provider}' not found in config.json")
        provider_config = self.config["llm"]["providers"][provider]
        if "models" not in provider_config:
            raise KeyError(f"No models configuration found for provider '{provider}' in config.json")
        return provider_config["models"]
    
    def update_model(self, provider: str, model_type: str, model_name: str):
        """Update model configuration"""
        if provider not in self.config["llm"]["providers"]:
            self.config["llm"]["providers"][provider] = {"models": {}}
        
        if "models" not in self.config["llm"]["providers"][provider]:
            self.config["llm"]["providers"][provider]["models"] = {}
        
        self.config["llm"]["providers"][provider]["models"][model_type] = model_name
        self._save_config()
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get API keys from environment"""
        return {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {
            "config_file_exists": os.path.exists(self.config_file),
            "env_file_exists": os.path.exists(".env"),
            "api_keys": {},
            "available_providers": []
        }
        
        # Check API keys
        keys = self.get_api_keys()
        for provider, key in keys.items():
            status["api_keys"][provider] = "configured" if key else "missing"
            if key:
                status["available_providers"].append(provider)
        
        return status


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global config manager instance"""
    return config_manager


# Convenience functions
def get_llm_config(provider: Optional[str] = None, model_type: str = "default") -> LLMConfig:
    return config_manager.get_llm_config(provider, model_type)


def get_parsing_config(algorithm: str = "basic") -> ParsingConfig:
    return config_manager.get_parsing_config(algorithm)


def get_output_config(algorithm: str = "basic") -> OutputConfig:
    return config_manager.get_output_config(algorithm)


# Usage example and testing
if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("=== Configuration Validation ===")
    status = config.validate_config()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n=== LLM Configurations ===")
    for provider in ["openai", "anthropic", "gemini"]:
        try:
            llm_config = get_llm_config(provider)
            print(f"{provider}: {llm_config.model} (temp: {llm_config.temperature})")
        except Exception as e:
            print(f"{provider}: Error - {e}")
    
    print("\n=== Available Models ===")
    for provider in ["openai", "anthropic", "gemini"]:
        try:
            models = config.list_available_models(provider)
            print(f"{provider}: {models}")
        except Exception as e:
            print(f"{provider}: Error - {e}")
    
    print("\n=== Parsing Configurations ===")
    for algo in ["basic", "enhanced"]:
        parsing_config = get_parsing_config(algo)
        print(f"{algo}: batch_size={parsing_config.batch_size}, queue_size={parsing_config.max_queue_size}")
    
    print("\n=== Output Configurations ===")
    for algo in ["basic", "enhanced"]:
        output_config = get_output_config(algo)
        print(f"{algo}: {output_config.json_file}")