# free_llm_providers.py
"""
Free LLM providers for the PM Reporting Tool

This module provides integrations with various free LLM APIs that can be used
instead of the mock LLM for more realistic testing.
"""

import requests
import json
import os
from typing import Optional
from .mock_llm import LLMInterface

class OllamaLLM(LLMInterface):
    """
    Ollama LLM integration - completely free, runs locally
    
    Installation:
    1. Install Ollama: https://ollama.ai/
    2. Run: ollama pull llama2 (or any other model)
    3. Start Ollama server: ollama serve
    """
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def generate_response(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=200)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "Error: No response generated")
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running locally."
        except Exception as e:
            return f"Error generating response: {str(e)}"

class HuggingFaceLLM(LLMInterface):
    """
    Hugging Face Inference API - free tier available
    
    Setup:
    1. Sign up at https://huggingface.co/
    2. Get free API token from https://huggingface.co/settings/tokens
    3. Set environment variable: export HUGGINGFACE_API_KEY=your_token
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not set. Set it as environment variable.")
    
    def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: Hugging Face API key not configured. Please set HUGGINGFACE_API_KEY environment variable."
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Error: No text generated")
            else:
                return "Error: Unexpected response format"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Hugging Face API: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

class GroqLLM(LLMInterface):
    """
    Groq API - free tier available with very fast inference
    
    Setup:
    1. Sign up at https://console.groq.com/
    2. Get free API key
    3. Set environment variable: export GROQ_API_KEY=your_key
    """
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not set. Set it as environment variable.")
    
    def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: Groq API key not configured. Please set GROQ_API_KEY environment variable."
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates professional project management reports based on team updates."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Groq API: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

class OpenAICompatibleLLM(LLMInterface):
    """
    Generic OpenAI-compatible API client
    
    Works with:
    - OpenAI API (paid)
    - LocalAI (free, self-hosted)
    - Oobabooga Text Generation WebUI (free, self-hosted)
    - vLLM (free, self-hosted)
    """
    
    def __init__(self, base_url: str, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"{base_url}/v1/chat/completions"
    
    def generate_response(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates professional project management reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to API: {str(e)}"
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Factory function to create LLM instances
def create_llm(provider: str = "mock", **kwargs) -> LLMInterface:
    """
    Factory function to create LLM instances
    
    Args:
        provider: One of 'mock', 'ollama', 'huggingface', 'groq', 'openai_compatible'
        **kwargs: Additional arguments specific to each provider
    
    Returns:
        LLMInterface instance
    """
    if provider == "mock":
        from mock_llm import MockLLM
        return MockLLM()
    
    elif provider == "ollama":
        model_name = kwargs.get("model_name", "llama2")
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaLLM(model_name=model_name, base_url=base_url)
    
    elif provider == "huggingface":
        model_name = kwargs.get("model_name", "microsoft/DialoGPT-medium")
        return HuggingFaceLLM(model_name=model_name)
    
    elif provider == "groq":
        model_name = kwargs.get("model_name", "llama3-8b-8192")
        return GroqLLM(model_name=model_name)
    
    elif provider == "openai_compatible":
        base_url = kwargs.get("base_url", "https://api.openai.com")
        api_key = kwargs.get("api_key", "")
        model_name = kwargs.get("model_name", "gpt-3.5-turbo")
        return OpenAICompatibleLLM(base_url=base_url, api_key=api_key, model_name=model_name)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Usage examples
def get_usage_examples():
    """
    Return usage examples for different LLM providers
    """
    examples = {
        "mock": """
# Mock LLM (no setup required)
from free_llm_providers import create_llm
llm = create_llm("mock")
""",
        
        "ollama": """
# Ollama (completely free, runs locally)
# 1. Install Ollama: https://ollama.ai/
# 2. Run: ollama pull llama2
# 3. Start: ollama serve
from free_llm_providers import create_llm
llm = create_llm("ollama", model_name="llama2")
""",
        
        "groq": """
# Groq (free tier, very fast)
# 1. Sign up at https://console.groq.com/
# 2. Get API key and set: export GROQ_API_KEY=your_key
from free_llm_providers import create_llm
llm = create_llm("groq")
""",
        
        "huggingface": """
# Hugging Face (free tier available)
# 1. Sign up at https://huggingface.co/
# 2. Get token and set: export HUGGINGFACE_API_KEY=your_token
from free_llm_providers import create_llm
llm = create_llm("huggingface")
""",
        
        "openai_compatible": """
# OpenAI-compatible APIs (LocalAI, etc.)
from free_llm_providers import create_llm
llm = create_llm("openai_compatible", 
                base_url="http://localhost:8080",
                api_key="your_key",
                model_name="your_model")
"""
    }
    
    return examples

if __name__ == "__main__":
    # Print usage examples
    print("=== FREE LLM PROVIDERS ===")
    examples = get_usage_examples()
    
    for provider, example in examples.items():
        print(f"\n{provider.upper()}:")
        print(example)