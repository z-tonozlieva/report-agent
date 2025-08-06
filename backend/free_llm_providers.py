# free_llm_providers_langchain.py
"""
Free LLM providers for the PM Reporting Tool using LangChain

This module provides integrations with various free LLM APIs using LangChain
for better standardization and error handling.
"""

# LangChain imports
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.utils import convert_to_secret_str

# For local Hugging Face models - DISABLED for Render optimization
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.pipelines import pipeline

from .mock_llm import LLMInterface  # Use the single source of truth for LLMInterface

class LangChainLLMWrapper(LLMInterface):
    """Base wrapper for LangChain LLM implementations"""
    
    def __init__(self, llm, use_chat_model=False):
        self.llm = llm
        self.use_chat_model = use_chat_model
        self.output_parser = StrOutputParser()
    
    def generate_response(self, prompt: str) -> str:
        try:
            if self.use_chat_model:
                # For chat models, use message format
                messages = [
                    SystemMessage(content="You are a helpful assistant that creates professional project management reports based on team updates."),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                return self.output_parser.invoke(response)
            else:
                # For completion models, use direct prompt
                response = self.llm.invoke(prompt)
                return response if isinstance(response, str) else str(response)
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

class OllamaLLM(LangChainLLMWrapper):
    """
    Ollama LLM integration using LangChain - completely free, runs locally
    
    Installation:
    1. Install Ollama: https://ollama.ai/
    2. Run: ollama pull llama2 (or any other model)
    3. Start Ollama server: ollama serve
    4. Install: pip install langchain-community
    """
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        try:
            # Try chat model first (newer, better for conversations)
            llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0.7,
            )
            super().__init__(llm, use_chat_model=True)
        except Exception:
            # Fallback to completion model
            llm = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=0.7,
            )
            super().__init__(llm, use_chat_model=False)

class HuggingFaceLLM(LangChainLLMWrapper):
    """
    Hugging Face LLM integration using LangChain - DISABLED for Render optimization
    
    This provider is disabled to reduce memory usage on Render free tier.
    Use Groq instead for cloud-based inference.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        # Create a dummy LLM that returns error message
        class DummyLLM:
            def invoke(self, prompt):
                return "HuggingFace LLM is disabled to optimize for Render free tier. Please use Groq provider instead."
        
        super().__init__(DummyLLM(), use_chat_model=False)

class GroqLLM(LangChainLLMWrapper):
    """
    Groq API using LangChain - free tier available with very fast inference
    
    Setup:
    1. Sign up at https://console.groq.com/
    2. Get free API key
    3. Set environment variable: export GROQ_API_KEY=your_key
    4. Install: pip install langchain-groq
    """
    
    def __init__(self, model_name: str = "llama3-8b-8192"):
        from .config import Config
        api_key = Config.get_groq_api_key()
        
        if not api_key:
            print("Warning: GROQ_API_KEY not set. Set it as environment variable.")
            # Create a dummy LLM that returns error message
            class DummyLLM:
                def invoke(self, messages):
                    return "Error: Groq API key not configured. Please set GROQ_API_KEY environment variable."
            
            super().__init__(DummyLLM(), use_chat_model=True)
        else:
            print(f"Using Groq")
            llm = ChatGroq(
                model=model_name,
                temperature=0.7,
                max_tokens=1000,
                api_key=convert_to_secret_str(api_key)
            )
            super().__init__(llm, use_chat_model=True)

# class OpenAICompatibleLLM(LangChainLLMWrapper):
#     """
#     Generic OpenAI-compatible API client using LangChain
    
#     Works with:
#     - OpenAI API (paid)
#     - LocalAI (free, self-hosted)
#     - Oobabooga Text Generation WebUI (free, self-hosted)
#     - vLLM (free, self-hosted)
#     """
    
#     def __init__(self, base_url: str, api_key: str, model_name: str = "gpt-3.5-turbo"):
#         try:
#             llm = ChatOpenAI(
#                 openai_api_key=api_key,
#                 openai_api_base=base_url,
#                 model_name=model_name,
#                 temperature=0.7,
#                 max_tokens=1000
#             )
#             super().__init__(llm, use_chat_model=True)
#         except Exception as e:
#             # Create a dummy LLM that returns error message
#             class DummyLLM:
#                 def invoke(self, messages):
#                     return f"Error connecting to OpenAI-compatible API: {str(e)}"
            
#             super().__init__(DummyLLM(), use_chat_model=True)

# class AnthropicLLM(LangChainLLMWrapper):
#     """
#     Anthropic Claude API using LangChain
    
#     Setup:
#     1. Sign up at https://console.anthropic.com/
#     2. Get API key
#     3. Set environment variable: export ANTHROPIC_API_KEY=your_key
#     4. Install: pip install langchain-anthropic
#     """
    
#     def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
#         api_key = os.getenv("ANTHROPIC_API_KEY")
        
#         if not api_key:
#             print("Warning: ANTHROPIC_API_KEY not set. Set it as environment variable.")
#             # Create a dummy LLM that returns error message
#             class DummyLLM:
#                 def invoke(self, messages):
#                     return "Error: Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
            
#             super().__init__(DummyLLM(), use_chat_model=True)
#         else:
#             try:
#                 from langchain_anthropic import ChatAnthropic
#                 llm = ChatAnthropic(
#                     anthropic_api_key=api_key,
#                     model=model_name,
#                     temperature=0.7,
#                     max_tokens=1000
#                 )
#                 super().__init__(llm, use_chat_model=True)
#             except ImportError:
#                 # Create a dummy LLM that returns error message
#                 class DummyLLM:
#                     def invoke(self, messages):
#                         return "Error: Please install langchain-anthropic: pip install langchain-anthropic"
                
#                 super().__init__(DummyLLM(), use_chat_model=True)

# Factory function to create LLM instances
def create_llm(provider: str = "mock", **kwargs) -> LLMInterface:
    """
    Factory function to create LLM instances using LangChain
    
    Args:
        provider: One of 'mock', 'ollama', 'huggingface', 'groq', 'openai_compatible', 'anthropic'
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
    
    # elif provider == "openai_compatible":
    #     base_url = kwargs.get("base_url", "https://api.openai.com")
    #     api_key = kwargs.get("api_key", "")
    #     model_name = kwargs.get("model_name", "gpt-3.5-turbo")
    #     return OpenAICompatibleLLM(base_url=base_url, api_key=api_key, model_name=model_name)
    
    # elif provider == "anthropic":
    #     model_name = kwargs.get("model_name", "claude-3-sonnet-20240229")
    #     return AnthropicLLM(model_name=model_name)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Installation requirements
def get_installation_requirements():
    """
    Return installation requirements for different providers
    """
    requirements = {
        "base": ["langchain-core", "langchain-community"],
        "ollama": ["langchain-community"],
        "huggingface": ["langchain-community", "transformers", "torch"],
        "groq": ["langchain-groq"],
        "openai_compatible": ["langchain-openai"],
        "anthropic": ["langchain-anthropic"]
    }
    
    return requirements

# Usage examples
def get_usage_examples():
    """
    Return usage examples for different LLM providers using LangChain
    """
    examples = {
        "installation": """
# Base installation
pip install langchain-core langchain-community

# For specific providers, install additional packages:
pip install langchain-groq          # For Groq
pip install langchain-openai        # For OpenAI-compatible APIs
pip install langchain-anthropic     # For Anthropic
pip install transformers torch      # For local Hugging Face models
""",
        
        "mock": """
# Mock LLM (no setup required)
from free_llm_providers_langchain import create_llm
llm = create_llm("mock")
response = llm.generate_response("Create a project status report")
""",
        
        "ollama": """
# Ollama (completely free, runs locally)
# 1. Install Ollama: https://ollama.ai/
# 2. Run: ollama pull llama2
# 3. Start: ollama serve
from free_llm_providers_langchain import create_llm
llm = create_llm("ollama", model_name="llama2")
response = llm.generate_response("Create a project status report")
""",
        
        "groq": """
# Groq (free tier, very fast)
# 1. Sign up at https://console.groq.com/
# 2. Get API key and set: export GROQ_API_KEY=your_key
# 3. Install: pip install langchain-groq
from free_llm_providers_langchain import create_llm
llm = create_llm("groq")
response = llm.generate_response("Create a project status report")
""",
        
        "huggingface": """
# Hugging Face (free, runs locally)
# 1. Install: pip install transformers torch
from free_llm_providers_langchain import create_llm
llm = create_llm("huggingface", model_name="microsoft/DialoGPT-medium")
response = llm.generate_response("Create a project status report")
""",
        
        "anthropic": """
# Anthropic Claude (paid, but excellent quality)
# 1. Sign up at https://console.anthropic.com/
# 2. Get API key and set: export ANTHROPIC_API_KEY=your_key
# 3. Install: pip install langchain-anthropic
from free_llm_providers_langchain import create_llm
llm = create_llm("anthropic")
response = llm.generate_response("Create a project status report")
""",
        
        "openai_compatible": """
# OpenAI-compatible APIs (LocalAI, vLLM, etc.)
# Install: pip install langchain-openai
from free_llm_providers_langchain import create_llm
llm = create_llm("openai_compatible", 
                base_url="http://localhost:8080/v1",
                api_key="your_key",
                model_name="your_model")
response = llm.generate_response("Create a project status report")
"""
    }
    
    return examples

if __name__ == "__main__":
    # Print usage examples
    print("=== FREE LLM PROVIDERS WITH LANGCHAIN ===")
    examples = get_usage_examples()
    
    for provider, example in examples.items():
        print(f"\n{provider.upper()}:")
        print(example)
    
    print("\n=== INSTALLATION REQUIREMENTS ===")
    requirements = get_installation_requirements()
    for provider, deps in requirements.items():
        print(f"{provider}: {' '.join(deps)}")