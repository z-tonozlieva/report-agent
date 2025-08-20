# free_llm_providers_langchain.py
"""
Groq LLM provider for the PM Reporting Tool using LangChain

This module provides Groq integration using LangChain for report generation.
"""

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.utils import convert_to_secret_str
from langchain_groq import ChatGroq

from core import LLMInterface


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
                    SystemMessage(
                        content="You are a helpful assistant that creates professional project management reports based on team updates."
                    ),
                    HumanMessage(content=prompt),
                ]
                response = self.llm.invoke(messages)
                return self.output_parser.invoke(response)
            else:
                # For completion models, use direct prompt
                response = self.llm.invoke(prompt)
                return response if isinstance(response, str) else str(response)

        except Exception as e:
            return f"Error generating response: {str(e)}"


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
        import os
        
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            print("Warning: GROQ_API_KEY not set. Set it as environment variable.")

            # Create a dummy LLM that returns error message
            class DummyLLM:
                def invoke(self, messages):
                    return "Error: Groq API key not configured. Please set GROQ_API_KEY environment variable."

            super().__init__(DummyLLM(), use_chat_model=True)
        else:
            print("Using Groq")
            llm = ChatGroq(
                model=model_name,
                temperature=0.7,
                max_tokens=1000,
                api_key=convert_to_secret_str(api_key),
            )
            super().__init__(llm, use_chat_model=True)


# Factory function to create LLM instances
def create_llm(model_name: str = "llama3-8b-8192") -> LLMInterface:
    """
    Factory function to create Groq LLM instance

    Args:
        model_name: Groq model name to use

    Returns:
        GroqLLM instance
    """
    return GroqLLM(model_name=model_name)


# Installation requirements
def get_installation_requirements():
    """
    Return installation requirements for Groq provider
    """
    return ["langchain-core", "langchain-groq"]


# Usage example
def get_usage_example():
    """
    Return usage example for Groq provider
    """
    return """
# Groq (free tier, very fast)
# 1. Sign up at https://console.groq.com/
# 2. Get API key and set: export GROQ_API_KEY=your_key
# 3. Install: pip install langchain-groq
from backend.free_llm_providers import create_llm
llm = create_llm()  # Uses default model
response = llm.generate_response("Create a project status report")
"""


if __name__ == "__main__":
    # Print usage example
    print("=== GROQ LLM PROVIDER WITH LANGCHAIN ===")
    print(get_usage_example())

    print("\n=== INSTALLATION REQUIREMENTS ===")
    requirements = get_installation_requirements()
    print(f"Dependencies: {' '.join(requirements)}")
