import os
from typing import Optional

from loguru import logger
from openai import OpenAI


class LLMService:
    """LLM Service utility class
    
    Encapsulates OpenAI-compatible API conversation interface with automatic environment variable configuration reading.
    """

    _DEFAULT_MODEL = "google/gemini-3-pro"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """Initialize LLM service.
        
        Args:
            api_key: API key, reads from OPENAI_API_KEY environment variable by default
            base_url: API base URL, reads from OPENAI_BASE_URL environment variable by default
            
        Raises:
            ValueError: Raised when API Key or Base URL is missing.

        Example:
            >>> llm = LLMService()
            >>> llm = LLMService(api_key="sk-xxx", base_url="https://api.example.com")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key or not self.base_url:
            raise ValueError(
                "LLMService initialization failed: missing OPENAI_API_KEY or OPENAI_BASE_URL, "
                "please set environment variables or pass parameters during initialization"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None
    ) -> str:
        """Chat interface call.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            model: Model name to use

        Returns:
            Model response content
            
        Raises:
            RuntimeError: Raised when LLM API call fails
            
        Example:
            >>> llm = LLMService()
            >>> response = llm.chat(
            ...     system_prompt="You are a helpful AI assistant.",
            ...     user_prompt="Please explain what artificial intelligence is in one sentence.",
            ...     model="Qwen/Qwen2.5-72B-Instruct-128K"
            ... )
            >>> print(response)
        """
        model = model or self._DEFAULT_MODEL

        try:
            # Construct message list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=32768
            )
            
            # Return result
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.exception(f"LLM conversation failed: {e}")
            raise RuntimeError(f"LLM API call failed: {e}") from e


# Module internal unit test
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    llm = LLMService()

    # Test chat interface
    print("\n=== Basic Conversation Test ===")
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "Please explain what artificial intelligence is in one sentence."
    response = llm.chat(system_prompt, user_prompt)
    print(f"Response: {response}")