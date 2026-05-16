import os
from typing import Optional

from loguru import logger
from openai import OpenAI


class LLMService:
    """LLM service utility class.

    Thin wrapper around an OpenAI-compatible chat completion API.
    Credentials and the default model are read from environment
    variables by default, so the same code works against OpenAI,
    Azure OpenAI, vLLM, LiteLLM proxies, etc.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the LLM service.

        Args:
            api_key: API key. Defaults to ``OPENAI_API_KEY``.
            base_url: API base URL. Defaults to ``OPENAI_BASE_URL``.
            default_model: Default model id used when ``chat()`` is
                called without a ``model`` argument. Defaults to
                ``OPENAI_MODEL``.

        Raises:
            ValueError: If the API key, base URL, or default model is
                missing.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.default_model = default_model or os.getenv("OPENAI_MODEL")

        if not self.api_key or not self.base_url:
            raise ValueError(
                "LLMService initialization failed: missing OPENAI_API_KEY or "
                "OPENAI_BASE_URL — set them in the environment or pass them "
                "explicitly."
            )
        if not self.default_model:
            raise ValueError(
                "LLMService initialization failed: missing model id — set "
                "OPENAI_MODEL in the environment or pass default_model "
                "explicitly."
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
        model = model or self.default_model

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