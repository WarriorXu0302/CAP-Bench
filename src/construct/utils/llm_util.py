import os
from typing import Optional

from loguru import logger
from openai import OpenAI


class LLMService:
    """LLM 服务工具类
    
    封装OpenAI兼容API的对话接口，支持自动从环境变量读取配置。
    """

    _DEFAULT_MODEL = "google/gemini-3-pro"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """初始化LLM服务
        
        Args:
            api_key: API密钥，默认从环境变量 OPENAI_API_KEY 读取
            base_url: API基础地址，默认从环境变量 OPENAI_BASE_URL 读取
            
        Raises:
            ValueError: 当缺少 API Key 或 Base URL 时抛出。

        Example:
            >>> llm = LLMService()
            >>> llm = LLMService(api_key="sk-xxx", base_url="https://api.example.com")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key or not self.base_url:
            raise ValueError(
                "LLMService 初始化失败：缺少 OPENAI_API_KEY 或 OPENAI_BASE_URL，"
                "请设置环境变量或在初始化时传入参数"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None
    ) -> str:
        """chat接口调用
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            model: 使用的模型名称

        Returns:
            模型回复内容
            
        Raises:
            RuntimeError: 当调用 LLM API 失败时抛出
            
        Example:
            >>> llm = LLMService()
            >>> response = llm.chat(
            ...     system_prompt="你是一个乐于助人的AI助手。",
            ...     user_prompt="请用一句话解释人工智能是什么。",
            ...     model="Qwen/Qwen2.5-72B-Instruct-128K"
            ... )
            >>> print(response)
        """
        model = model or self._DEFAULT_MODEL

        try:
            # 构造消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用 API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=32768
            )
            
            # 返回结果
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.exception(f"LLM 对话失败: {e}")
            raise RuntimeError(f"调用 LLM API 失败: {e}") from e


# 模块内部单元测试
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    llm = LLMService()

    # 测试chat接口
    print("\n=== 基本对话测试 ===")
    system_prompt = "你是一个乐于助人的AI助手。"
    user_prompt = "请用一句话解释人工智能是什么。"
    response = llm.chat(system_prompt, user_prompt)
    print(f"回复: {response}")