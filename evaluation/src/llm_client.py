"""
LLM Client wrapper to support both OpenAI and Alibaba Cloud Qwen APIs
"""
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Unified LLM client that supports OpenAI and Qwen"""
    
    def __init__(self, provider=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
        
        if self.provider == "qwen":
            self._init_qwen()
        else:
            self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        self.client = OpenAI()
        self.model = os.getenv("MODEL", "gpt-4o-mini")
    
    def _init_qwen(self):
        """Initialize Qwen client (compatible with OpenAI SDK)"""
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = os.getenv("MODEL", "qwen-plus")
    
    def chat_completion(self, messages, temperature=0, max_tokens=None, response_format=None):
        """Create a chat completion (compatible interface)"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        
        # Only add response_format for OpenAI, Qwen might not support it
        if response_format and self.provider == "openai":
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        return response
    
    def get_embedding(self, text, model=None):
        """Get embedding for text"""
        if self.provider == "qwen":
            # Qwen embeddings use text-embedding-v3
            embedding_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        else:
            embedding_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        response = self.client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
