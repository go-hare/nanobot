"""
向量生成器 - 为记忆生成向量表示

支持多种embedding生成方式：
1. OpenAI Embedding API
2. 本地模型（通过Sentence Transformers）
3. 自定义embedding provider
"""

from typing import List, Optional, Dict, Any
from loguru import logger
import asyncio

TAG = __name__


class EmbeddingGenerator:
    """向量生成器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化向量生成器
        
        Args:
            config: 配置字典，包含：
                - provider: "openai", "local", "custom"
                - model: 模型名称
                - api_key: API密钥（如果使用OpenAI）
                - base_url: API基础URL（可选）
                - dimension: 向量维度
        """
        self.config = config if config is not None else {}
        self.provider = self.config.get("provider", "dummy")
        self.model = self.config.get("model", "text-embedding-ada-002")
        self.dimension = self.config.get("dimension", 1536)
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url")
        
        # 初始化embedding client
        self.client = None
        self._initialize_client()
        
        logger.bind(tag=TAG).info(
            f"EmbeddingGenerator initialized with provider={self.provider}, "
            f"model={self.model}, dimension={self.dimension}"
        )
    
    def _initialize_client(self):
        """初始化embedding客户端"""
        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                logger.bind(tag=TAG).info("OpenAI embedding client initialized")
            except ImportError:
                logger.bind(tag=TAG).error("OpenAI package not installed, falling back to dummy embeddings")
                self.provider = "dummy"
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to initialize OpenAI client: {e}, falling back to dummy")
                self.provider = "dummy"
        
        elif self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.client = SentenceTransformer(self.model)
                logger.bind(tag=TAG).info(f"Local sentence-transformer model '{self.model}' loaded")
            except ImportError:
                logger.bind(tag=TAG).error("sentence-transformers not installed, falling back to dummy embeddings")
                self.provider = "dummy"
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to load local model: {e}, falling back to dummy")
                self.provider = "dummy"
        
        elif self.provider == "dummy":
            logger.bind(tag=TAG).warning("Using dummy embeddings (random vectors)")
        
        else:
            logger.bind(tag=TAG).warning(f"Unknown provider '{self.provider}', using dummy embeddings")
            self.provider = "dummy"
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        为文本生成向量表示
        
        Args:
            text: 输入文本
        
        Returns:
            向量表示（List[float]）
        """
        if not text or not text.strip():
            logger.bind(tag=TAG).warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * self.dimension
        
        try:
            if self.provider == "openai":
                return await self._generate_openai_embedding(text)
            elif self.provider == "local":
                return await self._generate_local_embedding(text)
            else:
                return self._generate_dummy_embedding(text)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to generate embedding: {e}, using dummy embedding")
            return self._generate_dummy_embedding(text)
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成向量表示（提高效率）
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表
        """
        if self.provider == "openai":
            return await self._generate_openai_embeddings_batch(texts)
        elif self.provider == "local":
            return await self._generate_local_embeddings_batch(texts)
        else:
            return [self._generate_dummy_embedding(text) for text in texts]
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """使用OpenAI API生成embedding"""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成OpenAI embeddings"""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
    
    async def _generate_local_embedding(self, text: str) -> List[float]:
        """使用本地模型生成embedding"""
        if self.client is None:
            raise RuntimeError("Local model not initialized")
        
        # SentenceTransformer的encode是同步的，在asyncio中运行
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.client.encode, text)
        return embedding.tolist()
    
    async def _generate_local_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成本地embeddings"""
        if self.client is None:
            raise RuntimeError("Local model not initialized")
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.client.encode, texts)
        return [emb.tolist() for emb in embeddings]
    
    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """生成虚拟embedding（用于测试）"""
        import hashlib
        import numpy as np
        
        # 使用文本的hash值作为随机种子，保证相同文本总是生成相同的向量
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        
        # 生成随机向量并归一化
        vec = np.random.randn(self.dimension)
        vec = vec / np.linalg.norm(vec)
        
        return vec.tolist()
